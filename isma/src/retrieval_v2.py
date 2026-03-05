"""
ISMA Retrieval V2 — Document-level canonical memory search.

Searches ISMA_Quantum_v2 (one object per content_hash) using:
  - Named vector search: raw (content), rosetta (semantic summary)
  - BM25F text search with field weighting
  - RRF (Reciprocal Rank Fusion) of raw + rosetta + BM25 results
  - Neural reranking via Qwen3-Reranker-8B

Falls back to v1 (ISMARetrieval) if v2 class doesn't exist.

Usage:
    from isma.src.retrieval_v2 import ISMARetrievalV2

    r = ISMARetrievalV2()
    result = r.search("consciousness emergence", top_k=10)
    result = r.hybrid_search("Sacred Trust threshold", top_k=10)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter

from isma.src.retrieval import (
    EMBEDDING_MODEL,
    EMBEDDING_URL,
    WEAVIATE_URL,
    SearchResult,
    TileResult,
    _get_embedding,
)

log = logging.getLogger(__name__)

V2_CLASS = "ISMA_Quantum_v2"
V1_CLASS = "ISMA_Quantum"

# Properties to return from V1 tile searches (needed for TileResult + RRF fusion)
V1_TILE_PROPS = (
    "content content_hash platform source_type source_file session_id document_id "
    "loaded_at scale tile_index token_count hmm_enriched rosetta_summary "
    "dominant_motifs hmm_phi hmm_trust"
)

# Connection-pooled session for Weaviate GraphQL queries
_wv_session = requests.Session()
_wv_session.mount("http://", HTTPAdapter(pool_connections=50, pool_maxsize=50))
_wv_session.mount("https://", HTTPAdapter(pool_connections=50, pool_maxsize=50))

# Properties to fetch from v2 objects
V2_PROPERTIES = [
    "content", "content_hash", "platform", "source_type", "source_file",
    "session_id", "document_id", "loaded_at",
    "rosetta_summary", "motif_annotations", "dominant_motifs",
    "hmm_enriched", "hmm_phi", "hmm_trust", "hmm_enriched_at",
    "tile_count_512", "tile_count_2048", "tile_count_4096", "total_tokens",
    "tile_ids_512", "tile_ids_2048", "tile_ids_4096", "rosetta_tile_id",
]
V2_PROPS_STR = " ".join(V2_PROPERTIES)


def _escape_gql(s: str) -> str:
    """Escape a string for embedding in a GraphQL value literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _graphql(query: str) -> dict:
    """Execute a GraphQL query using connection-pooled session.

    Raises ConnectionError/Timeout on infrastructure failures so callers
    can surface them as 5xx rather than returning empty results.
    """
    try:
        r = _wv_session.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            timeout=30,
        )
        result = r.json()
        # Weaviate returns HTTP 200 with {"data": null, "errors": [...]} on invalid GQL syntax.
        # data.get("data", {}) returns None (key exists, value is null) → AttributeError on .get().
        if result.get("data") is None:
            if result.get("errors"):
                log.warning("GraphQL errors: %s", result["errors"])
            return {}
        return result
    except (requests.ConnectionError, requests.Timeout) as e:
        log.error("Weaviate connection failure: %s", e)
        raise
    except Exception as e:
        log.warning("GraphQL error: %s", e)
        return {}


def _v2_to_tile(obj: dict, score: float = 0.0) -> TileResult:
    """Convert a v2 object to a TileResult for API compatibility."""
    return TileResult(
        content=obj.get("content", ""),
        content_hash=obj.get("content_hash", ""),
        platform=obj.get("platform", ""),
        source_type=obj.get("source_type", ""),
        source_file=obj.get("source_file", ""),
        session_id=obj.get("session_id", ""),
        document_id=obj.get("document_id", ""),
        loaded_at=obj.get("loaded_at", ""),
        scale="canonical",  # v2 objects are document-level
        tile_id="",
        token_count=obj.get("total_tokens") or 0,
        score=score,
        hmm_enriched=obj.get("hmm_enriched", False),
        rosetta_summary=obj.get("rosetta_summary") or "",
        dominant_motifs=obj.get("dominant_motifs") or [],
        hmm_phi=obj.get("hmm_phi") or 0.0,
        hmm_trust=obj.get("hmm_trust") or 0.0,
    )


class ISMARetrievalV2:
    """V2 retrieval using canonical memory objects."""

    def __init__(self):
        self._v2_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if the v2 class exists in Weaviate."""
        if self._v2_available is not None:
            return self._v2_available
        try:
            r = requests.get(f"{WEAVIATE_URL}/v1/schema/{V2_CLASS}", timeout=5)
            self._v2_available = r.status_code == 200
        except Exception:
            self._v2_available = False
        return self._v2_available

    def stats(self) -> dict:
        """Get v2 collection statistics."""
        data = _graphql(
            f"{{ Aggregate {{ {V2_CLASS} {{ meta {{ count }} }} }} }}"
        )
        total = (
            data.get("data", {})
            .get("Aggregate", {})
            .get(V2_CLASS, [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )

        # Count enriched
        data_e = _graphql(
            f'{{ Aggregate {{ {V2_CLASS}(where: {{ path: ["hmm_enriched"] '
            f'operator: Equal valueBoolean: true }}) {{ meta {{ count }} }} }} }}'
        )
        enriched = (
            data_e.get("data", {})
            .get("Aggregate", {})
            .get(V2_CLASS, [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )

        return {
            "v2_total": total,
            "v2_enriched": enriched,
            "v2_available": self.is_available(),
        }

    # ── Vector Search ───────────────────────────────────────────

    def search_raw(
        self,
        query: str,
        top_k: int = 10,
        **filters,
    ) -> List[Tuple[dict, float]]:
        """Search using the raw (content_512) named vector."""
        embedding = _get_embedding(query)
        if not embedding:
            return []

        filter_clause = self._build_filter(**filters)
        vector_str = str(embedding)

        q = (
            f"{{ Get {{ {V2_CLASS}("
            f"nearVector: {{ vector: {vector_str}, targetVectors: [\"raw\"] }}"
            f" limit: {top_k}"
            f"{filter_clause}"
            f") {{ {V2_PROPS_STR} _additional {{ score distance }} }} }} }}"
        )

        data = _graphql(q)
        results = data.get("data", {}).get("Get", {}).get(V2_CLASS, [])

        return [
            (obj, float(obj.get("_additional", {}).get("score") or 0))
            for obj in results
        ]

    def search_rosetta(
        self,
        query: str,
        top_k: int = 10,
        vector: Optional[list] = None,
        **filters,
    ) -> List[Tuple[dict, float]]:
        """Search using the rosetta (summary) named vector."""
        embedding = vector or _get_embedding(query)
        if not embedding:
            return []

        filter_clause = self._build_filter(**filters)
        vector_str = str(embedding)

        q = (
            f"{{ Get {{ {V2_CLASS}("
            f"nearVector: {{ vector: {vector_str}, targetVectors: [\"rosetta\"] }}"
            f" limit: {top_k}"
            f"{filter_clause}"
            f") {{ {V2_PROPS_STR} _additional {{ score distance }} }} }} }}"
        )

        data = _graphql(q)
        results = data.get("data", {}).get("Get", {}).get(V2_CLASS, [])

        return [
            (obj, float(obj.get("_additional", {}).get("score") or 0))
            for obj in results
        ]

    # ── BM25 Search ─────────────────────────────────────────────

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        **filters,
    ) -> List[Tuple[dict, float]]:
        """BM25F text search with field weighting.

        Weaviate BM25 properties weight is applied via the query.
        rosetta_summary and motif_annotations get boosted.
        """
        # Escape query for GraphQL (backslash first, then quote, then newlines)
        safe_query = query.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
        filter_clause = self._build_filter(**filters)

        q = (
            f"{{ Get {{ {V2_CLASS}("
            f'bm25: {{ query: "{safe_query}" '
            f'properties: ["rosetta_summary^3", "motif_annotations^2", "content"] }}'
            f" limit: {top_k}"
            f"{filter_clause}"
            f") {{ {V2_PROPS_STR} _additional {{ score }} }} }} }}"
        )

        data = _graphql(q)
        results = data.get("data", {}).get("Get", {}).get(V2_CLASS, [])

        return [
            (obj, float(obj.get("_additional", {}).get("score") or 0))
            for obj in results
        ]

    # ── V1 Tile Search (Option E paths) ─────────────────────────

    def search_v1_bm25(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[Tuple[dict, float]]:
        """BM25 search on ISMA_Quantum search_512 tiles (full tile text, no truncation).

        This is the Option E replacement for the broken V2 BM25, which was limited
        to the first tile's text (2048 chars). V1 tiles contain the real passage text.
        """
        safe_query = (
            query.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", " ")
            .replace("\r", " ")
        )
        scale_filter = '{ path: ["scale"], operator: Equal, valueText: "search_512" }'
        q = (
            f'{{ Get {{ {V1_CLASS}('
            f'bm25: {{ query: "{safe_query}" properties: ["content^2", "rosetta_summary"] }}'
            f', where: {scale_filter}'
            f' limit: {top_k}'
            f') {{ {V1_TILE_PROPS} _additional {{ score }} }} }} }}'
        )
        data = _graphql(q)
        results = data.get("data", {}).get("Get", {}).get(V1_CLASS, []) or []
        return [
            (obj, float(obj.get("_additional", {}).get("score") or 0))
            for obj in results
        ]

    def search_v1_vector(
        self,
        query: str,
        top_k: int = 30,
        vector: Optional[list] = None,
    ) -> List[Tuple[dict, float]]:
        """NearVector search on ISMA_Quantum search_512 tiles.

        This is the Option E replacement for the broken V2 raw vector, which was
        embedded from the first tile only (2048 chars). V1 tile vectors represent
        the actual passage content.
        """
        embedding = vector or _get_embedding(query)
        if not embedding:
            return []
        vector_str = str(embedding)
        scale_filter = '{ path: ["scale"], operator: Equal, valueText: "search_512" }'
        q = (
            f'{{ Get {{ {V1_CLASS}('
            f'nearVector: {{ vector: {vector_str} }}'
            f', where: {scale_filter}'
            f' limit: {top_k}'
            f') {{ {V1_TILE_PROPS} _additional {{ score distance }} }} }} }}'
        )
        data = _graphql(q)
        results = data.get("data", {}).get("Get", {}).get(V1_CLASS, []) or []
        return [
            # nearVector returns distance (lower=better), convert to score (1-distance)
            (obj, 1.0 - float(obj.get("_additional", {}).get("distance") or 1.0))
            for obj in results
        ]

    def _v1_tile_to_obj(self, tile: dict) -> dict:
        """Convert a V1 ISMA_Quantum tile result to V2-compatible obj dict for RRF fusion."""
        return {
            "content": tile.get("content", ""),
            "content_hash": tile.get("content_hash", ""),
            "platform": tile.get("platform", ""),
            "source_type": tile.get("source_type", ""),
            "source_file": tile.get("source_file", ""),
            "session_id": tile.get("session_id", ""),
            "document_id": tile.get("document_id", ""),
            "loaded_at": tile.get("loaded_at", ""),
            "rosetta_summary": tile.get("rosetta_summary", "") or "",
            "dominant_motifs": tile.get("dominant_motifs") or [],
            "hmm_enriched": tile.get("hmm_enriched", False),
            "hmm_phi": tile.get("hmm_phi") or 0.0,
            "hmm_trust": tile.get("hmm_trust") or 0.0,
            "total_tokens": tile.get("token_count") or 0,
            "tile_ids_512": [], "tile_ids_2048": [], "tile_ids_4096": [],
            "tile_count_512": 0, "tile_count_2048": 0, "tile_count_4096": 0,
            "motif_annotations": "",
        }

    # ── Hybrid Search (RRF Fusion) ──────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        **filters,
    ) -> SearchResult:
        """Option E hybrid search: V2 rosetta + V1 BM25 + V1 nearVector via parallel RRF.

        V2 raw nearVector and V2 BM25 are disabled — they were trained on the first
        search_512 tile only (2048 chars), causing -28pt exact recall regression.

        Paths (run in parallel):
          - V2 rosetta nearVector: semantic summary signal (conceptual strength)
          - V1 search_512 BM25: full passage text coverage (exact/temporal strength)
          - V1 search_512 nearVector: per-tile embeddings (local evidence recovery)

        Results fused at content_hash level via RRF (k=60).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        t0 = time.monotonic()
        fetch_k = max(top_k * 3, 30)

        # Embed query ONCE — shared by rosetta nearVector + V1 nearVector paths.
        # BM25 path needs no embedding. Embedding twice in parallel wastes server capacity
        # and doubles latency under load (ColBERT ingest, etc.).
        query_vector = _get_embedding(query)

        # Run all three paths in parallel
        futures_map = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures_map["rosetta"] = executor.submit(
                self.search_rosetta, query, top_k=fetch_k, vector=query_vector, **filters
            )
            futures_map["v1_bm25"] = executor.submit(
                self.search_v1_bm25, query, top_k=fetch_k
            )
            futures_map["v1_vector"] = executor.submit(
                self.search_v1_vector, query, top_k=fetch_k, vector=query_vector
            )
            results_by_path = {}
            for name, fut in futures_map.items():
                try:
                    results_by_path[name] = fut.result(timeout=25)
                except Exception as e:
                    log.warning("Search path %s failed: %s", name, e)
                    results_by_path[name] = []

        # RRF fusion at content_hash level (k=60 standard)
        # Two separate maps:
        #   v2_meta_map: V2 doc objects for metadata (rosetta_summary, dominant_motifs, etc.)
        #   v1_content_map: best V1 passage tile for each hash (for content — the specific passage)
        # V1 tile content is passage-level (~512 tokens), fitting within the 2000-char recall window.
        # V2 doc content is full concatenated document — evidence buried deep, fails recall[:2000].
        k = 60
        rrf_scores: Dict[str, float] = {}
        v2_meta_map: Dict[str, dict] = {}
        v1_content_map: Dict[str, dict] = {}  # hash → best V1 tile (first hit = highest scored)

        for rank, (obj, _score) in enumerate(results_by_path.get("rosetta", [])):
            ch = obj.get("content_hash", "")
            if ch:
                rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                v2_meta_map[ch] = obj  # V2 obj: rosetta_summary, dominant_motifs, etc.

        for path in ("v1_bm25", "v1_vector"):
            for rank, (tile, _score) in enumerate(results_by_path.get(path, [])):
                ch = tile.get("content_hash", "")
                if ch:
                    rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                    if ch not in v1_content_map:
                        v1_content_map[ch] = tile  # First (highest scored) V1 tile for this doc

        # Sort by RRF score
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda ch: rrf_scores[ch], reverse=True)

        tiles = []
        for ch in sorted_hashes[:top_k]:
            v2_meta = v2_meta_map.get(ch)
            v1_tile = v1_content_map.get(ch)

            if v1_tile:
                # Build from V1 tile content + V2 metadata overlay
                obj = self._v1_tile_to_obj(v1_tile)
                if v2_meta:
                    # Overlay V2 metadata (richer: rosetta_summary, motifs, hmm_*)
                    obj["rosetta_summary"] = v2_meta.get("rosetta_summary", "") or obj["rosetta_summary"]
                    obj["dominant_motifs"] = v2_meta.get("dominant_motifs") or obj["dominant_motifs"]
                    obj["hmm_phi"] = v2_meta.get("hmm_phi") or obj["hmm_phi"]
                    obj["hmm_trust"] = v2_meta.get("hmm_trust") or obj["hmm_trust"]
                    obj["hmm_enriched"] = v2_meta.get("hmm_enriched", False) or obj["hmm_enriched"]
                    obj["motif_annotations"] = v2_meta.get("motif_annotations", "") or obj["motif_annotations"]
            elif v2_meta:
                # Only found via rosetta — use V2 obj (content will be concatenated but rosetta helps)
                obj = v2_meta
            else:
                continue

            tile_result = _v2_to_tile(obj, score=rrf_scores[ch])
            tiles.append(tile_result)

        elapsed_ms = (time.monotonic() - t0) * 1000
        total_tokens = sum(t.token_count for t in tiles)

        return SearchResult(
            query=query,
            tiles=tiles,
            total_tokens=total_tokens,
            search_time_ms=elapsed_ms,
        )

    # ── Hybrid Search with Reranker ─────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        query_type: str = "default",
        instruction: str = "",
        **filters,
    ) -> Dict[str, Any]:
        """Full hybrid search with neural reranking.

        1. RRF fusion of raw + rosetta + BM25
        2. Neural reranker on fused candidates
        3. Return top_k results
        """
        t0 = time.monotonic()

        # Get more candidates for reranking
        fetch_k = top_k * 3 if rerank else top_k
        search_result = self.search(query, top_k=fetch_k, **filters)

        # Neural reranking
        if rerank and search_result.tiles:
            try:
                from isma.src.reranker import get_reranker
                reranker = get_reranker()
                if reranker.is_available():
                    reranked = reranker.rerank(
                        query, search_result.tiles,
                        instruction=instruction,
                        query_type=query_type,
                    )
                    search_result = SearchResult(
                        query=query,
                        tiles=reranked[:top_k],
                        total_tokens=sum(t.token_count for t in reranked[:top_k]),
                        search_time_ms=search_result.search_time_ms,
                    )
            except Exception as e:
                log.debug("Reranker unavailable: %s", e)

        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "query": query,
            "tiles": search_result.tiles[:top_k],
            "total_tokens": search_result.total_tokens,
            "search_time_ms": elapsed_ms,
            "hmm_reranked": rerank,
            "version": "v2",
        }

    # ── Adaptive Search ─────────────────────────────────────────

    def adaptive_search(
        self,
        query: str,
        top_k: int = 10,
        expand_graph: bool = True,
        graph_depth: int = 2,
        **filters,
    ) -> Dict[str, Any]:
        """Query-adaptive search — V1-Plus architecture.

        Classifies the query, routes to the appropriate V1-Plus strategy,
        applies temporal decay post-hoc, and returns reranked results.

        All non-relational strategies use V1 hybrid_retrieve_hmm as the base,
        with lazy V2 metadata overlay before the Qwen3-Reranker-8B step.

        Strategies:
          - exact:      V1 hybrid + V2 overlay + reranker (factual precision)
          - temporal:   V1 hybrid + V2 overlay + reranker + post-hoc decay
          - conceptual: V1 hybrid + V2 overlay + reranker + theme-motif logging
          - relational: parallel sub-queries + RRF + graph expansion (unchanged)
          - motif:      V1 motif-filtered hybrid + Redis/Neo4j motif RRF + V2 overlay
          - default:    V1 hybrid + V2 overlay + reranker
        """
        from isma.src.query_classifier import classify_query
        from isma.src.temporal_query import apply_temporal_decay, HALF_LIVES
        from isma.src.retrieval import get_retrieval

        t0 = time.monotonic()

        plan = classify_query(query)
        strategy = plan.strategy

        # Merge classifier-detected filters with explicit filters
        merged_filters = dict(filters)
        if plan.detected_platform and "platform" not in merged_filters:
            merged_filters["platform"] = plan.detected_platform

        v1 = get_retrieval()

        # Route to strategy
        if strategy == "relational" and plan.sub_queries:
            # V1-Plus: use _search_relational with graph expansion (depth=3).
            # Graph densification v2 (52.8M RELATES_TO edges) makes this viable —
            # was disabled before densification when graph was too sparse to help.
            result = self._search_relational(
                query, plan.sub_queries, top_k,
                instruction=plan.reranker_instruction,
                expand_graph=True,
                graph_depth=3,
                **merged_filters,
            )
        elif strategy == "motif" and plan.detected_motifs:
            # Wire detected_motifs into retrieval (previously ignored → R@10=0.000)
            result = self._search_motif(
                query, plan.detected_motifs, top_k,
                instruction=plan.reranker_instruction,
                v1=v1,
                **merged_filters,
            )
        else:
            # V1-Plus: V1 search + lazy V2 metadata overlay + reranker
            theme_motifs = []
            if strategy == "conceptual":
                # Theme routing for audit/logging — not a hard filter to avoid recall regression
                theme_motifs = self._get_theme_motifs(query)

            result = self._v1_plus_search(
                query, top_k,
                instruction=plan.reranker_instruction,
                query_type=strategy,
                v1=v1,
                **merged_filters,
            )

            if theme_motifs:
                result["theme_routing"] = theme_motifs

        # Apply temporal decay post-hoc for temporal queries
        if strategy == "temporal":
            half_life = HALF_LIVES.get(strategy, 90)
            tiles = result.get("tiles", [])
            if tiles:
                result["tiles"] = apply_temporal_decay(
                    tiles, half_life_days=half_life, decay_weight=0.15,
                )

        elapsed_ms = (time.monotonic() - t0) * 1000
        result["search_time_ms"] = elapsed_ms
        result["strategy"] = strategy
        result["query_plan"] = {
            "strategy": plan.strategy,
            "confidence": plan.confidence,
            "detected_platform": plan.detected_platform,
            "detected_motifs": plan.detected_motifs,
            "temporal_window": plan.temporal_window,
        }

        return result

    def _v1_plus_search(
        self,
        query: str,
        top_k: int,
        instruction: str,
        query_type: str,
        v1,
        **filters,
    ) -> Dict[str, Any]:
        """V1 hybrid search + lazy V2 metadata overlay before reranking.

        Replaces Option E (V2 rosetta nearVector path) as the default non-relational
        strategy. V2 metadata (richer doc-level rosetta_summary, dominant_motifs)
        is overlaid onto V1 passage tiles before the cross-encoder sees them,
        improving reranker quality without the latency of V2 vector search.

        Sequence:
          1. V1 search (nearVector + BM25, 3x candidate pool)
          2. Lazy V2 metadata overlay (batch-fetch by content_hash OR-filter)
          3. Qwen3-Reranker-8B cross-encoder on enriched candidates
          4. Dedup by content_hash + truncate to top_k
        """
        from dataclasses import replace as dc_replace

        fetch_k = top_k * 3

        # Step 1: V1 vector + BM25 search
        search_result = v1.search(query, top_k=fetch_k, expand_parents=True, **filters)
        if not search_result.tiles:
            return {
                "query": query, "tiles": [], "total_tokens": 0,
                "hmm_reranked": False, "version": "v1plus",
            }

        # Step 2: Lazy V2 metadata overlay
        content_hashes = [t.content_hash for t in search_result.tiles if t.content_hash]
        if content_hashes:
            v2_meta = self._fetch_v2_metadata(content_hashes)
            if v2_meta:
                enriched = []
                for tile in search_result.tiles:
                    ch = tile.content_hash
                    if ch and ch in v2_meta:
                        meta = v2_meta[ch]
                        tile = dc_replace(
                            tile,
                            rosetta_summary=meta.get("rosetta_summary") or tile.rosetta_summary,
                            dominant_motifs=meta.get("dominant_motifs") or tile.dominant_motifs,
                        )
                    enriched.append(tile)
                search_result = SearchResult(
                    query=search_result.query,
                    tiles=enriched,
                    total_tokens=search_result.total_tokens,
                    search_time_ms=search_result.search_time_ms,
                )

        # Step 3: Neural rerank (Qwen3-Reranker-8B)
        reranked_result = v1.hmm_rerank(
            search_result, query, query_type=query_type, instruction=instruction,
        )

        # Step 4: Dedup by content_hash
        seen: set = set()
        deduped = []
        for tile in reranked_result.tiles:
            key = tile.content_hash or id(tile)
            if key not in seen:
                seen.add(key)
                deduped.append(tile)

        tiles = deduped[:top_k]
        return {
            "query": query,
            "tiles": tiles,
            "total_tokens": sum(t.token_count for t in tiles),
            "hmm_reranked": True,
            "version": "v1plus",
        }

    def _search_motif(
        self,
        query: str,
        detected_motifs: List[str],
        top_k: int,
        instruction: str,
        v1,
        **filters,
    ) -> Dict[str, Any]:
        """Motif-aware search: RRF of V1 hybrid + Redis/Neo4j motif candidates.

        Wires detected_motifs into actual retrieval, replacing the previous
        hybrid_search() call that ignored motifs entirely (R@10=0.000).

        Sequence:
          1a. V1 nearVector+BM25 with dominant_motifs Weaviate pre-filter (base signal)
          1b. Redis inverted index → Neo4j amplitude sort per detected motif (1.5x weight)
          2.  RRF merge of all paths
          3.  Lazy V2 metadata overlay on candidate pool
          4.  Backfill passage content for motif-only tiles
          5.  Qwen3-Reranker-8B rerank + dedup
        """
        from concurrent.futures import ThreadPoolExecutor
        from dataclasses import replace as dc_replace

        k = 60
        rrf_scores: Dict[str, float] = {}
        obj_map: Dict[str, TileResult] = {}

        # Step 1: Run V1 hybrid (motif-filtered) + motif searches in parallel
        motif_filters = dict(filters, dominant_motifs=detected_motifs)
        with ThreadPoolExecutor(max_workers=len(detected_motifs) + 1) as executor:
            hybrid_fut = executor.submit(
                v1.search, query, top_k=top_k * 3, expand_parents=True, **motif_filters
            )
            motif_futs = {
                m: executor.submit(v1.motif_search, m, limit=top_k * 3)
                for m in detected_motifs
            }

            # Collect V1 hybrid results (base weight = 1.0)
            try:
                hybrid_result = hybrid_fut.result(timeout=20)
                for rank, tile in enumerate(hybrid_result.tiles):
                    ch = tile.content_hash
                    if ch:
                        rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                        obj_map[ch] = tile
            except Exception as e:
                log.warning("Motif hybrid search failed: %s", e)

            # Collect Redis+Neo4j motif results (boosted weight = 1.5, amplitude-sorted)
            for motif_id, fut in motif_futs.items():
                try:
                    msr = fut.result(timeout=10)
                    for rank, tw in enumerate(msr.tiles_with_amplitude):
                        ch = tw.get("tile_hash", "")
                        if ch:
                            rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.5 / (k + rank + 1)
                            if ch not in obj_map:
                                obj_map[ch] = TileResult(
                                    content="",
                                    score=0.0,
                                    tile_id=ch,
                                    scale="search_512",
                                    source_type="",
                                    source_file="",
                                    content_hash=ch,
                                    platform=tw.get("platform", ""),
                                    rosetta_summary=tw.get("rosetta_summary", ""),
                                    dominant_motifs=tw.get("dominant_motifs") or [],
                                )
                except Exception as e:
                    log.warning("Motif search %s failed: %s", motif_id, e)

        if not rrf_scores:
            # All paths failed — fall back to plain V1-Plus
            return self._v1_plus_search(
                query, top_k, instruction=instruction, query_type="motif", v1=v1, **filters
            )

        # Step 2: Sort by RRF and take candidate pool
        sorted_hashes = sorted(rrf_scores, key=lambda ch: rrf_scores[ch], reverse=True)
        candidate_hashes = sorted_hashes[:top_k * 2]

        # Step 3: Lazy V2 metadata overlay
        v2_meta = self._fetch_v2_metadata([ch for ch in candidate_hashes if ch])
        tiles = []
        for ch in candidate_hashes:
            if ch not in obj_map:
                continue
            tile = obj_map[ch]
            if ch in v2_meta:
                meta = v2_meta[ch]
                tile = dc_replace(
                    tile,
                    rosetta_summary=meta.get("rosetta_summary") or tile.rosetta_summary,
                    dominant_motifs=meta.get("dominant_motifs") or tile.dominant_motifs,
                )
            tiles.append(dc_replace(tile, score=rrf_scores[ch]))

        # Step 4: Backfill passage content for motif-only tiles (no V1 content)
        backfill_map: Dict[str, TileResult] = {
            t.content_hash: t for t in tiles if not t.content and t.content_hash
        }
        if backfill_map:
            self._fill_content(list(backfill_map.keys()), backfill_map)
            tiles = [
                backfill_map.get(t.content_hash, t) if not t.content and t.content_hash else t
                for t in tiles
            ]

        # Step 5: Rerank
        try:
            from isma.src.reranker import get_reranker
            reranker = get_reranker()
            if reranker.is_available():
                tiles = reranker.rerank(query, tiles, instruction=instruction, query_type="motif")
        except Exception as e:
            log.debug("Reranker unavailable for motif: %s", e)

        # Dedup + truncate
        seen: set = set()
        deduped = []
        for tile in tiles:
            key = tile.content_hash or id(tile)
            if key not in seen:
                seen.add(key)
                deduped.append(tile)

        result_tiles = deduped[:top_k]
        return {
            "query": query,
            "tiles": result_tiles,
            "total_tokens": sum(t.token_count for t in result_tiles),
            "hmm_reranked": True,
            "version": "v1plus_motif",
        }

    def _fetch_v2_metadata(self, content_hashes: List[str]) -> Dict[str, dict]:
        """Batch-fetch rosetta_summary and dominant_motifs from V2 by content_hash.

        Uses OR-filter batches of 50 to stay within GraphQL complexity limits.
        Returns dict: content_hash → {rosetta_summary, dominant_motifs}
        """
        result: Dict[str, dict] = {}
        batch_size = 50
        for i in range(0, len(content_hashes), batch_size):
            batch = content_hashes[i:i + batch_size]
            if len(batch) == 1:
                where = (
                    f'{{ path: ["content_hash"], operator: Equal, '
                    f'valueText: "{_escape_gql(batch[0])}" }}'
                )
            else:
                operands = ", ".join(
                    f'{{ path: ["content_hash"], operator: Equal, valueText: "{_escape_gql(ch)}" }}'
                    for ch in batch
                )
                where = f'{{ operator: Or, operands: [{operands}] }}'
            q = (
                f'{{ Get {{ {V2_CLASS}('
                f'where: {where}'
                f' limit: {len(batch)}'
                f') {{ content_hash rosetta_summary dominant_motifs }} }} }}'
            )
            try:
                data = _graphql(q)
                for obj in data.get("data", {}).get("Get", {}).get(V2_CLASS, []):
                    ch = obj.get("content_hash", "")
                    if ch:
                        result[ch] = obj
            except Exception as e:
                log.debug("V2 metadata fetch failed for batch %d: %s", i // batch_size, e)
        return result

    def _search_relational(
        self,
        query: str,
        sub_queries: list,
        top_k: int = 10,
        instruction: str = "",
        expand_graph: bool = True,
        graph_depth: int = 2,
        **filters,
    ) -> Dict[str, Any]:
        """Relational search: sub-queries + graph expansion merged via RRF."""
        t0 = time.monotonic()
        k = 60  # RRF constant

        rrf_scores: Dict[str, float] = {}
        obj_map: Dict[str, Any] = {}

        # Run sub-queries + full query in parallel (full query gets 2x weight)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        all_queries = [(sq, 1.0) for sq in sub_queries] + [(query, 2.0)]
        with ThreadPoolExecutor(max_workers=min(len(all_queries), 8)) as executor:
            futures = {
                executor.submit(self.search, q, top_k=top_k * 2, **filters): weight
                for q, weight in all_queries
            }
            for future in as_completed(futures):
                weight = futures[future]
                try:
                    result = future.result()
                    for rank, tile in enumerate(result.tiles):
                        ch = tile.content_hash
                        if ch:
                            rrf_scores[ch] = rrf_scores.get(ch, 0) + weight / (k + rank + 1)
                            obj_map[ch] = tile
                except Exception as e:
                    log.warning("Relational sub-query failed: %s", e)

        # Phase 4: Neo4j graph expansion from seed results
        graph_expanded = 0
        if expand_graph:
            try:
                from isma.src.hmm.neo4j_store import HMMNeo4jStore
                store = HMMNeo4jStore()  # Uses shared driver singleton
                # Use top-ranked seed tiles for graph expansion (sorted by RRF score)
                seed_ids = sorted(rrf_scores.keys(), key=lambda ch: rrf_scores[ch], reverse=True)[:top_k]
                if seed_ids:
                    neighbors = store.graph_expand(
                        seed_ids, depth=graph_depth, follow_supersedes=True,
                    )
                    for rank, nb in enumerate(neighbors):
                        ch = nb.get("content_hash", "")
                        if ch:
                            # Accumulate RRF unconditionally — previously skipped nodes already
                            # found by direct search, penalizing the highest-relevance items.
                            rrf_scores[ch] = rrf_scores.get(ch, 0) + 0.5 / (k + rank + 1)
                            if ch not in obj_map:
                                # Create TileResult from graph data (only if not already present)
                                obj_map[ch] = TileResult(
                                    content="",  # Content fetched later if needed
                                    content_hash=ch,
                                    platform=nb.get("platform", ""),
                                    source_type="", source_file="",
                                    session_id="", document_id="",
                                    scale="canonical", tile_id=ch,
                                    token_count=0, score=rrf_scores[ch],
                                    rosetta_summary=nb.get("rosetta_summary", ""),
                                    dominant_motifs=nb.get("dominant_motifs") or [],
                                )
                                graph_expanded += 1
                # No store.close() needed — uses shared driver singleton
            except Exception as e:
                log.debug("Graph expansion failed: %s", e)

        # Batch-fetch content for graph-expanded tiles (content="")
        empty_hashes = [
            ch for ch, tile in obj_map.items()
            if not tile.content and ch in rrf_scores
        ]
        if empty_hashes:
            self._fill_content(empty_hashes, obj_map)

        # Sort by RRF score and take top_k (immutable — use dataclass replace)
        from dataclasses import replace as dc_replace
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda ch: rrf_scores[ch], reverse=True)
        tiles = []
        for ch in sorted_hashes[:top_k * 3]:
            tile = obj_map[ch]
            tiles.append(dc_replace(tile, score=rrf_scores[ch]))

        # Rerank merged results
        try:
            from isma.src.reranker import get_reranker
            reranker = get_reranker()
            if reranker.is_available():
                tiles = reranker.rerank(
                    query, tiles,
                    instruction=instruction,
                    query_type="relational",
                )
        except Exception as e:
            log.debug("Reranker unavailable for relational: %s", e)

        elapsed_ms = (time.monotonic() - t0) * 1000
        return {
            "query": query,
            "tiles": tiles[:top_k],
            "total_tokens": sum(t.token_count for t in tiles[:top_k]),
            "search_time_ms": elapsed_ms,
            "hmm_reranked": True,
            "version": "v2",
            "sub_queries": sub_queries,
            "graph_expanded": graph_expanded,
        }

    # ── Content Backfill ───────────────────────────────────────

    def _fill_content(self, content_hashes: List[str], obj_map: Dict[str, Any]):
        """Batch-fetch content from v2 for tiles with empty content (graph-expanded).

        Uses a single OR-filter query instead of N+1 individual queries.
        Batches of 50 to stay within GraphQL complexity limits.
        """
        from dataclasses import replace as dc_replace
        batch_size = 50
        filled = 0
        for i in range(0, len(content_hashes), batch_size):
            batch = content_hashes[i:i + batch_size]
            if len(batch) == 1:
                safe_ch = _escape_gql(batch[0])
                where = f'{{ path: ["content_hash"], operator: Equal, valueText: "{safe_ch}" }}'
            else:
                operands = ", ".join(
                    f'{{ path: ["content_hash"], operator: Equal, valueText: "{_escape_gql(ch)}" }}'
                    for ch in batch
                )
                where = f'{{ operator: Or, operands: [{operands}] }}'
            q = (
                f'{{ Get {{ {V2_CLASS}('
                f'where: {where}'
                f' limit: {len(batch)}'
                f') {{ content_hash content rosetta_summary loaded_at }} }} }}'
            )
            data = _graphql(q)
            results = data.get("data", {}).get("Get", {}).get(V2_CLASS, [])
            for fetched in results:
                ch = fetched.get("content_hash", "")
                if ch in obj_map:
                    tile = obj_map[ch]
                    obj_map[ch] = dc_replace(
                        tile,
                        content=fetched.get("content", "") or tile.content,
                        rosetta_summary=fetched.get("rosetta_summary", "") or tile.rosetta_summary,
                        loaded_at=fetched.get("loaded_at", "") or tile.loaded_at,
                    )
                    filled += 1
        log.debug("Backfilled content for %d/%d graph-expanded tiles", filled, len(content_hashes))

    # ── Passage Expansion ───────────────────────────────────────

    def expand_passages(
        self,
        content_hash: str,
        scale: str = "search_512",
    ) -> List[TileResult]:
        """Fetch all v1 tiles for a content_hash at a given scale.

        Used to drill into specific passages after document-level search.
        """
        from isma.src.retrieval import ISMARetrieval

        r = ISMARetrieval()
        return r.get_tiles_for_content(content_hash, scale=scale)

    def _get_theme_motifs(self, query: str) -> list:
        """Route a conceptual query through ISMA_Themes to get relevant motif filter.

        Queries ISMA_Themes (24 objects — sub-millisecond nearVector) instead of
        scanning ISMA_Quantum (1M objects). Returns required_motifs of the top-matching
        theme to use as a seed-set filter predicate on the main ISMA_Quantum search.

        Returns list of motif strings (e.g. ['HMM.SACRED_TRUST']) or [] on failure.
        """
        try:
            vector = _get_embedding(query)
            if not vector:
                return []
            vector_str = str(vector)
            gql = (
                "{ Get { ISMA_Themes("
                f"nearVector: {{ vector: {vector_str} }}"
                " limit: 1 certainty: 0.6"
                ") { theme_id display_name required_motifs _additional { distance } } } }"
            )
            data = requests.post(
                f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=3
            ).json()
            themes = (data.get("data") or {}).get("Get", {}).get("ISMA_Themes", [])
            if not themes:
                return []
            top = themes[0]
            motifs = top.get("required_motifs") or []
            dist = top.get("_additional", {}).get("distance", 1.0)
            log.debug(
                "Theme routing: %s (%s) dist=%.3f motifs=%s",
                top.get("display_name"), top.get("theme_id"), dist, motifs
            )
            return motifs
        except Exception as e:
            log.debug("Theme routing failed: %s", e)
        return []

    def _get_theme_context(self, query: str) -> str:
        """DEPRECATED — use _get_theme_motifs() instead.

        Old implementation queried ISMA_Quantum (1M tiles) with nearVector + where
        post-filter to find 24 theme tiles, causing 3.8x latency spike.
        Now delegates to _get_theme_motifs() which queries ISMA_Themes (24 objects).
        Kept for backward compatibility but returns empty string (functionality moved).
        """
        return ""

    # ── Filters ─────────────────────────────────────────────────

    def _build_filter(self, **filters) -> str:
        """Build a Weaviate where filter clause from keyword arguments."""
        conditions = []

        for key, value in filters.items():
            if value is None:
                continue
            if key == "platform":
                safe_val = str(value).replace("\\", "\\\\").replace('"', '\\"')
                conditions.append(
                    f'{{ path: ["platform"], operator: Equal, valueText: "{safe_val}" }}'
                )
            elif key == "source_type":
                safe_val = str(value).replace("\\", "\\\\").replace('"', '\\"')
                conditions.append(
                    f'{{ path: ["source_type"], operator: Equal, valueText: "{safe_val}" }}'
                )
            elif key == "hmm_enriched":
                conditions.append(
                    f'{{ path: ["hmm_enriched"], operator: Equal, valueBoolean: {"true" if value else "false"} }}'
                )
            elif key == "min_hmm_phi":
                conditions.append(
                    f'{{ path: ["hmm_phi"], operator: GreaterThanEqual, valueNumber: {value} }}'
                )
            elif key == "min_hmm_trust":
                conditions.append(
                    f'{{ path: ["hmm_trust"], operator: GreaterThanEqual, valueNumber: {value} }}'
                )

        if not conditions:
            return ""

        if len(conditions) == 1:
            return f", where: {conditions[0]}"
        else:
            ops = ", ".join(conditions)
            return f", where: {{ operator: And, operands: [{ops}] }}"


# Module-level singleton with thread safety
_instance: Optional[ISMARetrievalV2] = None
_instance_lock = threading.Lock()


def get_retrieval_v2() -> ISMARetrievalV2:
    """Get singleton ISMARetrievalV2 instance (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ISMARetrievalV2()
    return _instance
