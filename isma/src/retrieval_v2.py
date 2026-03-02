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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

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

# Connection-pooled session for Weaviate GraphQL queries
_wv_session = requests.Session()

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


def _graphql(query: str) -> dict:
    """Execute a GraphQL query using connection-pooled session."""
    try:
        r = _wv_session.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            timeout=30,
        )
        return r.json()
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
        **filters,
    ) -> List[Tuple[dict, float]]:
        """Search using the rosetta (summary) named vector."""
        embedding = _get_embedding(query)
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
        # Escape query for GraphQL
        safe_query = query.replace('"', '\\"').replace("\\", "\\\\")
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

    # ── Hybrid Search (RRF Fusion) ──────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        **filters,
    ) -> SearchResult:
        """Hybrid search with RRF fusion of raw + rosetta + BM25 results."""
        t0 = time.monotonic()

        # Fetch candidates from all three paths
        fetch_k = top_k * 2
        raw_results = self.search_raw(query, top_k=fetch_k, **filters)
        rosetta_results = self.search_rosetta(query, top_k=fetch_k, **filters)
        bm25_results = self.search_bm25(query, top_k=fetch_k, **filters)

        # RRF fusion (k=60 is standard)
        k = 60
        rrf_scores: Dict[str, float] = {}
        obj_map: Dict[str, dict] = {}

        for rank, (obj, _score) in enumerate(raw_results):
            ch = obj.get("content_hash", "")
            if ch:
                rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                obj_map[ch] = obj

        for rank, (obj, _score) in enumerate(rosetta_results):
            ch = obj.get("content_hash", "")
            if ch:
                rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                obj_map[ch] = obj

        for rank, (obj, _score) in enumerate(bm25_results):
            ch = obj.get("content_hash", "")
            if ch:
                rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                if ch not in obj_map:
                    obj_map[ch] = obj

        # Sort by RRF score
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda ch: rrf_scores[ch], reverse=True)

        tiles = []
        for ch in sorted_hashes[:top_k]:
            obj = obj_map[ch]
            tile = _v2_to_tile(obj, score=rrf_scores[ch])
            tiles.append(tile)

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
        """Query-adaptive search with automatic strategy selection.

        Classifies the query, selects optimal retrieval strategy,
        applies temporal decay, optionally expands through Neo4j graph,
        and returns reranked results.

        Strategies:
          - exact: hybrid search with factual precision instruction
          - temporal: hybrid search + temporal decay scoring
          - conceptual: hybrid search with thematic depth instruction
          - relational: parallel sub-queries + RRF merge + graph expansion
          - motif: motif-filtered search + hybrid fallback
          - default: standard hybrid search
        """
        from isma.src.query_classifier import classify_query
        from isma.src.temporal_query import (
            apply_temporal_decay,
            HALF_LIVES,
        )

        t0 = time.monotonic()

        plan = classify_query(query)
        strategy = plan.strategy

        # Merge classifier-detected filters with explicit filters
        merged_filters = dict(filters)
        if plan.detected_platform and "platform" not in merged_filters:
            merged_filters["platform"] = plan.detected_platform

        # Route to strategy
        if strategy == "relational" and plan.sub_queries:
            result = self._search_relational(
                query, plan.sub_queries, top_k,
                instruction=plan.reranker_instruction,
                expand_graph=expand_graph,
                graph_depth=graph_depth,
                **merged_filters,
            )
        else:
            # All other strategies use hybrid_search with appropriate instruction
            result = self.hybrid_search(
                query,
                top_k=top_k,
                rerank=True,
                query_type=strategy,
                instruction=plan.reranker_instruction,
                **merged_filters,
            )

        # Apply temporal decay for temporal queries
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

        # Run sub-queries (sequential for now, could be parallel)
        for sq in sub_queries:
            sub_result = self.search(sq, top_k=top_k * 2, **filters)
            for rank, tile in enumerate(sub_result.tiles):
                ch = tile.content_hash
                if ch:
                    rrf_scores[ch] = rrf_scores.get(ch, 0) + 1.0 / (k + rank + 1)
                    obj_map[ch] = tile

        # Also search with full query
        full_result = self.search(query, top_k=top_k * 2, **filters)
        for rank, tile in enumerate(full_result.tiles):
            ch = tile.content_hash
            if ch:
                # Full query gets 2x weight
                rrf_scores[ch] = rrf_scores.get(ch, 0) + 2.0 / (k + rank + 1)
                obj_map[ch] = tile

        # Phase 4: Neo4j graph expansion from seed results
        graph_expanded = 0
        if expand_graph:
            try:
                from isma.src.hmm.neo4j_store import HMMNeo4jStore
                store = HMMNeo4jStore()  # Uses shared driver singleton
                # Use top seed tiles for graph expansion
                seed_ids = list(rrf_scores.keys())[:top_k]
                if seed_ids:
                    neighbors = store.graph_expand(
                        seed_ids, depth=graph_depth, follow_supersedes=True,
                    )
                    for rank, nb in enumerate(neighbors):
                        ch = nb.get("content_hash", "")
                        if ch and ch not in obj_map:
                            # Graph neighbors get lower weight than direct search
                            rrf_scores[ch] = rrf_scores.get(ch, 0) + 0.5 / (k + rank + 1)
                            # Create TileResult from graph data
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

        # Sort by RRF score and take top_k
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda ch: rrf_scores[ch], reverse=True)
        tiles = []
        for ch in sorted_hashes[:top_k * 3]:
            tile = obj_map[ch]
            tile.score = rrf_scores[ch]
            tiles.append(tile)

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


# Module-level singleton
_instance: Optional[ISMARetrievalV2] = None


def get_retrieval_v2() -> ISMARetrievalV2:
    """Get singleton ISMARetrievalV2 instance."""
    global _instance
    if _instance is None:
        _instance = ISMARetrievalV2()
    return _instance
