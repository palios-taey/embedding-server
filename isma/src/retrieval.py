"""
ISMA Retrieval Layer - Clean access to everything.

Works with the unified pipeline schema:
  Weaviate: ISMA_Quantum (tiles with vectors + rich metadata)
  Neo4j: ISMASession → ISMAExchange, Document (graph navigation)

Provides:
  1. Vector search (semantic similarity via embeddings)
  2. Hybrid search (BM25 + vector combined)
  3. Graph queries (sessions, exchanges, documents via Neo4j)
  4. Content retrieval (full text from tiles, exchanges, documents)
  5. Filtered search (by platform, session, scale, time range, etc.)
  6. Parent expansion (search_512 → context_2048 → full_4096)
  7. Batch operations (multi-query for efficiency)

Usage:
    from retrieval import ISMARetrieval

    r = ISMARetrieval()

    # Semantic search
    results = r.search("consciousness emergence", top_k=10)

    # Filtered search
    results = r.search("trust", top_k=5, platform="grok", scale="search_512")

    # Get full session
    session = r.get_session("uuid-here")

    # Get all exchanges in a session
    exchanges = r.get_exchanges(session_id="uuid-here")

    # Get document by hash
    doc = r.get_document(content_hash="abc123")

    # Batch semantic search
    results = r.batch_search(["query1", "query2", "query3"], top_k=5)

    # Get everything for a platform
    sessions = r.list_sessions(platform="chatgpt", limit=100)
"""

import json
import hashlib
import os
import requests
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field

import redis

# =============================================================================
# CONFIGURATION
# =============================================================================

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://10.0.0.163:7689")
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://192.168.100.10:8091/v1/embeddings")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

WEAVIATE_CLASS = "ISMA_Quantum"

# All queryable Weaviate properties
TILE_PROPERTIES = [
    "content", "source_type", "source_file", "layer", "platform",
    "session_id", "document_id", "scale", "parent_tile_id",
    "tile_index", "start_char", "end_char", "token_count",
    "content_hash", "timestamp", "loaded_at", "actor", "model",
    "has_artifacts", "artifact_count", "has_thinking",
    "conversation_id", "priority", "exchange_index",
    # HMM enrichment properties
    "dominant_motifs", "rosetta_summary", "hmm_phi", "hmm_trust",
    "hmm_platforms", "hmm_enriched", "hmm_enrichment_version",
    "hmm_enriched_at", "hmm_consensus", "hmm_gate_flags",
    # motif_data_json intentionally omitted (too large for default queries)
]


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class TileResult:
    """A single tile from vector search."""
    content: str
    score: float
    tile_id: str
    scale: str
    source_type: str  # "document" or "transcript"
    source_file: str
    content_hash: str
    # Metadata
    platform: str = ""
    session_id: str = ""
    document_id: str = ""
    conversation_id: str = ""
    model: str = ""
    timestamp: str = ""
    layer: int = 0
    priority: float = 0.0
    tile_index: int = 0
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    exchange_index: int = 0
    has_artifacts: bool = False
    artifact_count: int = 0
    has_thinking: bool = False
    parent_tile_id: str = ""
    # HMM enrichment fields
    dominant_motifs: List[str] = field(default_factory=list)
    rosetta_summary: str = ""
    hmm_phi: float = 0.0
    hmm_trust: float = 0.0
    hmm_platforms: List[str] = field(default_factory=list)
    hmm_enriched: bool = False
    hmm_enrichment_version: str = ""
    hmm_enriched_at: str = ""
    hmm_consensus: bool = False
    hmm_gate_flags: List[str] = field(default_factory=list)
    # Temporal
    loaded_at: str = ""
    # Expanded context (filled on demand)
    expanded_context: str = ""


@dataclass
class ExchangeResult:
    """A full exchange from Neo4j."""
    content_hash: str
    session_id: str
    exchange_index: int
    user_prompt: str
    response: str
    timestamp: str
    model: str


@dataclass
class SessionResult:
    """A session from Neo4j."""
    session_id: str
    platform: str
    title: str
    source_file: str
    exchange_count: int
    created_at: str
    model: str


@dataclass
class DocumentResult:
    """A document from Neo4j."""
    content_hash: str
    document_id: str
    filename: str
    all_paths: List[str]
    layer: str
    priority: float
    file_size: int
    tile_count: int


@dataclass
class SearchResult:
    """Combined search result with tiles and optional expansions."""
    query: str
    tiles: List[TileResult]
    total_tokens: int = 0
    search_time_ms: float = 0.0


# =============================================================================
# EMBEDDING
# =============================================================================

_EMBED_LOCK = threading.Lock()


def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector for a query string."""
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": [text],
        }, timeout=30)
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
    except Exception:
        pass
    return None


def _get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings for multiple texts in one call."""
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        }, timeout=60)
        if r.status_code == 200:
            data = r.json()["data"]
            return [item["embedding"]
                    for item in sorted(data, key=lambda x: x["index"])]
    except Exception:
        pass
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================================
# F1 CANONICAL MAPPING (lazy singleton)
# =============================================================================

_CANONICAL_MAPPING = None
_CANONICAL_LOCK = threading.Lock()
CANONICAL_MAPPING_PATH = "/var/spark/isma/f1_canonical_mapping.json"


def _get_canonical_mapping() -> Dict[str, Any]:
    """Load the F1 canonical mapping (lazy, thread-safe singleton)."""
    global _CANONICAL_MAPPING
    with _CANONICAL_LOCK:
        if _CANONICAL_MAPPING is None:
            with open(CANONICAL_MAPPING_PATH) as f:
                _CANONICAL_MAPPING = json.load(f)
    return _CANONICAL_MAPPING


def _expand_theme_to_motifs(theme_id: str) -> Tuple[List[str], List[str], str, float]:
    """Expand a theme_id to its constituent motifs.

    Returns (required_motifs, supporting_motifs, activation_rule, threshold).
    """
    mapping = _get_canonical_mapping()
    theme = mapping["theme_registry"].get(theme_id)
    if not theme:
        return [], [], "any_required", 0.4
    return (
        theme["required_motifs"],
        theme["supporting_motifs"],
        theme.get("activation_rule", "any_required"),
        theme.get("threshold", 0.4),
    )


def _motifs_for_band(band: str) -> List[str]:
    """Return all motif_ids with the given band (slow/mid/fast)."""
    mapping = _get_canonical_mapping()
    return [
        mid for mid, info in mapping["motif_registry"].items()
        if info["band"] == band
    ]


# =============================================================================
# NEO4J CONNECTION
# =============================================================================

_NEO4J_DRIVER = None
_NEO4J_LOCK = threading.Lock()


def _get_neo4j():
    global _NEO4J_DRIVER
    with _NEO4J_LOCK:
        if _NEO4J_DRIVER is None:
            from neo4j import GraphDatabase
            _NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=None)
            _NEO4J_DRIVER.verify_connectivity()
    return _NEO4J_DRIVER


# =============================================================================
# REDIS CONNECTION
# =============================================================================

REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379

_REDIS_CONN = None
_REDIS_LOCK = threading.Lock()


def _get_redis():
    """Get the singleton Redis connection (lazy, thread-safe)."""
    global _REDIS_CONN
    with _REDIS_LOCK:
        if _REDIS_CONN is None:
            _REDIS_CONN = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
            )
    return _REDIS_CONN


# =============================================================================
# HMM RESULT TYPES
# =============================================================================

@dataclass
class MotifSearchResult:
    """Result of a motif-based search via Redis + Neo4j."""
    motif_id: str
    tile_hashes: List[str]
    tiles_with_amplitude: List[Dict[str, Any]]
    total_candidates: int


@dataclass
class GraphExpansionResult:
    """Result of graph expansion via RELATES_TO edges."""
    source_tile_hash: str
    related_tiles: List[Dict[str, Any]]
    depth: int


# =============================================================================
# WEAVIATE GRAPHQL HELPERS
# =============================================================================

def _escape_graphql(s: str) -> str:
    """Escape a string for GraphQL query embedding."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')[:500]


def _build_where_filter(
    platform: str = None,
    source_type: str = None,
    scale: str = None,
    session_id: str = None,
    document_id: str = None,
    content_hash: str = None,
    has_artifacts: bool = None,
    has_thinking: bool = None,
    layer: int = None,
    min_priority: float = None,
    model: str = None,
    # HMM filters
    dominant_motifs: List[str] = None,
    hmm_enriched: bool = None,
    hmm_consensus: bool = None,
    min_hmm_phi: float = None,
    min_hmm_trust: float = None,
) -> Optional[str]:
    """Build a Weaviate where filter from keyword arguments."""
    conditions = []

    if platform:
        conditions.append(
            f'{{ path: ["platform"], operator: Equal, valueText: "{_escape_graphql(platform)}" }}')
    if source_type:
        conditions.append(
            f'{{ path: ["source_type"], operator: Equal, valueText: "{_escape_graphql(source_type)}" }}')
    if scale:
        conditions.append(
            f'{{ path: ["scale"], operator: Equal, valueText: "{_escape_graphql(scale)}" }}')
    if session_id:
        conditions.append(
            f'{{ path: ["session_id"], operator: Equal, valueText: "{_escape_graphql(session_id)}" }}')
    if document_id:
        conditions.append(
            f'{{ path: ["document_id"], operator: Equal, valueText: "{_escape_graphql(document_id)}" }}')
    if content_hash:
        conditions.append(
            f'{{ path: ["content_hash"], operator: Equal, valueText: "{_escape_graphql(content_hash)}" }}')
    if has_artifacts is not None:
        val = "true" if has_artifacts else "false"
        conditions.append(
            f'{{ path: ["has_artifacts"], operator: Equal, valueBoolean: {val} }}')
    if has_thinking is not None:
        val = "true" if has_thinking else "false"
        conditions.append(
            f'{{ path: ["has_thinking"], operator: Equal, valueBoolean: {val} }}')
    if layer is not None:
        conditions.append(
            f'{{ path: ["layer"], operator: Equal, valueInt: {layer} }}')
    if min_priority is not None:
        conditions.append(
            f'{{ path: ["priority"], operator: GreaterThanEqual, valueNumber: {min_priority} }}')
    if model:
        conditions.append(
            f'{{ path: ["model"], operator: Equal, valueText: "{_escape_graphql(model)}" }}')
    # HMM filters
    if dominant_motifs:
        # ContainsAny: match tiles with ANY of the specified motifs
        motif_values = ", ".join(f'"{_escape_graphql(m)}"' for m in dominant_motifs)
        conditions.append(
            f'{{ path: ["dominant_motifs"], operator: ContainsAny, valueText: [{motif_values}] }}')
    if hmm_enriched is not None:
        val = "true" if hmm_enriched else "false"
        conditions.append(
            f'{{ path: ["hmm_enriched"], operator: Equal, valueBoolean: {val} }}')
    if hmm_consensus is not None:
        val = "true" if hmm_consensus else "false"
        conditions.append(
            f'{{ path: ["hmm_consensus"], operator: Equal, valueBoolean: {val} }}')
    if min_hmm_phi is not None:
        conditions.append(
            f'{{ path: ["hmm_phi"], operator: GreaterThanEqual, valueNumber: {min_hmm_phi} }}')
    if min_hmm_trust is not None:
        conditions.append(
            f'{{ path: ["hmm_trust"], operator: GreaterThanEqual, valueNumber: {min_hmm_trust} }}')

    if not conditions:
        return None
    if len(conditions) == 1:
        return f"where: {conditions[0]}"
    return f"where: {{ operator: And, operands: [{', '.join(conditions)}] }}"


def _parse_tile(obj: dict) -> TileResult:
    """Parse a Weaviate result object into a TileResult."""
    additional = obj.get("_additional", {})
    return TileResult(
        content=obj.get("content", ""),
        score=additional.get("score", additional.get("certainty", 0.0)) or 0.0,
        tile_id=additional.get("id", ""),
        scale=obj.get("scale", ""),
        source_type=obj.get("source_type", ""),
        source_file=obj.get("source_file", ""),
        content_hash=obj.get("content_hash", ""),
        platform=obj.get("platform", ""),
        session_id=obj.get("session_id", ""),
        document_id=obj.get("document_id", ""),
        conversation_id=obj.get("conversation_id", ""),
        model=obj.get("model", ""),
        timestamp=obj.get("timestamp", ""),
        layer=obj.get("layer", 0) or 0,
        priority=obj.get("priority", 0.0) or 0.0,
        tile_index=obj.get("tile_index", 0) or 0,
        start_char=obj.get("start_char", 0) or 0,
        end_char=obj.get("end_char", 0) or 0,
        token_count=obj.get("token_count", 0) or 0,
        exchange_index=obj.get("exchange_index", 0) or 0,
        has_artifacts=obj.get("has_artifacts", False) or False,
        artifact_count=obj.get("artifact_count", 0) or 0,
        has_thinking=obj.get("has_thinking", False) or False,
        parent_tile_id=obj.get("parent_tile_id", ""),
        # HMM enrichment
        dominant_motifs=obj.get("dominant_motifs") or [],
        rosetta_summary=obj.get("rosetta_summary") or "",
        hmm_phi=obj.get("hmm_phi") or 0.0,
        hmm_trust=obj.get("hmm_trust") or 0.0,
        hmm_platforms=obj.get("hmm_platforms") or [],
        hmm_enriched=obj.get("hmm_enriched") or False,
        hmm_enrichment_version=obj.get("hmm_enrichment_version") or "",
        hmm_enriched_at=obj.get("hmm_enriched_at") or "",
        hmm_consensus=obj.get("hmm_consensus") or False,
        hmm_gate_flags=obj.get("hmm_gate_flags") or [],
        loaded_at=obj.get("loaded_at") or "",
    )


# =============================================================================
# MAIN RETRIEVAL CLASS
# =============================================================================

class ISMARetrieval:
    """Clean retrieval interface for ISMA unified data."""

    def __init__(self):
        self._session = requests.Session()

    # -----------------------------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------------------------

    def search(self, query: str, top_k: int = 10,
               expand_parents: bool = False,
               # Filters
               platform: str = None,
               source_type: str = None,
               scale: str = None,
               session_id: str = None,
               document_id: str = None,
               has_artifacts: bool = None,
               has_thinking: bool = None,
               layer: int = None,
               min_priority: float = None,
               model: str = None,
               # HMM filters
               dominant_motifs: List[str] = None,
               hmm_enriched: bool = None,
               hmm_consensus: bool = None,
               min_hmm_phi: float = None,
               min_hmm_trust: float = None,
               # Theme/band expansion
               theme_id: str = None,
               motif_band: str = None,
               ) -> SearchResult:
        """Semantic vector search with optional filters.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            expand_parents: If True, expand search_512 tiles to their
                           context_2048 parents
            platform: Filter by platform (chatgpt, claude_code, gemini, etc.)
            source_type: Filter by "document" or "transcript"
            scale: Filter by tile scale (search_512, context_2048, full_4096)
            session_id: Filter to specific session
            document_id: Filter to specific document
            has_artifacts: Filter to exchanges with/without artifacts
            has_thinking: Filter to exchanges with/without thinking
            layer: Filter by corpus layer (-1=kernel, 0=layer_0, 1, 2)
            min_priority: Filter by minimum priority score
            model: Filter by AI model name
            dominant_motifs: Filter by HMM motif IDs (ContainsAny)
            hmm_enriched: Filter to enriched/unenriched tiles
            hmm_consensus: Filter to consensus/non-consensus tiles
            min_hmm_phi: Filter by minimum HMM phi score
            min_hmm_trust: Filter by minimum HMM trust score
            theme_id: Expand theme to constituent motifs, merge into dominant_motifs
            motif_band: Filter to motifs in specified band (slow/mid/fast)
        """
        import time
        t0 = time.time()

        # Expand theme_id to motifs
        if theme_id:
            req, sup, _, _ = _expand_theme_to_motifs(theme_id)
            theme_motifs = req + sup
            if dominant_motifs:
                dominant_motifs = list(set(dominant_motifs + theme_motifs))
            else:
                dominant_motifs = theme_motifs

        # Expand motif_band to motifs
        if motif_band:
            band_motifs = _motifs_for_band(motif_band)
            if dominant_motifs:
                dominant_motifs = list(set(dominant_motifs) & set(band_motifs)) or band_motifs
            else:
                dominant_motifs = band_motifs

        embedding = _get_embedding(query)
        if not embedding:
            return SearchResult(query=query, tiles=[])

        where = _build_where_filter(
            platform=platform, source_type=source_type, scale=scale,
            session_id=session_id, document_id=document_id,
            has_artifacts=has_artifacts, has_thinking=has_thinking,
            layer=layer, min_priority=min_priority, model=model,
            dominant_motifs=dominant_motifs, hmm_enriched=hmm_enriched,
            hmm_consensus=hmm_consensus, min_hmm_phi=min_hmm_phi,
            min_hmm_trust=min_hmm_trust,
        )

        props = " ".join(TILE_PROPERTIES)
        safe_q = _escape_graphql(query)

        # Build GraphQL query
        filters = []
        if where:
            filters.append(where)

        filter_str = ", ".join(filters)
        if filter_str:
            filter_str = ", " + filter_str

        gql = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    hybrid: {{
                        query: "{safe_q}"
                        alpha: 0.65
                        vector: {json.dumps(embedding)}
                    }}
                    limit: {top_k}
                    {filter_str}
                ) {{
                    {props}
                    _additional {{ id score }}
                }}
            }}
        }}"""

        try:
            r = self._session.post(f"{WEAVIATE_URL}/v1/graphql",
                                   json={"query": gql}, timeout=30)
            if r.status_code != 200:
                return SearchResult(query=query, tiles=[])

            data = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
            tiles = [_parse_tile(obj) for obj in data]
        except Exception:
            return SearchResult(query=query, tiles=[])

        if expand_parents:
            self._expand_parents(tiles)

        total_tokens = sum(t.token_count for t in tiles)
        elapsed = (time.time() - t0) * 1000

        return SearchResult(
            query=query, tiles=tiles,
            total_tokens=total_tokens, search_time_ms=elapsed,
        )

    def search_by_vector(self, vector: List[float], top_k: int = 10,
                         **filters) -> SearchResult:
        """Search by raw embedding vector (no query text needed)."""
        where = _build_where_filter(**filters)

        props = " ".join(TILE_PROPERTIES)
        filter_parts = []
        if where:
            filter_parts.append(where)
        filter_str = ", ".join(filter_parts)
        if filter_str:
            filter_str = ", " + filter_str

        gql = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    nearVector: {{
                        vector: {json.dumps(vector)}
                    }}
                    limit: {top_k}
                    {filter_str}
                ) {{
                    {props}
                    _additional {{ id certainty }}
                }}
            }}
        }}"""

        try:
            r = self._session.post(f"{WEAVIATE_URL}/v1/graphql",
                                   json={"query": gql}, timeout=30)
            if r.status_code != 200:
                return SearchResult(query="[vector]", tiles=[])
            data = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
            tiles = [_parse_tile(obj) for obj in data]
        except Exception:
            return SearchResult(query="[vector]", tiles=[])

        return SearchResult(query="[vector]", tiles=tiles,
                            total_tokens=sum(t.token_count for t in tiles))

    def search_bm25(self, query: str, top_k: int = 10,
                    **filters) -> SearchResult:
        """BM25 keyword search (no vectors, fast)."""
        where = _build_where_filter(**filters)
        props = " ".join(TILE_PROPERTIES)
        safe_q = _escape_graphql(query)

        filter_parts = []
        if where:
            filter_parts.append(where)
        filter_str = ", ".join(filter_parts)
        if filter_str:
            filter_str = ", " + filter_str

        gql = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    bm25: {{ query: "{safe_q}" }}
                    limit: {top_k}
                    {filter_str}
                ) {{
                    {props}
                    _additional {{ id score }}
                }}
            }}
        }}"""

        try:
            r = self._session.post(f"{WEAVIATE_URL}/v1/graphql",
                                   json={"query": gql}, timeout=30)
            if r.status_code != 200:
                return SearchResult(query=query, tiles=[])
            data = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
            tiles = [_parse_tile(obj) for obj in data]
        except Exception:
            return SearchResult(query=query, tiles=[])

        return SearchResult(query=query, tiles=tiles,
                            total_tokens=sum(t.token_count for t in tiles))

    # -----------------------------------------------------------------
    # BATCH SEARCH
    # -----------------------------------------------------------------

    def batch_search(self, queries: List[str], top_k: int = 5,
                     **filters) -> List[SearchResult]:
        """Batch semantic search - embeds all queries in one call."""
        embeddings = _get_embeddings_batch(queries)
        if not embeddings or len(embeddings) != len(queries):
            return [SearchResult(query=q, tiles=[]) for q in queries]

        results = []
        for query, embedding in zip(queries, embeddings):
            result = self.search_by_vector(embedding, top_k=top_k, **filters)
            result.query = query
            results.append(result)
        return results

    # -----------------------------------------------------------------
    # PARENT EXPANSION (search_512 → context_2048 → full_4096)
    # -----------------------------------------------------------------

    def _expand_parents(self, tiles: List[TileResult]):
        """Expand search_512 tiles to their context_2048 parents."""
        parent_ids = set()
        for t in tiles:
            if t.scale == "search_512" and t.parent_tile_id:
                parent_ids.add(t.parent_tile_id)

        if not parent_ids:
            return

        # Fetch parents by ID
        parents = {}
        for pid in parent_ids:
            parent = self._get_tile_by_id(pid)
            if parent:
                parents[pid] = parent

        # Attach expanded context
        for t in tiles:
            if t.parent_tile_id in parents:
                t.expanded_context = parents[t.parent_tile_id].get("content", "")

    def _get_tile_by_id(self, tile_id: str) -> Optional[dict]:
        """Fetch a single tile by its Weaviate UUID."""
        props = " ".join(TILE_PROPERTIES)
        try:
            r = self._session.get(
                f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{tile_id}",
                timeout=10)
            if r.status_code == 200:
                return r.json().get("properties", {})
        except Exception:
            pass
        return None

    def get_tiles_for_content(self, content_hash: str,
                              scale: str = None) -> List[TileResult]:
        """Get all tiles for a content hash (document or exchange)."""
        where = _build_where_filter(content_hash=content_hash, scale=scale)
        props = " ".join(TILE_PROPERTIES)

        gql = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    {where or ''}
                    limit: 500
                ) {{
                    {props}
                    _additional {{ id }}
                }}
            }}
        }}"""

        try:
            r = self._session.post(f"{WEAVIATE_URL}/v1/graphql",
                                   json={"query": gql}, timeout=30)
            if r.status_code != 200:
                return []
            data = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
            tiles = [_parse_tile(obj) for obj in data]
            tiles.sort(key=lambda t: (t.scale, t.tile_index))
            return tiles
        except Exception:
            return []

    # -----------------------------------------------------------------
    # NEO4J: SESSION QUERIES
    # -----------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[SessionResult]:
        """Get a single session by ID."""
        driver = _get_neo4j()
        with driver.session() as s:
            r = s.run("""
                MATCH (s:ISMASession {session_id: $sid})
                RETURN s.session_id AS session_id, s.platform AS platform,
                       s.title AS title, s.source_file AS source_file,
                       s.exchange_count AS exchange_count,
                       s.created_at AS created_at, s.model AS model
            """, sid=session_id)
            rec = r.single()
            if not rec:
                return None
            return SessionResult(
                session_id=rec["session_id"],
                platform=rec["platform"] or "",
                title=rec["title"] or "",
                source_file=rec["source_file"] or "",
                exchange_count=rec["exchange_count"] or 0,
                created_at=rec["created_at"] or "",
                model=rec["model"] or "",
            )

    def list_sessions(self, platform: str = None,
                      limit: int = 100,
                      order_by: str = "created_at") -> List[SessionResult]:
        """List sessions, optionally filtered by platform."""
        driver = _get_neo4j()
        with driver.session() as s:
            if platform:
                r = s.run(f"""
                    MATCH (s:ISMASession {{platform: $platform}})
                    RETURN s ORDER BY s.{order_by} DESC LIMIT $limit
                """, platform=platform, limit=limit)
            else:
                r = s.run(f"""
                    MATCH (s:ISMASession)
                    RETURN s ORDER BY s.{order_by} DESC LIMIT $limit
                """, limit=limit)

            results = []
            for rec in r:
                node = rec["s"]
                results.append(SessionResult(
                    session_id=node.get("session_id", ""),
                    platform=node.get("platform", ""),
                    title=node.get("title", ""),
                    source_file=node.get("source_file", ""),
                    exchange_count=node.get("exchange_count", 0),
                    created_at=node.get("created_at", ""),
                    model=node.get("model", ""),
                ))
            return results

    def count_sessions(self, platform: str = None) -> int:
        """Count sessions, optionally by platform."""
        driver = _get_neo4j()
        with driver.session() as s:
            if platform:
                r = s.run("""
                    MATCH (s:ISMASession {platform: $platform})
                    RETURN count(s) AS c
                """, platform=platform)
            else:
                r = s.run("MATCH (s:ISMASession) RETURN count(s) AS c")
            return r.single()["c"]

    # -----------------------------------------------------------------
    # NEO4J: EXCHANGE QUERIES
    # -----------------------------------------------------------------

    def get_exchanges(self, session_id: str,
                      limit: int = 500) -> List[ExchangeResult]:
        """Get all exchanges for a session, ordered by index."""
        driver = _get_neo4j()
        with driver.session() as s:
            r = s.run("""
                MATCH (s:ISMASession {session_id: $sid})-[:CONTAINS]->(e:ISMAExchange)
                RETURN e ORDER BY e.exchange_index LIMIT $limit
            """, sid=session_id, limit=limit)

            results = []
            for rec in r:
                node = rec["e"]
                results.append(ExchangeResult(
                    content_hash=node.get("content_hash", ""),
                    session_id=node.get("session_id", ""),
                    exchange_index=node.get("exchange_index", 0),
                    user_prompt=node.get("user_prompt", ""),
                    response=node.get("response", ""),
                    timestamp=node.get("timestamp", ""),
                    model=node.get("model", ""),
                ))
            return results

    def get_exchange(self, content_hash: str) -> Optional[ExchangeResult]:
        """Get a single exchange by content hash."""
        driver = _get_neo4j()
        with driver.session() as s:
            r = s.run("""
                MATCH (e:ISMAExchange {content_hash: $hash})
                RETURN e
            """, hash=content_hash)
            rec = r.single()
            if not rec:
                return None
            node = rec["e"]
            return ExchangeResult(
                content_hash=node.get("content_hash", ""),
                session_id=node.get("session_id", ""),
                exchange_index=node.get("exchange_index", 0),
                user_prompt=node.get("user_prompt", ""),
                response=node.get("response", ""),
                timestamp=node.get("timestamp", ""),
                model=node.get("model", ""),
            )

    def count_exchanges(self, session_id: str = None) -> int:
        """Count exchanges, optionally for a specific session."""
        driver = _get_neo4j()
        with driver.session() as s:
            if session_id:
                r = s.run("""
                    MATCH (e:ISMAExchange {session_id: $sid})
                    RETURN count(e) AS c
                """, sid=session_id)
            else:
                r = s.run("MATCH (e:ISMAExchange) RETURN count(e) AS c")
            return r.single()["c"]

    def search_exchanges(self, text: str,
                         platform: str = None,
                         limit: int = 20) -> List[ExchangeResult]:
        """Full-text search across exchange prompts and responses."""
        driver = _get_neo4j()
        with driver.session() as s:
            safe = text.replace("'", "\\'")
            if platform:
                r = s.run("""
                    MATCH (s:ISMASession {platform: $platform})-[:CONTAINS]->(e:ISMAExchange)
                    WHERE e.user_prompt CONTAINS $text
                       OR e.response CONTAINS $text
                    RETURN e, s.platform AS platform
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                """, text=text, platform=platform, limit=limit)
            else:
                r = s.run("""
                    MATCH (e:ISMAExchange)
                    WHERE e.user_prompt CONTAINS $text
                       OR e.response CONTAINS $text
                    RETURN e
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                """, text=text, limit=limit)

            results = []
            for rec in r:
                node = rec["e"]
                results.append(ExchangeResult(
                    content_hash=node.get("content_hash", ""),
                    session_id=node.get("session_id", ""),
                    exchange_index=node.get("exchange_index", 0),
                    user_prompt=node.get("user_prompt", ""),
                    response=node.get("response", ""),
                    timestamp=node.get("timestamp", ""),
                    model=node.get("model", ""),
                ))
            return results

    # -----------------------------------------------------------------
    # NEO4J: DOCUMENT QUERIES
    # -----------------------------------------------------------------

    def get_document(self, content_hash: str) -> Optional[DocumentResult]:
        """Get a document by content hash."""
        driver = _get_neo4j()
        with driver.session() as s:
            r = s.run("""
                MATCH (d:Document {content_hash: $hash})
                RETURN d
            """, hash=content_hash)
            rec = r.single()
            if not rec:
                return None
            node = rec["d"]
            return DocumentResult(
                content_hash=node.get("content_hash", ""),
                document_id=node.get("document_id", ""),
                filename=node.get("filename", ""),
                all_paths=node.get("all_paths", []),
                layer=node.get("layer", ""),
                priority=node.get("priority", 0.0),
                file_size=node.get("file_size", 0),
                tile_count=node.get("tile_count", 0),
            )

    def list_documents(self, layer: str = None,
                       min_priority: float = None,
                       limit: int = 100) -> List[DocumentResult]:
        """List documents, optionally filtered."""
        driver = _get_neo4j()
        with driver.session() as s:
            conditions = []
            params = {"limit": limit}
            if layer:
                conditions.append("d.layer = $layer")
                params["layer"] = layer
            if min_priority is not None:
                conditions.append("d.priority >= $min_pri")
                params["min_pri"] = min_priority

            where = ""
            if conditions:
                where = "WHERE " + " AND ".join(conditions)

            r = s.run(f"""
                MATCH (d:Document)
                {where}
                RETURN d ORDER BY d.priority DESC, d.filename
                LIMIT $limit
            """, **params)

            results = []
            for rec in r:
                node = rec["d"]
                results.append(DocumentResult(
                    content_hash=node.get("content_hash", ""),
                    document_id=node.get("document_id", ""),
                    filename=node.get("filename", ""),
                    all_paths=node.get("all_paths", []),
                    layer=node.get("layer", ""),
                    priority=node.get("priority", 0.0),
                    file_size=node.get("file_size", 0),
                    tile_count=node.get("tile_count", 0),
                ))
            return results

    def count_documents(self, layer: str = None) -> int:
        """Count documents, optionally by layer."""
        driver = _get_neo4j()
        with driver.session() as s:
            if layer:
                r = s.run("""
                    MATCH (d:Document {layer: $layer})
                    RETURN count(d) AS c
                """, layer=layer)
            else:
                r = s.run("MATCH (d:Document) RETURN count(d) AS c")
            return r.single()["c"]

    # -----------------------------------------------------------------
    # THEME / MOTIF SEARCH
    # -----------------------------------------------------------------

    def search_by_theme(self, theme_id: str, top_k: int = 10,
                        include_supporting: bool = True,
                        **filters) -> SearchResult:
        """Search by theme_id, auto-expanding to constituent motifs.

        Uses the theme's display_name as vector search query and filters
        by the theme's motif list.
        """
        mapping = _get_canonical_mapping()
        theme = mapping["theme_registry"].get(theme_id)
        if not theme:
            return SearchResult(query=f"[theme:{theme_id}]", tiles=[])

        # Expand to motifs
        motifs = list(theme["required_motifs"])
        if include_supporting:
            motifs.extend(theme["supporting_motifs"])

        # Merge with any existing dominant_motifs filter
        existing_motifs = filters.pop("dominant_motifs", None)
        if existing_motifs:
            motifs = list(set(motifs + existing_motifs))

        return self.search(
            query=theme["display_name"],
            top_k=top_k,
            dominant_motifs=motifs,
            **filters,
        )

    def search_by_band(self, query: str, band: str, top_k: int = 10,
                       **filters) -> SearchResult:
        """Search filtered to motifs in a specific band (slow/mid/fast)."""
        motifs = _motifs_for_band(band)
        if not motifs:
            return SearchResult(query=query, tiles=[])

        existing_motifs = filters.pop("dominant_motifs", None)
        if existing_motifs:
            motifs = list(set(motifs + existing_motifs))

        return self.search(query=query, top_k=top_k,
                           dominant_motifs=motifs, **filters)

    @staticmethod
    def list_themes() -> List[Dict[str, Any]]:
        """List all 18 themes from the canonical mapping."""
        mapping = _get_canonical_mapping()
        result = []
        for tid, theme in sorted(mapping["theme_registry"].items()):
            result.append({
                "theme_id": tid,
                "display_name": theme["display_name"],
                "description": theme["description"],
                "required_motifs": theme["required_motifs"],
                "supporting_motifs": theme["supporting_motifs"],
            })
        return result

    @staticmethod
    def list_motifs(band: str = None) -> List[Dict[str, str]]:
        """List motifs, optionally filtered by band."""
        mapping = _get_canonical_mapping()
        result = []
        for mid, info in sorted(mapping["motif_registry"].items()):
            if band and info["band"] != band:
                continue
            result.append({
                "motif_id": mid,
                "band": info["band"],
                "definition": info["definition"],
            })
        return result

    # -----------------------------------------------------------------
    # HMM: MOTIF SEARCH
    # -----------------------------------------------------------------

    def motif_search(self, motif_id: str, min_amplitude: float = 0.5,
                     limit: int = 20) -> MotifSearchResult:
        """Search for tiles expressing a specific motif.

        Uses Redis inverted index for candidate selection, then Neo4j
        EXPRESSES edges for amplitude filtering and sorting.

        Args:
            motif_id: HMM motif ID (e.g., "HMM.SACRED_TRUST")
            min_amplitude: Minimum amplitude on the EXPRESSES edge
            limit: Maximum results to return

        Returns:
            MotifSearchResult with tile hashes and amplitude details.
        """
        r = _get_redis()

        # Step 1: Get candidate tile hashes from Redis inverted index
        candidate_hashes = r.smembers(f"hmm:inv:{motif_id}")
        total_candidates = len(candidate_hashes)

        if not candidate_hashes:
            return MotifSearchResult(
                motif_id=motif_id, tile_hashes=[],
                tiles_with_amplitude=[], total_candidates=0,
            )

        # Step 2: Query Neo4j for amplitude filtering and rosetta summaries
        driver = _get_neo4j()
        tiles_with_amp = []
        with driver.session() as s:
            result = s.run("""
                MATCH (t:HMMTile)-[r:EXPRESSES]->(m:HMMMotif {motif_id: $motif_id})
                WHERE r.amp >= $min_amp
                RETURN t.tile_id AS tile_hash,
                       t.rosetta_summary AS rosetta_summary,
                       t.dominant_motifs AS dominant_motifs,
                       t.platform AS platform,
                       r.amp AS amplitude,
                       r.confidence AS confidence
                ORDER BY r.amp DESC
                LIMIT $limit
            """, motif_id=motif_id, min_amp=min_amplitude, limit=limit)

            for rec in result:
                tiles_with_amp.append({
                    "tile_hash": rec["tile_hash"],
                    "rosetta_summary": rec["rosetta_summary"] or "",
                    "dominant_motifs": rec["dominant_motifs"] or [],
                    "platform": rec["platform"] or "",
                    "amplitude": rec["amplitude"],
                    "confidence": rec["confidence"],
                })

        return MotifSearchResult(
            motif_id=motif_id,
            tile_hashes=[t["tile_hash"] for t in tiles_with_amp],
            tiles_with_amplitude=tiles_with_amp,
            total_candidates=total_candidates,
        )

    # -----------------------------------------------------------------
    # HMM: RERANK WITH ROSETTA SUMMARIES
    # -----------------------------------------------------------------

    def hmm_rerank(self, results: SearchResult,
                   query: str,
                   rosetta_weight: float = 0.3,
                   motif_weight: float = 0.2,
                   query_type: str = "default",
                   instruction: str = "") -> SearchResult:
        """Rerank search results using neural cross-encoder or HMM formula fallback.

        Primary path: Qwen3-Reranker-8B cross-encoder on Spark 2 (port 8085).
        Fallback: Hand-tuned formula (0.5*base + 0.3*rosetta + 0.2*motif).

        Args:
            results: Original SearchResult from search()
            query: The original query text
            rosetta_weight: Weight for rosetta similarity (fallback only)
            motif_weight: Weight for motif overlap (fallback only)
            query_type: Query category for instruction selection (reranker)
            instruction: Custom reranker instruction (overrides query_type)

        Returns:
            New SearchResult with reranked tiles.
        """
        if not results.tiles:
            return results

        # Try neural reranker first
        try:
            from isma.src.reranker import get_reranker
            reranker = get_reranker()
            if reranker.is_available():
                reranked_tiles = reranker.rerank(
                    query, results.tiles,
                    instruction=instruction,
                    query_type=query_type,
                )
                return SearchResult(
                    query=results.query,
                    tiles=reranked_tiles,
                    total_tokens=results.total_tokens,
                    search_time_ms=results.search_time_ms,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Reranker unavailable: %s", e)

        # Fallback: original hand-tuned formula
        return self._hmm_rerank_formula(
            results, query,
            rosetta_weight=rosetta_weight,
            motif_weight=motif_weight,
        )

    def _hmm_rerank_formula(self, results: SearchResult,
                            query: str,
                            rosetta_weight: float = 0.3,
                            motif_weight: float = 0.2) -> SearchResult:
        """Fallback reranking using hand-tuned HMM formula.

        Blends: base_score * 0.5 + rosetta_sim * 0.3 + motif_overlap * 0.2
        Used when the neural reranker is unavailable.
        """
        from isma.src.hmm.motifs import assign_motifs

        if not results.tiles:
            return results

        # Step 1: Compile query to motifs for overlap scoring
        query_motifs = assign_motifs(query)
        query_motif_ids = {a.motif_id for a in query_motifs}

        # Step 2: Collect rosetta summaries from enriched tiles
        rosetta_texts = []
        rosetta_indices = []
        for i, tile in enumerate(results.tiles):
            if tile.hmm_enriched and tile.rosetta_summary:
                rosetta_texts.append(tile.rosetta_summary)
                rosetta_indices.append(i)

        # Step 3: Embed query and rosetta summaries together for cosine sim
        rosetta_sims = {}
        if rosetta_texts:
            all_texts = [query] + rosetta_texts
            embeddings = _get_embeddings_batch(all_texts)
            if embeddings and len(embeddings) == len(all_texts):
                query_vec = embeddings[0]
                for j, idx in enumerate(rosetta_indices):
                    rosetta_vec = embeddings[j + 1]
                    sim = _cosine_similarity(query_vec, rosetta_vec)
                    rosetta_sims[idx] = sim

        # Step 4: Compute blended scores
        scored = []
        for i, tile in enumerate(results.tiles):
            try:
                base_score = float(tile.score) if tile.score else 0.0
            except (ValueError, TypeError):
                base_score = 0.0

            # Rosetta similarity boost
            rosetta_boost = rosetta_sims.get(i, 0.0)

            # Motif overlap boost
            motif_boost = 0.0
            if tile.hmm_enriched and tile.dominant_motifs:
                tile_motif_set = set(tile.dominant_motifs)
                if query_motif_ids:
                    overlap = len(query_motif_ids & tile_motif_set)
                    motif_boost = overlap / len(query_motif_ids)

            blended = (
                base_score * (1.0 - rosetta_weight - motif_weight)
                + rosetta_boost * rosetta_weight
                + motif_boost * motif_weight
            )
            scored.append((i, blended))

        # Step 5: Sort by blended score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked_tiles = [results.tiles[i] for i, _ in scored]

        return SearchResult(
            query=results.query,
            tiles=reranked_tiles,
            total_tokens=results.total_tokens,
            search_time_ms=results.search_time_ms,
        )

    # -----------------------------------------------------------------
    # HMM: GRAPH EXPANSION
    # -----------------------------------------------------------------

    def graph_expand(self, tile_hash: str,
                     depth: int = 1) -> GraphExpansionResult:
        """Expand a tile's cross-references via RELATES_TO edges in Neo4j.

        Follows RELATES_TO edges up to the specified depth to find
        related tiles. Returns related tiles with their relationship
        metadata (type, note) and rosetta summaries.

        HMMTile nodes use 16-char hash prefixes as tile_id. If a longer
        content_hash is passed, the first 16 chars are used for lookup.

        Args:
            tile_hash: Content hash or HMM tile_id of the source tile
            depth: How many hops to follow (1 = direct neighbors only)

        Returns:
            GraphExpansionResult with related tiles and metadata.
        """
        driver = _get_neo4j()
        related = []

        # HMMTile uses 16-char tile_id prefixes
        hmm_id = tile_hash[:16]

        with driver.session() as s:
            # Use variable-length path for depth > 1
            result = s.run("""
                MATCH path = (source:HMMTile {tile_id: $tile_hash})
                             -[:RELATES_TO*1..""" + str(depth) + """]->(target:HMMTile)
                WHERE source <> target
                WITH target, relationships(path) AS rels,
                     length(path) AS hop_distance
                ORDER BY hop_distance, target.tile_id
                WITH target, head(rels) AS first_rel, hop_distance
                RETURN DISTINCT target.tile_id AS tile_hash,
                       target.rosetta_summary AS rosetta_summary,
                       target.dominant_motifs AS dominant_motifs,
                       target.platform AS platform,
                       first_rel.type AS rel_type,
                       first_rel.note AS rel_note,
                       hop_distance AS distance
            """, tile_hash=hmm_id)

            seen = set()
            for rec in result:
                th = rec["tile_hash"]
                if th in seen:
                    continue
                seen.add(th)
                related.append({
                    "tile_hash": th,
                    "rosetta_summary": rec["rosetta_summary"] or "",
                    "dominant_motifs": rec["dominant_motifs"] or [],
                    "platform": rec["platform"] or "",
                    "rel_type": rec["rel_type"] or "",
                    "rel_note": rec["rel_note"] or "",
                    "distance": rec["distance"],
                })

        return GraphExpansionResult(
            source_tile_hash=hmm_id,
            related_tiles=related,
            depth=depth,
        )

    # -----------------------------------------------------------------
    # HMM-ENHANCED HYBRID SEARCH
    # -----------------------------------------------------------------

    def hybrid_retrieve_hmm(self, query: str, top_k: int = 10,
                            hmm_rerank_enabled: bool = True,
                            expand_graph: bool = False,
                            graph_depth: int = 1,
                            expand_to_session: bool = False,
                            expand_to_document: bool = False,
                            rosetta_weight: float = 0.3,
                            motif_weight: float = 0.2,
                            query_type: str = "default",
                            instruction: str = "",
                            **filters) -> Dict[str, Any]:
        """Full hybrid retrieval with HMM enrichment.

        Extends hybrid_retrieve() by:
        1. Running vector search (same as hybrid_retrieve)
        2. Reranking via neural cross-encoder (or fallback formula)
        3. Optionally expanding top results via RELATES_TO graph edges
        4. Optionally expanding to sessions/documents

        Args:
            query: Natural language search query
            top_k: Number of results to return
            hmm_rerank_enabled: Whether to apply reranking
            expand_graph: Whether to expand top results via RELATES_TO
            graph_depth: Depth for graph expansion
            expand_to_session: Expand to full sessions via Neo4j
            expand_to_document: Expand to full documents via Neo4j
            rosetta_weight: Weight for rosetta similarity (fallback)
            motif_weight: Weight for motif overlap (fallback)
            query_type: Query category for reranker instruction
            instruction: Custom reranker instruction
            **filters: Additional Weaviate filters
        """
        import time
        t0 = time.time()

        # Step 1: Vector search (fetch 3x for reranker candidate pool)
        fetch_k = top_k * 3 if hmm_rerank_enabled else top_k
        search_result = self.search(query, top_k=fetch_k,
                                    expand_parents=True, **filters)

        # Step 2: Reranking (neural cross-encoder with formula fallback)
        if hmm_rerank_enabled and search_result.tiles:
            search_result = self.hmm_rerank(
                search_result, query,
                rosetta_weight=rosetta_weight,
                motif_weight=motif_weight,
                query_type=query_type,
                instruction=instruction,
            )

        # Step 2.5: Deduplicate by content_hash (keep first = best ranked)
        # Multi-scale tiles (search_512, context_2048, full_4096) cause the same
        # content_hash to appear multiple times. Keep best-ranked occurrence only.
        if search_result.tiles:
            seen_hashes = set()
            deduped = []
            for tile in search_result.tiles:
                key = tile.content_hash or id(tile)
                if key not in seen_hashes:
                    seen_hashes.add(key)
                    deduped.append(tile)
            search_result = SearchResult(
                query=search_result.query,
                tiles=deduped[:top_k],
                total_tokens=sum(t.token_count for t in deduped[:top_k]),
                search_time_ms=search_result.search_time_ms,
            )

        result = {
            "query": query,
            "tiles": search_result.tiles,
            "total_tokens": search_result.total_tokens,
            "search_time_ms": (time.time() - t0) * 1000,
            "hmm_reranked": hmm_rerank_enabled,
            "sessions": {},
            "documents": {},
            "graph_expansions": {},
        }

        # Step 3: Graph expansion for enriched tiles
        if expand_graph:
            for tile in search_result.tiles:
                if tile.hmm_enriched and tile.content_hash:
                    expansion = self.graph_expand(tile.content_hash,
                                                  depth=graph_depth)
                    if expansion.related_tiles:
                        result["graph_expansions"][tile.content_hash] = {
                            "related": expansion.related_tiles,
                            "depth": expansion.depth,
                        }

        # Step 4: Session/document expansion (same as hybrid_retrieve)
        if expand_to_session:
            seen_sessions = set()
            for tile in search_result.tiles:
                if tile.session_id and tile.session_id not in seen_sessions:
                    seen_sessions.add(tile.session_id)
                    session = self.get_session(tile.session_id)
                    if session:
                        result["sessions"][tile.session_id] = session

        if expand_to_document:
            seen_docs = set()
            for tile in search_result.tiles:
                if tile.document_id and tile.document_id not in seen_docs:
                    seen_docs.add(tile.document_id)
                    doc = self.get_document(tile.content_hash)
                    if doc:
                        result["documents"][tile.content_hash] = doc

        return result

    # -----------------------------------------------------------------
    # AGGREGATE STATS
    # -----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Get aggregate stats from all stores including HMM."""
        result = {}

        # Weaviate
        try:
            r = self._session.post(f"{WEAVIATE_URL}/v1/graphql", json={
                "query": """{ Aggregate { ISMA_Quantum { meta { count } } } }"""
            }, timeout=10)
            result["weaviate_tiles"] = (
                r.json()["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]
            )
        except Exception:
            result["weaviate_tiles"] = -1

        # Neo4j (ISMA)
        try:
            driver = _get_neo4j()
            with driver.session() as s:
                for label in ["ISMASession", "ISMAExchange", "Document"]:
                    r = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                    result[f"neo4j_{label}"] = r.single()["c"]
        except Exception:
            result["neo4j_error"] = True

        # Neo4j (HMM)
        try:
            driver = _get_neo4j()
            with driver.session() as s:
                for label in ["HMMTile", "HMMMotif"]:
                    r = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                    result[f"hmm_{label}"] = r.single()["c"]
                r = s.run("MATCH ()-[r:EXPRESSES]->() RETURN count(r) AS c")
                result["hmm_EXPRESSES"] = r.single()["c"]
                r = s.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c")
                result["hmm_RELATES_TO"] = r.single()["c"]
        except Exception:
            result["hmm_neo4j_error"] = True

        # Redis (HMM inverted index)
        try:
            rc = _get_redis()
            inv_keys = list(rc.scan_iter("hmm:inv:*", count=1000))
            result["hmm_redis_motifs"] = len(inv_keys)
            total_inv = sum(rc.scard(k) for k in inv_keys)
            result["hmm_redis_inv_entries"] = total_inv
        except Exception:
            result["hmm_redis_error"] = True

        return result

    # -----------------------------------------------------------------
    # CONTENT RECONSTRUCTION
    # -----------------------------------------------------------------

    def get_full_text(self, content_hash: str) -> str:
        """Reconstruct full text from full_4096 tiles for a content hash.

        Returns the complete text of a document or exchange by assembling
        all full_4096 scale tiles in order.
        """
        tiles = self.get_tiles_for_content(content_hash, scale="full_4096")
        if not tiles:
            # Fall back to context_2048
            tiles = self.get_tiles_for_content(content_hash, scale="context_2048")
        if not tiles:
            # Fall back to any scale
            tiles = self.get_tiles_for_content(content_hash)

        tiles.sort(key=lambda t: t.start_char)
        return "\n".join(t.content for t in tiles)

    def get_session_full_text(self, session_id: str) -> str:
        """Get complete conversation text for a session.

        Retrieves all exchanges from Neo4j and formats them.
        """
        exchanges = self.get_exchanges(session_id)
        parts = []
        for ex in exchanges:
            parts.append(f"[User]: {ex.user_prompt}")
            parts.append(f"[Assistant]: {ex.response}")
            parts.append("")
        return "\n".join(parts)

    # -----------------------------------------------------------------
    # HYBRID SEARCH (Vector + Graph)
    # -----------------------------------------------------------------

    def hybrid_retrieve(self, query: str, top_k: int = 10,
                        expand_to_session: bool = False,
                        expand_to_document: bool = False,
                        **filters) -> Dict[str, Any]:
        """Full hybrid retrieval: vector search + graph expansion.

        Returns tiles from vector search, plus optionally expands
        to full sessions or documents via Neo4j.
        """
        search_result = self.search(query, top_k=top_k,
                                    expand_parents=True, **filters)

        result = {
            "query": query,
            "tiles": search_result.tiles,
            "total_tokens": search_result.total_tokens,
            "search_time_ms": search_result.search_time_ms,
            "sessions": {},
            "documents": {},
        }

        if expand_to_session:
            seen_sessions = set()
            for tile in search_result.tiles:
                if tile.session_id and tile.session_id not in seen_sessions:
                    seen_sessions.add(tile.session_id)
                    session = self.get_session(tile.session_id)
                    if session:
                        result["sessions"][tile.session_id] = session

        if expand_to_document:
            seen_docs = set()
            for tile in search_result.tiles:
                if tile.document_id and tile.document_id not in seen_docs:
                    seen_docs.add(tile.document_id)
                    doc = self.get_document(tile.content_hash)
                    if doc:
                        result["documents"][tile.content_hash] = doc

        return result

    # -----------------------------------------------------------------
    # ITERATION (for bulk processing)
    # -----------------------------------------------------------------

    def iter_all_sessions(self, platform: str = None,
                          batch_size: int = 100):
        """Iterate all sessions in batches. Yields SessionResult objects."""
        offset = 0
        while True:
            driver = _get_neo4j()
            with driver.session() as s:
                if platform:
                    r = s.run("""
                        MATCH (s:ISMASession {platform: $platform})
                        RETURN s ORDER BY s.session_id
                        SKIP $offset LIMIT $limit
                    """, platform=platform, offset=offset, limit=batch_size)
                else:
                    r = s.run("""
                        MATCH (s:ISMASession)
                        RETURN s ORDER BY s.session_id
                        SKIP $offset LIMIT $limit
                    """, offset=offset, limit=batch_size)

                batch = []
                for rec in r:
                    node = rec["s"]
                    batch.append(SessionResult(
                        session_id=node.get("session_id", ""),
                        platform=node.get("platform", ""),
                        title=node.get("title", ""),
                        source_file=node.get("source_file", ""),
                        exchange_count=node.get("exchange_count", 0),
                        created_at=node.get("created_at", ""),
                        model=node.get("model", ""),
                    ))

            if not batch:
                break
            yield from batch
            offset += batch_size

    def iter_all_documents(self, layer: str = None,
                           batch_size: int = 100):
        """Iterate all documents in batches. Yields DocumentResult objects."""
        offset = 0
        while True:
            driver = _get_neo4j()
            with driver.session() as s:
                if layer:
                    r = s.run("""
                        MATCH (d:Document {layer: $layer})
                        RETURN d ORDER BY d.content_hash
                        SKIP $offset LIMIT $limit
                    """, layer=layer, offset=offset, limit=batch_size)
                else:
                    r = s.run("""
                        MATCH (d:Document)
                        RETURN d ORDER BY d.content_hash
                        SKIP $offset LIMIT $limit
                    """, offset=offset, limit=batch_size)

                batch = []
                for rec in r:
                    node = rec["d"]
                    batch.append(DocumentResult(
                        content_hash=node.get("content_hash", ""),
                        document_id=node.get("document_id", ""),
                        filename=node.get("filename", ""),
                        all_paths=node.get("all_paths", []),
                        layer=node.get("layer", ""),
                        priority=node.get("priority", 0.0),
                        file_size=node.get("file_size", 0),
                        tile_count=node.get("tile_count", 0),
                    ))

            if not batch:
                break
            yield from batch
            offset += batch_size


# =============================================================================
# CONVENIENCE (module-level singleton)
# =============================================================================

_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def get_retrieval() -> ISMARetrieval:
    """Get the singleton ISMARetrieval instance."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = ISMARetrieval()
    return _INSTANCE
