"""
ISMA Query API - FastAPI wrapper around ISMARetrieval.

Provides HTTP endpoints for semantic search, motif exploration,
graph traversal, and stats across the ISMA + HMM data stores.

Usage:
    uvicorn isma.src.query_api:app --host 0.0.0.0 --port 8095

Endpoints:
    GET  /health              - Health check
    GET  /stats               - Aggregate stats from all stores
    POST /search              - Semantic vector search
    POST /search/hmm          - HMM-enhanced hybrid retrieval
    POST /search/motif        - Search by motif ID
    POST /search/bm25         - Keyword search (BM25)
    GET  /motifs              - List all motifs
    GET  /themes              - List all themes
    GET  /session/{id}        - Get session details
    GET  /session/{id}/text   - Get full session text
    GET  /document/{hash}     - Get document details
    GET  /document/{hash}/text - Get full document text
    GET  /tile/{hash}         - Get all tiles for content hash
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from dataclasses import asdict
import json
import logging
import os
import sys
import tempfile
import time
import threading

from isma.src.retrieval import ISMARetrieval, TileResult, SearchResult
from isma.src.semantic_cache import SemanticCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Increase thread pool for sync endpoints (default 40 is too low)."""
    import anyio
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 100
    yield


app = FastAPI(
    title="ISMA Query API",
    description="Semantic search over 993K embedded tiles with HMM enrichment",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.100.10:8095"],  # NCCL fabric only
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Singleton retrieval instance (thread-safe)
_retrieval = None
_retrieval_lock = threading.Lock()

def get_retrieval() -> ISMARetrieval:
    global _retrieval
    if _retrieval is None:
        with _retrieval_lock:
            if _retrieval is None:
                _retrieval = ISMARetrieval()
    return _retrieval


# Singleton cache instance (thread-safe, avoids new Redis connection per request)
_cache = None
_cache_lock = threading.Lock()

def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SemanticCache()
    return _cache


# ── Request Models ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    expand_parents: bool = False
    platform: Optional[str] = None
    source_type: Optional[str] = None
    scale: Optional[str] = None
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    has_artifacts: Optional[bool] = None
    has_thinking: Optional[bool] = None
    layer: Optional[int] = None
    min_priority: Optional[float] = None
    model: Optional[str] = None
    dominant_motifs: Optional[List[str]] = None
    hmm_enriched: Optional[bool] = None
    min_hmm_phi: Optional[float] = None
    min_hmm_trust: Optional[float] = None
    theme_id: Optional[str] = None
    motif_band: Optional[str] = None


class HMMSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    hmm_rerank: bool = True
    expand_graph: bool = False
    graph_depth: int = 1
    expand_to_session: bool = False
    expand_to_document: bool = False
    rosetta_weight: float = 0.3
    motif_weight: float = 0.2
    query_type: str = "default"
    instruction: Optional[str] = None
    platform: Optional[str] = None
    source_type: Optional[str] = None
    hmm_enriched: Optional[bool] = None


class MotifSearchRequest(BaseModel):
    motif_id: str
    min_amplitude: float = 0.5
    top_k: int = Field(default=20, ge=1, le=100)
    platform: Optional[str] = None


class BM25Request(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    platform: Optional[str] = None
    source_type: Optional[str] = None


class HMMStoreRequest(BaseModel):
    platform: str
    content: str
    pkg_id: Optional[str] = None


# ── Helpers ─────────────────────────────────────────────────────

def _tile_to_dict(t: TileResult) -> dict:
    d = asdict(t)
    # Trim empty fields for cleaner output
    return {k: v for k, v in d.items()
            if v and v != 0 and v != 0.0 and v != [] and v != ""}


def _search_result_to_dict(sr: SearchResult) -> dict:
    return {
        "query": sr.query,
        "total_tokens": sr.total_tokens,
        "search_time_ms": round(sr.search_time_ms, 1),
        "count": len(sr.tiles),
        "tiles": [_tile_to_dict(t) for t in sr.tiles],
    }


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "isma-query-api", "timestamp": time.time()}


@app.get("/stats")
def stats():
    r = get_retrieval()
    return r.stats()


@app.post("/search")
def search(req: SearchRequest):
    r = get_retrieval()
    filters = {}
    for field_name in ["platform", "source_type", "scale", "session_id",
                       "document_id", "has_artifacts", "has_thinking",
                       "layer", "min_priority", "model", "dominant_motifs",
                       "hmm_enriched", "min_hmm_phi", "min_hmm_trust",
                       "theme_id", "motif_band"]:
        val = getattr(req, field_name)
        if val is not None:
            filters[field_name] = val

    result = r.search(
        query=req.query,
        top_k=req.top_k,
        expand_parents=req.expand_parents,
        **filters,
    )
    return _search_result_to_dict(result)


@app.post("/search/hmm")
def search_hmm(req: HMMSearchRequest):
    r = get_retrieval()
    filters = {}
    for field_name in ["platform", "source_type", "hmm_enriched"]:
        val = getattr(req, field_name)
        if val is not None:
            filters[field_name] = val

    result = r.hybrid_retrieve_hmm(
        query=req.query,
        top_k=req.top_k,
        hmm_rerank_enabled=req.hmm_rerank,
        expand_graph=req.expand_graph,
        graph_depth=req.graph_depth,
        expand_to_session=req.expand_to_session,
        expand_to_document=req.expand_to_document,
        rosetta_weight=req.rosetta_weight,
        motif_weight=req.motif_weight,
        query_type=req.query_type,
        instruction=req.instruction or "",
        **filters,
    )

    # Convert tiles in result
    tiles = result.get("tiles", [])
    result["tiles"] = [_tile_to_dict(t) if isinstance(t, TileResult) else t
                       for t in tiles]
    result["count"] = len(result["tiles"])
    result["search_time_ms"] = round(result.get("search_time_ms", 0), 1)

    # Convert session/document results
    for key in ["sessions", "documents", "graph_expansions"]:
        sub = result.get(key, {})
        if sub:
            for k, v in sub.items():
                if hasattr(v, '__dataclass_fields__'):
                    sub[k] = asdict(v)
    return result


@app.post("/search/motif")
def search_motif(req: MotifSearchRequest):
    r = get_retrieval()
    result = r.motif_search(
        motif_id=req.motif_id,
        min_amplitude=req.min_amplitude,
        limit=req.top_k,
    )
    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    return result


@app.post("/search/bm25")
def search_bm25(req: BM25Request):
    r = get_retrieval()
    filters = {}
    if req.platform:
        filters["platform"] = req.platform
    if req.source_type:
        filters["source_type"] = req.source_type

    result = r.search_bm25(
        query=req.query,
        top_k=req.top_k,
        **filters,
    )
    return _search_result_to_dict(result)


@app.get("/motifs")
def list_motifs(band: Optional[str] = None):
    return ISMARetrieval.list_motifs(band=band)


@app.get("/themes")
def list_themes():
    return ISMARetrieval.list_themes()


@app.get("/session/{session_id}")
def get_session(session_id: str):
    r = get_retrieval()
    result = r.get_session(session_id)
    if result is None:
        return {"error": "Session not found"}
    return asdict(result)


@app.get("/session/{session_id}/text")
def get_session_text(session_id: str):
    r = get_retrieval()
    text = r.get_session_full_text(session_id)
    return {"session_id": session_id, "text": text, "length": len(text)}


@app.get("/session/{session_id}/exchanges")
def get_exchanges(session_id: str):
    r = get_retrieval()
    exchanges = r.get_exchanges(session_id)
    return {"session_id": session_id, "count": len(exchanges),
            "exchanges": [asdict(e) for e in exchanges]}


@app.get("/document/{content_hash}")
def get_document(content_hash: str):
    r = get_retrieval()
    result = r.get_document(content_hash)
    if result is None:
        return {"error": "Document not found"}
    return asdict(result)


@app.get("/document/{content_hash}/text")
def get_document_text(content_hash: str):
    r = get_retrieval()
    text = r.get_full_text(content_hash)
    return {"content_hash": content_hash, "text": text, "length": len(text)}


@app.get("/tiles/{content_hash}")
def get_tiles(content_hash: str, scale: Optional[str] = None):
    r = get_retrieval()
    tiles = r.get_tiles_for_content(content_hash, scale=scale)
    return {"content_hash": content_hash, "count": len(tiles),
            "tiles": [_tile_to_dict(t) for t in tiles]}


@app.get("/sessions")
def list_sessions(
    platform: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
):
    r = get_retrieval()
    sessions = r.list_sessions(platform=platform, limit=limit)
    total = r.count_sessions(platform=platform)
    return {
        "total": total,
        "limit": limit,
        "sessions": [asdict(s) for s in sessions],
    }


@app.get("/documents")
def list_documents(
    layer: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
):
    r = get_retrieval()
    docs = r.list_documents(layer=layer, limit=limit)
    total = r.count_documents(layer=layer)
    return {
        "total": total,
        "limit": limit,
        "documents": [asdict(d) for d in docs],
    }


# ── V2 Endpoints (Shadow Deployment) ─────────────────────────

class V2SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True
    query_type: str = "default"
    instruction: Optional[str] = None
    platform: Optional[str] = None
    source_type: Optional[str] = None
    hmm_enriched: Optional[bool] = None


@app.get("/v2/stats")
def v2_stats():
    from isma.src.retrieval_v2 import get_retrieval_v2
    r = get_retrieval_v2()
    return r.stats()


@app.post("/v2/search")
def v2_search(req: V2SearchRequest):
    from isma.src.retrieval_v2 import get_retrieval_v2
    r = get_retrieval_v2()

    if not r.is_available():
        raise HTTPException(status_code=503, detail="V2 class not available")

    filters = {}
    for field_name in ["platform", "source_type", "hmm_enriched"]:
        val = getattr(req, field_name)
        if val is not None:
            filters[field_name] = val

    result = r.search(req.query, top_k=req.top_k, **filters)
    return _search_result_to_dict(result)


@app.post("/v2/search/hmm")
def v2_search_hmm(req: V2SearchRequest):
    from isma.src.retrieval_v2 import get_retrieval_v2
    r = get_retrieval_v2()

    if not r.is_available():
        raise HTTPException(status_code=503, detail="V2 class not available")

    filters = {}
    for field_name in ["platform", "source_type", "hmm_enriched"]:
        val = getattr(req, field_name)
        if val is not None:
            filters[field_name] = val

    result = r.hybrid_search(
        req.query,
        top_k=req.top_k,
        rerank=req.rerank,
        query_type=req.query_type,
        instruction=req.instruction or "",
        **filters,
    )

    tiles = result.get("tiles", [])
    result["tiles"] = [_tile_to_dict(t) if isinstance(t, TileResult) else t for t in tiles]
    result["count"] = len(result["tiles"])
    result["search_time_ms"] = round(result.get("search_time_ms", 0), 1)
    return result


@app.get("/v2/expand/{content_hash}")
def v2_expand(content_hash: str, scale: Optional[str] = "search_512"):
    from isma.src.retrieval_v2 import get_retrieval_v2
    r = get_retrieval_v2()
    tiles = r.expand_passages(content_hash, scale=scale or "search_512")
    return {
        "content_hash": content_hash,
        "scale": scale,
        "count": len(tiles),
        "tiles": [_tile_to_dict(t) for t in tiles],
    }


@app.post("/v2/search/adaptive")
def v2_search_adaptive(req: V2SearchRequest):
    from isma.src.retrieval_v2 import get_retrieval_v2
    from isma.src.semantic_cache import SemanticCache

    r = get_retrieval_v2()
    if not r.is_available():
        raise HTTPException(status_code=503, detail="V2 class not available")

    filters = {}
    for field_name in ["platform", "source_type", "hmm_enriched"]:
        val = getattr(req, field_name)
        if val is not None:
            filters[field_name] = val

    # Phase 5: Check semantic cache first (includes filters in key)
    try:
        cache = _get_cache()
        cached = cache.get(req.query, query_type="adaptive", top_k=req.top_k, **filters)
        if cached:
            cached_result = cached.get("result", cached)
            cached_result["cache_hit"] = True
            return cached_result
    except Exception:
        pass  # Cache failure is non-fatal

    result = r.adaptive_search(
        req.query,
        top_k=req.top_k,
        **filters,
    )

    tiles = result.get("tiles", [])
    result["tiles"] = [_tile_to_dict(t) if isinstance(t, TileResult) else t for t in tiles]
    result["count"] = len(result["tiles"])
    result["search_time_ms"] = round(result.get("search_time_ms", 0), 1)
    result["cache_hit"] = False

    # Phase 5: Store in cache — use same query_type as read for key consistency
    try:
        cache.put(req.query, result, query_type="adaptive", top_k=req.top_k, **filters)
    except Exception:
        pass  # Cache failure is non-fatal

    return result


@app.post("/v2/search/retry")
def v2_search_retry(req: V2SearchRequest):
    """Adaptive search with agentic retry on low-quality results.

    If the first attempt returns results with top score < 0.3,
    retries once with expanded strategy and loosened filters.
    """
    from isma.src.retrieval_v2 import get_retrieval_v2
    from isma.src.agentic_retry import retrieval_with_retry

    r = get_retrieval_v2()
    if not r.is_available():
        raise HTTPException(status_code=503, detail="V2 class not available")

    filters = {}
    for field_name in ["platform", "source_type", "hmm_enriched"]:
        val = getattr(req, field_name)
        if val is not None:
            filters[field_name] = val

    result = retrieval_with_retry(
        req.query,
        top_k=req.top_k,
        **filters,
    )

    tiles = result.get("tiles", [])
    result["tiles"] = [_tile_to_dict(t) if isinstance(t, TileResult) else t for t in tiles]
    result["count"] = len(result["tiles"])
    result["search_time_ms"] = round(result.get("search_time_ms", 0), 1)
    return result


# ── V2 Phase 4: Temporal Truth Endpoints ─────────────────────

@app.get("/v2/timeline/{content_hash}")
def v2_timeline(content_hash: str):
    """Get the temporal version chain for a content_hash.

    Shows all enrichment versions from newest to oldest,
    following SUPERSEDES edges.
    """
    from isma.src.hmm.neo4j_store import HMMNeo4jStore
    store = HMMNeo4jStore()
    try:
        chain = store.get_temporal_chain(content_hash)
        return {
            "content_hash": content_hash,
            "versions": len(chain),
            "chain": chain,
        }
    finally:
        store.close()


@app.get("/v2/contradictions")
def v2_contradictions(
    min_confidence: float = Query(default=0.5, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List detected contradictions across the knowledge base."""
    from isma.src.hmm.neo4j_store import HMMNeo4jStore
    store = HMMNeo4jStore()
    try:
        contradictions = store.get_contradictions(
            min_confidence=min_confidence, limit=limit,
        )
        return {
            "count": len(contradictions),
            "min_confidence": min_confidence,
            "contradictions": contradictions,
        }
    finally:
        store.close()


@app.get("/v2/session/{session_id}/reconstruct")
def v2_reconstruct_session(session_id: str):
    """Reconstruct a session from its HMM-enriched tiles.

    Returns tiles in exchange order, preferring latest versions
    (following SUPERSEDES chains). Includes contradiction info.
    """
    from isma.src.hmm.neo4j_store import HMMNeo4jStore
    store = HMMNeo4jStore()
    try:
        tiles = store.reconstruct_session(session_id)
        return {
            "session_id": session_id,
            "tile_count": len(tiles),
            "tiles": tiles,
        }
    finally:
        store.close()


@app.get("/v2/tile/{tile_id}/contradictions")
def v2_tile_contradictions(tile_id: str):
    """Get all contradictions for a specific tile."""
    from isma.src.hmm.neo4j_store import HMMNeo4jStore
    store = HMMNeo4jStore()
    try:
        contradictions = store.get_tile_contradictions(tile_id)
        return {
            "tile_id": tile_id,
            "count": len(contradictions),
            "contradictions": contradictions,
        }
    finally:
        store.close()


@app.post("/v2/contradictions/check")
def v2_check_contradictions(limit: int = Query(default=100, ge=1, le=1000)):
    """Trigger batch contradiction verification.

    Scans RELATES_TO {type: 'contradicts'} edges that don't yet
    have a corresponding CONTRADICTS edge and verifies them via
    the cross-encoder reranker.
    """
    from isma.src.contradiction_detector import check_contradictions_batch
    results = check_contradictions_batch(limit=limit)
    return {
        "checked": limit,
        "confirmed": len(results),
        "contradictions": results,
    }


@app.get("/v2/cache/stats")
def v2_cache_stats():
    """Get semantic cache statistics."""
    from isma.src.semantic_cache import SemanticCache
    cache = SemanticCache()
    return cache.stats()


@app.post("/v2/cache/clear")
def v2_cache_clear():
    """Clear all semantic cache entries."""
    from isma.src.semantic_cache import SemanticCache
    cache = SemanticCache()
    cache.clear()
    return {"status": "cleared"}


@app.post("/v2/backfill/session-links")
def v2_backfill_session_links(limit: int = Query(default=5000, ge=1, le=50000)):
    """Backfill IN_SESSION edges between HMMTiles and ISMASessions.

    Links tiles to their originating sessions via shared content_hash
    in ISMAExchange nodes.
    """
    from isma.src.hmm.neo4j_store import HMMNeo4jStore
    store = HMMNeo4jStore()
    try:
        created = store.backfill_session_links(limit=limit)
        return {
            "created": created,
            "limit": limit,
        }
    finally:
        store.close()


# ── HMM Store Endpoint ────────────────────────────────────────

_hmm_store_imported = False
_process_response = None

def _get_process_response():
    """Lazy import of hmm_store_results.process_response."""
    global _hmm_store_imported, _process_response
    if not _hmm_store_imported:
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        )
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from hmm_store_results import process_response
        _process_response = process_response
        _hmm_store_imported = True
    return _process_response


@app.post("/hmm/store-response")
def hmm_store_response(req: HMMStoreRequest):
    """Store HMM enrichment response in Weaviate + Neo4j + Redis.

    Accepts raw AI response JSON from enrichment runs and calls
    hmm_store_results.process_response() for the triple-write.
    """
    log = logging.getLogger("hmm-store")

    # Validate content is parseable JSON
    try:
        parsed = json.loads(req.content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Content is not valid JSON: {e}")

    # Extract pkg_id from response if not provided
    pkg_id = req.pkg_id or parsed.get("package_id", f"api_{int(time.time())}")

    # Sanitize platform to prevent path traversal
    ALLOWED_PLATFORMS = {"claude", "claude_chat", "claude_code", "grok", "gemini", "chatgpt", "perplexity", "corpus"}
    if req.platform not in ALLOWED_PLATFORMS:
        raise HTTPException(status_code=400, detail=f"Invalid platform: {req.platform}")

    # Write to temp file (process_response expects a file path)
    response_dir = "/var/spark/isma/hmm_responses"
    os.makedirs(response_dir, exist_ok=True)

    # Also save a permanent copy for audit trail
    safe_pkg = pkg_id.replace("/", "_")[:60]
    permanent_path = os.path.join(response_dir, f"{safe_pkg}_{req.platform}.json")

    try:
        with open(permanent_path, "w") as f:
            json.dump(parsed, f)
        log.info(f"Saved response to {permanent_path}")
    except Exception as e:
        log.warning(f"Failed to save permanent copy: {e}")
        # Fall back to temp file
        fd, permanent_path = tempfile.mkstemp(suffix=".json", dir=response_dir)
        with os.fdopen(fd, "w") as f:
            json.dump(parsed, f)

    # Call the triple-write — 6SIGMA: HTTP 500 on failure, never hide errors
    try:
        process_fn = _get_process_response()
        result = process_fn(
            permanent_path,
            platform=req.platform,
            pkg_id=pkg_id,
        )
        log.info(f"HMM store result: {result}")

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Storage failed: {result.get('stored',0)}/{result.get('parsed',0)} stored, "
                       f"{result.get('failed',0)} failed. File: {permanent_path}",
            )

        return {
            "success": True,
            "parsed": result.get("parsed", 0),
            "stored": result.get("stored", 0),
            "pkg_id": pkg_id,
            "file": permanent_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"HMM store failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"HMM store error: {str(e)}. File: {permanent_path}",
        )
