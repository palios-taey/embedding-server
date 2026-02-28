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

from isma.src.retrieval import ISMARetrieval, TileResult, SearchResult

app = FastAPI(
    title="ISMA Query API",
    description="Semantic search over 993K embedded tiles with HMM enrichment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton retrieval instance
_retrieval = None

def get_retrieval() -> ISMARetrieval:
    global _retrieval
    if _retrieval is None:
        _retrieval = ISMARetrieval()
    return _retrieval


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
