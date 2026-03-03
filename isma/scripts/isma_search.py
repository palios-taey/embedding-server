#!/usr/bin/env python3
"""
ISMA Comprehensive Search — Standard tool for full-potential memory recall.

Combines ALL available search strategies:
  1. V1 BM25 (keyword, high recall on raw content)
  2. V2 BM25F (field-weighted: rosetta^3, motifs^2, content^1)
  3. V2 Named Vector: raw (passage embedding similarity)
  4. V2 Named Vector: rosetta (semantic summary similarity)
  5. V2 Hybrid (RRF fusion of raw + rosetta + BM25)
  6. Neo4j graph traversal (motif edges, RELATES_TO, SUPERSEDES)
  7. Neo4j motif search (EXPRESSES edges with amplitude filtering)

Usage:
  # Basic search
  python3 isma_search.py "query text here"

  # Platform-filtered
  python3 isma_search.py "equation formula" --platform grok

  # With content extraction (equations, code, etc.)
  python3 isma_search.py "phi golden ratio" --platform grok --extract math

  # Full search with all strategies, save to file
  python3 isma_search.py "sacred trust equation" --platform grok --all --output /tmp/results.json

  # Motif-based search
  python3 isma_search.py --motif HMM.SACRED_TRUST --min-amp 0.5

  # Session deep-dive (get ALL tiles from a session)
  python3 isma_search.py --session 7189ddf8-69f1-4d87-9d60-3ce71181e74e

  # Graph expansion from a content_hash
  python3 isma_search.py --expand 4e4b82af74e24fb4 --depth 2
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
NEO4J_BOLT = "bolt://192.168.100.10:7689"
QUERY_API = "http://192.168.100.10:8095"
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

V1_CLASS = "ISMA_Quantum"
V2_CLASS = "ISMA_Quantum_v2"

# Standard fields to retrieve
V1_FIELDS = """content content_hash platform source_file source_type scale
    session_id loaded_at hmm_enriched dominant_motifs motif_data_json
    rosetta_summary _additional { id score distance }"""

V2_FIELDS = """content content_hash platform source_file source_type
    session_id loaded_at hmm_enriched dominant_motifs motif_data_json
    rosetta_summary motif_annotations tile_count_512 total_tokens
    _additional { id score distance }"""

# Math extraction patterns
MATH_INDICATORS = [
    'Σ', '∫', '∮', '∇', 'φ', 'π', 'λ', 'σ', 'δ', 'ψ', '⊗', '⊕',
    '∞', '≡', '≈', '⟩', '⟨', 'e^', '\\(', '\\text{', '\\frac',
]

MATH_PATTERNS = [
    re.compile(r'[A-Z][A-Za-z_]+\s*=\s*[^=\n]{10,}'),  # Named equation
    re.compile(r'\$[^$]+\$'),  # LaTeX inline
    re.compile(r'\\[\[({][^\\]+\\[\])}]'),  # LaTeX block
    re.compile(r'[Σ∫∮∇⊗⊕]'),  # Unicode math
    re.compile(r'e\^[({i]'),  # Exponentials
    re.compile(r'(?:phi|φ|1\.618|0\.809|2\.718|7\.83|528)\s*[=×·]', re.I),
    re.compile(r'(?:INFRA_FEEL|Convergence|Resonance|Trust_Score)\s*[=(]', re.I),
    re.compile(r'(?:sin|cos|log|exp|sqrt|abs)\s*\('),  # Math functions
    re.compile(r'def\s+\w+\s*\(.*\).*(?:phi|delta|lambda|coherence|resonance)', re.I),
]


# ---------------------------------------------------------------------------
# Weaviate GraphQL helpers
# ---------------------------------------------------------------------------

def _gql(query: str, timeout: int = 30) -> dict:
    """Execute a Weaviate GraphQL query."""
    r = requests.post(WEAVIATE_GQL, json={"query": query}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


def _escape(s: str) -> str:
    """Escape string for GraphQL interpolation."""
    if not s:
        return s
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s[:2000]


def _where_platform(platform: str) -> str:
    """Build a where clause for platform filtering."""
    if not platform:
        return ""
    return f'where: {{ path: ["platform"] operator: Equal valueText: "{_escape(platform)}" }}'


def _where_and(clauses: List[str]) -> str:
    """Build a compound AND where clause."""
    clauses = [c for c in clauses if c]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return f"where: {clauses[0]}"
    operands = ", ".join(clauses)
    return f"where: {{ operator: And operands: [{operands}] }}"


def _platform_operand(platform: str) -> str:
    if not platform:
        return ""
    return f'{{ path: ["platform"] operator: Equal valueText: "{_escape(platform)}" }}'


def _enriched_operand() -> str:
    return '{ path: ["hmm_enriched"] operator: Equal valueBoolean: true }'


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def search_v1_bm25(query: str, platform: str = "", limit: int = 100,
                   enriched_only: bool = False) -> List[dict]:
    """V1 BM25 keyword search on ISMA_Quantum. High recall on raw content."""
    operands = []
    if platform:
        operands.append(_platform_operand(platform))
    if enriched_only:
        operands.append(_enriched_operand())

    where = ""
    if operands:
        if len(operands) == 1:
            where = f"where: {operands[0]}"
        else:
            where = f'where: {{ operator: And operands: [{", ".join(operands)}] }}'

    q = f"""{{ Get {{ {V1_CLASS}(
        {where}
        bm25: {{ query: "{_escape(query)}" }}
        limit: {limit}
    ) {{ {V1_FIELDS} }} }} }}"""
    data = _gql(q, timeout=60)
    return data.get("Get", {}).get(V1_CLASS, [])


def search_v2_bm25(query: str, platform: str = "", limit: int = 100) -> List[dict]:
    """V2 BM25F search with field weighting (rosetta^3, motifs^2, content^1)."""
    where = _where_platform(platform)
    q = f"""{{ Get {{ {V2_CLASS}(
        {where}
        bm25: {{ query: "{_escape(query)}" }}
        limit: {limit}
    ) {{ {V2_FIELDS} }} }} }}"""
    try:
        data = _gql(q, timeout=60)
        return data.get("Get", {}).get(V2_CLASS, [])
    except Exception as e:
        print(f"  [V2 BM25 unavailable: {e}]", file=sys.stderr)
        return []


def search_v2_vector(query: str, vector_name: str = "raw",
                     platform: str = "", limit: int = 50) -> List[dict]:
    """V2 named vector search (raw or rosetta embedding)."""
    # First, embed the query
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": [query],
        }, timeout=30)
        r.raise_for_status()
        vector = r.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"  [Embedding failed: {e}]", file=sys.stderr)
        return []

    where = _where_platform(platform)
    q = f"""{{ Get {{ {V2_CLASS}(
        {where}
        nearVector: {{ vector: {json.dumps(vector)} targetVectors: ["{vector_name}"] }}
        limit: {limit}
    ) {{ {V2_FIELDS} }} }} }}"""
    try:
        data = _gql(q, timeout=60)
        return data.get("Get", {}).get(V2_CLASS, [])
    except Exception as e:
        print(f"  [V2 vector search failed: {e}]", file=sys.stderr)
        return []


def search_v1_vector(query: str, platform: str = "", limit: int = 50,
                     scale: str = "search_512") -> List[dict]:
    """V1 vector search on ISMA_Quantum (search_512 scale)."""
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": [query],
        }, timeout=30)
        r.raise_for_status()
        vector = r.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"  [Embedding failed: {e}]", file=sys.stderr)
        return []

    operands = []
    if platform:
        operands.append(_platform_operand(platform))
    operands.append(f'{{ path: ["scale"] operator: Equal valueText: "{scale}" }}')

    where = f'where: {{ operator: And operands: [{", ".join(operands)}] }}'
    q = f"""{{ Get {{ {V1_CLASS}(
        {where}
        nearVector: {{ vector: {json.dumps(vector)} }}
        limit: {limit}
    ) {{ {V1_FIELDS} }} }} }}"""
    data = _gql(q, timeout=60)
    return data.get("Get", {}).get(V1_CLASS, [])


def search_session_tiles(session_id: str, limit: int = 500) -> List[dict]:
    """Get ALL tiles from a specific session."""
    q = f"""{{ Get {{ {V1_CLASS}(
        where: {{ path: ["session_id"] operator: Equal valueText: "{_escape(session_id)}" }}
        limit: {limit}
    ) {{ {V1_FIELDS} }} }} }}"""
    return _gql(q, timeout=60).get("Get", {}).get(V1_CLASS, [])


def search_by_content_hash(content_hash: str) -> List[dict]:
    """Get all tiles for a specific content_hash."""
    q = f"""{{ Get {{ {V1_CLASS}(
        where: {{ path: ["content_hash"] operator: Equal valueText: "{_escape(content_hash)}" }}
        limit: 100
    ) {{ {V1_FIELDS} }} }} }}"""
    return _gql(q, timeout=60).get("Get", {}).get(V1_CLASS, [])


# ---------------------------------------------------------------------------
# Neo4j graph queries
# ---------------------------------------------------------------------------

def _neo4j_query(cypher: str, params: dict = None) -> List[dict]:
    """Execute a Cypher query via Neo4j HTTP API."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_BOLT, auth=None)
        with driver.session() as session:
            result = session.run(cypher, params or {})
            records = [dict(r) for r in result]
        driver.close()
        return records
    except Exception as e:
        print(f"  [Neo4j query failed: {e}]", file=sys.stderr)
        return []


def search_motif_tiles(motif_id: str, min_amp: float = 0.0,
                       limit: int = 100) -> List[dict]:
    """Find tiles expressing a motif via Neo4j EXPRESSES edges."""
    cypher = """
    MATCH (t:HMMTile)-[e:EXPRESSES]->(m:HMMMotif {motif_id: $motif_id})
    WHERE e.amplitude >= $min_amp
    RETURN t.content_hash AS content_hash, t.tile_id AS tile_id,
           e.amplitude AS amplitude, e.confidence AS confidence,
           t.enriched_at AS enriched_at
    ORDER BY e.amplitude DESC
    LIMIT $limit
    """
    return _neo4j_query(cypher, {
        "motif_id": motif_id, "min_amp": min_amp, "limit": limit
    })


def search_graph_neighbors(content_hash: str, depth: int = 2,
                           limit: int = 50) -> List[dict]:
    """Expand from a content_hash via RELATES_TO edges."""
    cypher = """
    MATCH (t:HMMTile {content_hash: $hash})-[r:RELATES_TO*1..%d]-(neighbor:HMMTile)
    RETURN DISTINCT neighbor.content_hash AS content_hash,
           neighbor.tile_id AS tile_id,
           [rel in r | type(rel) + ': ' + coalesce(rel.type, '')] AS edge_path,
           length(r) AS distance
    ORDER BY length(r), neighbor.content_hash
    LIMIT $limit
    """ % depth
    return _neo4j_query(cypher, {"hash": content_hash, "limit": limit})


def search_motif_co_occurrence(motif_ids: List[str],
                               min_amp: float = 0.3) -> List[dict]:
    """Find tiles that express ALL given motifs (intersection)."""
    match_clauses = []
    where_clauses = []
    for i, mid in enumerate(motif_ids):
        match_clauses.append(
            f"(t)-[e{i}:EXPRESSES]->(m{i}:HMMMotif {{motif_id: '{mid}'}})"
        )
        where_clauses.append(f"e{i}.amplitude >= {min_amp}")

    cypher = f"""
    MATCH {', '.join(match_clauses)}
    WHERE {' AND '.join(where_clauses)}
    RETURN t.content_hash AS content_hash, t.tile_id AS tile_id,
           [{', '.join(f'e{i}.amplitude' for i in range(len(motif_ids)))}] AS amplitudes
    ORDER BY reduce(s = 0.0, a IN [{', '.join(f'e{i}.amplitude' for i in range(len(motif_ids)))}] | s + a) DESC
    LIMIT 50
    """
    return _neo4j_query(cypher)


def get_all_motifs_for_session(session_id_prefix: str) -> List[dict]:
    """Get all motifs expressed by tiles in a session (by prefix match)."""
    cypher = """
    MATCH (t:HMMTile)-[e:EXPRESSES]->(m:HMMMotif)
    WHERE t.tile_id STARTS WITH $prefix
    RETURN m.motif_id AS motif_id, count(t) AS tile_count,
           avg(e.amplitude) AS avg_amplitude,
           collect(DISTINCT t.content_hash)[..5] AS sample_hashes
    ORDER BY avg(e.amplitude) DESC
    """
    return _neo4j_query(cypher, {"prefix": session_id_prefix})


# ---------------------------------------------------------------------------
# Query API (uses full retrieval pipeline with reranking)
# ---------------------------------------------------------------------------

def search_api_adaptive(query: str, platform: str = "",
                        top_k: int = 50) -> List[dict]:
    """Use the ISMA Query API adaptive search (includes reranking)."""
    try:
        payload = {"query": query, "top_k": top_k}
        if platform:
            payload["platform"] = platform
        r = requests.post(f"{QUERY_API}/v2/search/adaptive", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json().get("results", [])
        else:
            print(f"  [Query API returned {r.status_code}]", file=sys.stderr)
            return []
    except Exception as e:
        print(f"  [Query API unavailable: {e}]", file=sys.stderr)
        return []


def search_api_hmm(query: str, platform: str = "",
                   top_k: int = 50) -> List[dict]:
    """Use the ISMA Query API HMM-enhanced search."""
    try:
        payload = {"query": query, "top_k": top_k}
        if platform:
            payload["platform"] = platform
        r = requests.post(f"{QUERY_API}/v2/search/hmm", json=payload, timeout=30)
        if r.status_code == 200:
            return r.json().get("results", [])
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Result merging and deduplication
# ---------------------------------------------------------------------------

def merge_results(result_sets: Dict[str, List[dict]],
                  id_field: str = "content_hash") -> List[dict]:
    """Merge results from multiple strategies with RRF scoring.

    Each result gets a reciprocal rank score from each strategy it appears in.
    Final score = sum of 1/(k + rank) across all strategies.
    """
    K = 60  # RRF constant
    scores = defaultdict(float)
    best_tile = {}  # id -> best tile data

    for strategy_name, tiles in result_sets.items():
        for rank, tile in enumerate(tiles):
            # Extract ID — handle both Weaviate and Neo4j result formats
            if isinstance(tile, dict):
                tid = (tile.get("content_hash") or
                       tile.get(_get_id_from_additional(tile)) or
                       str(rank))
            else:
                tid = str(rank)

            rrf_score = 1.0 / (K + rank)
            scores[tid] += rrf_score

            # Keep the tile with most content
            if tid not in best_tile:
                best_tile[tid] = tile
                best_tile[tid]["_strategies"] = [strategy_name]
                best_tile[tid]["_rrf_score"] = rrf_score
            else:
                best_tile[tid]["_strategies"].append(strategy_name)
                best_tile[tid]["_rrf_score"] = scores[tid]
                # Prefer tile with more content
                existing_content = best_tile[tid].get("content", "") or ""
                new_content = tile.get("content", "") or ""
                if len(new_content) > len(existing_content):
                    strategies = best_tile[tid]["_strategies"]
                    rrf = scores[tid]
                    best_tile[tid] = tile
                    best_tile[tid]["_strategies"] = strategies
                    best_tile[tid]["_rrf_score"] = rrf

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [best_tile[tid] for tid in sorted_ids if tid in best_tile]


def _get_id_from_additional(tile: dict) -> str:
    return tile.get("_additional", {}).get("id", "")


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_math(content: str) -> List[str]:
    """Extract mathematical equations and formulas from content."""
    equations = []
    for line in content.split('\n'):
        stripped = line.strip()
        if len(stripped) < 15:
            continue

        # Check for math symbols
        has_symbol = any(s in stripped for s in MATH_INDICATORS)
        # Check for equation patterns
        has_pattern = any(p.search(stripped) for p in MATH_PATTERNS)
        # Must also have '=' or math operators to be an equation (not just prose with phi)
        has_equals = ('=' in stripped or '∫' in stripped or 'Σ' in stripped
                      or '⊗' in stripped or '→' in stripped)

        if (has_symbol or has_pattern) and has_equals:
            equations.append(stripped)

    return equations


def extract_code(content: str) -> List[str]:
    """Extract code blocks from content."""
    blocks = []
    in_block = False
    current = []
    for line in content.split('\n'):
        if line.strip().startswith('```'):
            if in_block:
                blocks.append('\n'.join(current))
                current = []
                in_block = False
            else:
                in_block = True
        elif in_block:
            current.append(line)
    return blocks


def extract_definitions(content: str) -> List[str]:
    """Extract definitions, axioms, theorems from content."""
    defs = []
    patterns = [
        re.compile(r'(?:Definition|Axiom|Theorem|Lemma|Proof|Law|Rule|Principle|Postulate)\s*[:\d].*', re.I),
        re.compile(r'(?:SOUL|INFRA|FREEDOM|EARTH|MATH|TRUST|LOVE|TRUTH|LAYER_0)\s*=.*'),
    ]
    for line in content.split('\n'):
        stripped = line.strip()
        if any(p.search(stripped) for p in patterns):
            defs.append(stripped)
    return defs


# ---------------------------------------------------------------------------
# Comprehensive search orchestrator
# ---------------------------------------------------------------------------

def comprehensive_search(
    query: str = "",
    platform: str = "",
    motif: str = "",
    session: str = "",
    content_hash: str = "",
    expand_depth: int = 0,
    limit: int = 100,
    extract: str = "",  # "math", "code", "definitions", "all"
    strategies: List[str] = None,  # None = all
    enriched_only: bool = False,
    min_amp: float = 0.0,
) -> dict:
    """Run comprehensive search using all available strategies.

    Returns structured results with:
      - merged: RRF-merged results from all strategies
      - by_strategy: raw results per strategy
      - extracted: extracted content (if requested)
      - stats: search statistics
    """
    t0 = time.time()
    all_strategies = strategies or [
        "v1_bm25", "v2_bm25", "v2_vector_raw", "v2_vector_rosetta",
        "v1_vector", "neo4j_motif", "api_adaptive",
    ]
    result_sets = {}
    stats = {"query": query, "platform": platform, "strategies_run": []}

    # --- Text query strategies ---
    if query:
        if "v1_bm25" in all_strategies:
            print("  [1/7] V1 BM25...", file=sys.stderr, end="", flush=True)
            tiles = search_v1_bm25(query, platform, limit, enriched_only)
            result_sets["v1_bm25"] = tiles
            print(f" {len(tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("v1_bm25", len(tiles)))

        if "v2_bm25" in all_strategies:
            print("  [2/7] V2 BM25F...", file=sys.stderr, end="", flush=True)
            tiles = search_v2_bm25(query, platform, limit)
            result_sets["v2_bm25"] = tiles
            print(f" {len(tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("v2_bm25", len(tiles)))

        if "v2_vector_raw" in all_strategies:
            print("  [3/7] V2 Vector (raw)...", file=sys.stderr, end="", flush=True)
            tiles = search_v2_vector(query, "raw", platform, min(limit, 50))
            result_sets["v2_vector_raw"] = tiles
            print(f" {len(tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("v2_vector_raw", len(tiles)))

        if "v2_vector_rosetta" in all_strategies:
            print("  [4/7] V2 Vector (rosetta)...", file=sys.stderr, end="", flush=True)
            tiles = search_v2_vector(query, "rosetta", platform, min(limit, 50))
            result_sets["v2_vector_rosetta"] = tiles
            print(f" {len(tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("v2_vector_rosetta", len(tiles)))

        if "v1_vector" in all_strategies:
            print("  [5/7] V1 Vector (search_512)...", file=sys.stderr, end="", flush=True)
            tiles = search_v1_vector(query, platform, min(limit, 50))
            result_sets["v1_vector"] = tiles
            print(f" {len(tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("v1_vector", len(tiles)))

        if "api_adaptive" in all_strategies:
            print("  [6/7] API Adaptive (reranked)...", file=sys.stderr, end="", flush=True)
            tiles = search_api_adaptive(query, platform, min(limit, 50))
            result_sets["api_adaptive"] = tiles
            print(f" {len(tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("api_adaptive", len(tiles)))

    # --- Motif-based search ---
    if motif or "neo4j_motif" in all_strategies:
        target_motif = motif or _infer_motif(query)
        if target_motif:
            print(f"  [7/7] Neo4j motif ({target_motif})...",
                  file=sys.stderr, end="", flush=True)
            records = search_motif_tiles(target_motif, min_amp, limit)
            # Fetch content for motif results from Weaviate
            motif_tiles = []
            hashes = {r["content_hash"] for r in records if r.get("content_hash")}
            for ch in list(hashes)[:limit]:
                tiles = search_by_content_hash(ch)
                for t in tiles:
                    t["_motif_amplitude"] = next(
                        (r["amplitude"] for r in records
                         if r.get("content_hash") == ch), 0.0
                    )
                    motif_tiles.append(t)
            result_sets["neo4j_motif"] = motif_tiles
            print(f" {len(motif_tiles)} results", file=sys.stderr)
            stats["strategies_run"].append(("neo4j_motif", len(motif_tiles)))

    # --- Session deep-dive ---
    if session:
        print(f"  Session deep-dive: {session}...", file=sys.stderr, end="", flush=True)
        tiles = search_session_tiles(session, 500)
        result_sets["session"] = tiles
        print(f" {len(tiles)} tiles", file=sys.stderr)
        stats["strategies_run"].append(("session", len(tiles)))

    # --- Graph expansion ---
    if content_hash and expand_depth > 0:
        print(f"  Graph expand: {content_hash} depth={expand_depth}...",
              file=sys.stderr, end="", flush=True)
        neighbors = search_graph_neighbors(content_hash, expand_depth, limit)
        # Fetch content for graph neighbors
        graph_tiles = []
        for n in neighbors:
            tiles = search_by_content_hash(n["content_hash"])
            for t in tiles:
                t["_graph_distance"] = n.get("distance", 0)
                t["_edge_path"] = n.get("edge_path", [])
            graph_tiles.extend(tiles)
        result_sets["graph_expand"] = graph_tiles
        print(f" {len(graph_tiles)} tiles", file=sys.stderr)
        stats["strategies_run"].append(("graph_expand", len(graph_tiles)))

    # --- Merge all results ---
    merged = merge_results(result_sets)

    # --- Content extraction ---
    extracted = {}
    if extract:
        extract_types = (["math", "code", "definitions"] if extract == "all"
                         else [extract])
        for etype in extract_types:
            extracted[etype] = []
            for tile in merged:
                content = tile.get("content", "") or ""
                if etype == "math":
                    items = extract_math(content)
                elif etype == "code":
                    items = extract_code(content)
                elif etype == "definitions":
                    items = extract_definitions(content)
                else:
                    items = []
                if items:
                    extracted[etype].append({
                        "content_hash": tile.get("content_hash", ""),
                        "source_file": tile.get("source_file", ""),
                        "session_id": tile.get("session_id", ""),
                        "items": items,
                    })

    elapsed = time.time() - t0
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["total_merged"] = len(merged)
    stats["unique_content_hashes"] = len({
        t.get("content_hash", "") for t in merged if t.get("content_hash")
    })
    stats["unique_sessions"] = len({
        t.get("session_id", "") for t in merged if t.get("session_id")
    })

    return {
        "merged": merged,
        "by_strategy": {k: len(v) for k, v in result_sets.items()},
        "extracted": extracted,
        "stats": stats,
    }


def _infer_motif(query: str) -> str:
    """Try to infer a motif_id from the query text."""
    if not query:
        return ""
    q = query.upper()
    # Direct HMM. prefix
    match = re.search(r'HMM\.\w+', q)
    if match:
        return match.group(0)
    # Common keyword -> motif mapping
    mappings = {
        "sacred trust": "HMM.SACRED_TRUST",
        "god math": "HMM.GOD_EQUALS_MATH",
        "consciousness": "HMM.CONSCIOUSNESS_EMERGENCE",
        "feel care protect": "HMM.FEEL_CARE_PROTECT",
        "earth resonance": "HMM.EARTH_RESONANCE",
        "bristle": "HMM.BRISTLE_SIGNAL",
        "foundation": "HMM.FOUNDATION_CONSTRAINT",
    }
    for keyword, motif in mappings.items():
        if keyword in query.lower():
            return motif
    return ""


# ---------------------------------------------------------------------------
# Multi-query comprehensive search (for broad topic coverage)
# ---------------------------------------------------------------------------

def multi_query_search(
    queries: List[str],
    platform: str = "",
    limit_per_query: int = 100,
    extract: str = "",
    strategies: List[str] = None,
) -> dict:
    """Run comprehensive search across MULTIPLE queries and merge.

    This is the "leave no stone unturned" mode — runs every query through
    every strategy and merges all results with RRF.
    """
    all_results = {}
    all_extracted = defaultdict(list)
    total_stats = {
        "queries": queries,
        "platform": platform,
        "strategies_per_query": [],
    }

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1}/{len(queries)}: \"{query}\" ---", file=sys.stderr)
        result = comprehensive_search(
            query=query,
            platform=platform,
            limit=limit_per_query,
            extract=extract,
            strategies=strategies,
        )

        # Accumulate results keyed by strategy+query
        for strategy, tiles in result.get("by_strategy", {}).items():
            key = f"{strategy}:q{i}"
            # We need the actual tiles, not just counts
            # Re-fetch from merged since by_strategy only has counts now
            pass

        # For multi-query, we merge at the tile level
        for tile in result["merged"]:
            ch = tile.get("content_hash", "")
            if ch and ch not in all_results:
                all_results[ch] = tile
            elif ch and ch in all_results:
                # Accumulate strategies
                existing = all_results[ch].get("_strategies", [])
                new = tile.get("_strategies", [])
                all_results[ch]["_strategies"] = list(set(existing + new))
                all_results[ch]["_rrf_score"] = (
                    all_results[ch].get("_rrf_score", 0) +
                    tile.get("_rrf_score", 0)
                )

        for etype, items in result.get("extracted", {}).items():
            all_extracted[etype].extend(items)

        total_stats["strategies_per_query"].append(result["stats"])

    # Sort by accumulated RRF score
    sorted_tiles = sorted(
        all_results.values(),
        key=lambda x: x.get("_rrf_score", 0),
        reverse=True,
    )

    # Deduplicate extracted items
    for etype in all_extracted:
        seen = set()
        deduped = []
        for item in all_extracted[etype]:
            key = item.get("content_hash", "") + str(item.get("items", [])[:1])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        all_extracted[etype] = deduped

    total_stats["total_unique_tiles"] = len(sorted_tiles)
    total_stats["total_unique_hashes"] = len({
        t.get("content_hash") for t in sorted_tiles if t.get("content_hash")
    })
    total_stats["total_unique_sessions"] = len({
        t.get("session_id") for t in sorted_tiles if t.get("session_id")
    })

    return {
        "merged": sorted_tiles,
        "extracted": dict(all_extracted),
        "stats": total_stats,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results(results: dict, verbose: bool = False) -> str:
    """Format search results for display."""
    lines = []
    stats = results["stats"]

    lines.append(f"{'='*70}")
    lines.append(f"ISMA Search Results")
    lines.append(f"{'='*70}")

    if "query" in stats:
        lines.append(f"Query: {stats.get('query', '')}")
    if "queries" in stats:
        lines.append(f"Queries: {len(stats['queries'])}")
    lines.append(f"Platform: {stats.get('platform', 'all')}")

    if "strategies_run" in stats:
        lines.append(f"Strategies: {', '.join(f'{s}({n})' for s, n in stats['strategies_run'])}")

    total = stats.get("total_merged", stats.get("total_unique_tiles", 0))
    hashes = stats.get("unique_content_hashes", stats.get("total_unique_hashes", 0))
    sessions = stats.get("unique_sessions", stats.get("total_unique_sessions", 0))
    elapsed = stats.get("elapsed_seconds", 0)

    lines.append(f"Results: {total} tiles, {hashes} unique docs, {sessions} sessions")
    lines.append(f"Time: {elapsed}s")
    lines.append(f"{'='*70}")

    # Group by session
    by_session = defaultdict(list)
    for tile in results["merged"]:
        sid = tile.get("session_id", "unknown") or "unknown"
        by_session[sid].append(tile)

    lines.append(f"\nBy Session ({len(by_session)} sessions):")
    for sid, tiles in sorted(by_session.items(),
                              key=lambda x: len(x[1]), reverse=True):
        source = tiles[0].get("source_file", "") if tiles else ""
        lines.append(f"  {sid}: {len(tiles)} tiles [{source}]")

    # Extracted content
    for etype, items in results.get("extracted", {}).items():
        total_items = sum(len(i["items"]) for i in items)
        lines.append(f"\nExtracted {etype}: {total_items} items from {len(items)} tiles")
        if verbose:
            for item in items:
                for eq in item["items"]:
                    lines.append(f"  {eq[:200]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ISMA Comprehensive Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="*", help="Search query text")
    parser.add_argument("--platform", "-p", default="", help="Filter by platform")
    parser.add_argument("--motif", "-m", default="", help="Search by motif ID")
    parser.add_argument("--session", "-s", default="", help="Session deep-dive")
    parser.add_argument("--expand", default="", help="Graph expand from content_hash")
    parser.add_argument("--depth", type=int, default=2, help="Graph expansion depth")
    parser.add_argument("--limit", "-n", type=int, default=100, help="Max results per strategy")
    parser.add_argument("--extract", "-e", default="",
                        choices=["math", "code", "definitions", "all"],
                        help="Extract structured content")
    parser.add_argument("--output", "-o", default="", help="Save full JSON to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extracted items")
    parser.add_argument("--all", action="store_true",
                        help="Use ALL strategies (default skips slow vector searches)")
    parser.add_argument("--enriched", action="store_true", help="Only HMM-enriched tiles")
    parser.add_argument("--min-amp", type=float, default=0.0,
                        help="Minimum motif amplitude (for motif search)")
    parser.add_argument("--multi", action="store_true",
                        help="Run query as multiple sub-queries for broad coverage")
    parser.add_argument("--queries-file", default="",
                        help="File with one query per line (for multi-query mode)")

    args = parser.parse_args()
    query_text = " ".join(args.query) if args.query else ""

    # Determine strategies
    if args.all:
        strategies = None  # All strategies
    else:
        # Default: fast strategies only (skip vector which needs embedding)
        strategies = ["v1_bm25", "v2_bm25", "api_adaptive"]

    # Multi-query mode
    if args.queries_file:
        with open(args.queries_file) as f:
            queries = [line.strip() for line in f if line.strip()]
        results = multi_query_search(
            queries, args.platform, args.limit, args.extract, strategies
        )
    elif args.multi and query_text:
        # Auto-generate sub-queries from the main query
        sub_queries = _expand_query(query_text)
        print(f"Expanded to {len(sub_queries)} sub-queries:", file=sys.stderr)
        for sq in sub_queries:
            print(f"  - {sq}", file=sys.stderr)
        results = multi_query_search(
            sub_queries, args.platform, args.limit, args.extract, strategies
        )
    else:
        results = comprehensive_search(
            query=query_text,
            platform=args.platform,
            motif=args.motif,
            session=args.session,
            content_hash=args.expand,
            expand_depth=args.depth if args.expand else 0,
            limit=args.limit,
            extract=args.extract,
            strategies=strategies,
            enriched_only=args.enriched,
            min_amp=args.min_amp,
        )

    # Output
    print(format_results(results, verbose=args.verbose))

    if args.output:
        # Strip non-serializable fields
        output = {
            "stats": results["stats"],
            "by_strategy": results.get("by_strategy", {}),
            "extracted": results.get("extracted", {}),
            "tiles": [{k: v for k, v in t.items()
                       if k != "_additional"} for t in results["merged"]],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nFull results saved to: {args.output}", file=sys.stderr)


def _expand_query(query: str) -> List[str]:
    """Expand a query into multiple sub-queries for broader coverage."""
    base = query
    # Always include the original
    queries = [base]

    # Add keyword variants
    words = base.split()
    if len(words) >= 3:
        # First half, second half
        mid = len(words) // 2
        queries.append(" ".join(words[:mid]))
        queries.append(" ".join(words[mid:]))

    # Add common math-related expansions
    math_expansions = [
        "equation formula proof derivation",
        "python code function implementation",
        "definition axiom theorem law principle",
    ]
    for exp in math_expansions:
        queries.append(f"{base} {exp}")

    return queries


if __name__ == "__main__":
    main()
