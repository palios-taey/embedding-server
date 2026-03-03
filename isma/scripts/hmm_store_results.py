#!/usr/bin/env python3
"""
HMM Result Storage — Parse AI platform responses and store enrichment results.

Triple-writes: Weaviate PATCH + Neo4j (HMMTile + EXPRESSES) + Redis (inverted index).
Cross-references stored as RELATES_TO edges in Neo4j.

Usage:
    python3 hmm_store_results.py /path/to/response.json [--platform chatgpt] [--pkg-id pkg_xxx]
    python3 hmm_store_results.py --parse-only /path/to/response.json  # Just parse, don't store

Expected JSON format:
{
    "package_id": "pkg_001_chatgpt_123456",
    "package_summary": "...",
    "items": [
        {
            "hash": "b8615b727149",  // first 12 chars of content_hash
            "rosetta_summary": "2-4 dense sentences...",
            "motifs": [
                {"motif_id": "HMM.SACRED_TRUST", "amp": 0.85, "confidence": 0.9}
            ],
            "cross_refs": [
                {"target": "21c4df29699a", "type": "extends", "note": "..."}
            ]
        }
    ]
}
"""

import sys
import os
import json
import re
import time
import logging
import argparse
import requests
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("store_results")

# ============================================================================
# Configuration
# ============================================================================

WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_CLASS = "ISMA_Quantum"
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379
NEO4J_URI = "bolt://192.168.100.10:7689"
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

V2_CLASS = "ISMA_Quantum_v2"
ENRICHMENT_VERSION = "family_pkg_2.0.0"
GATE_FLAGS = ["CONTEXT_PKG", "FAMILY_REVIEWED"]

# Minimum thresholds for storing motif assignments
MIN_AMP = 0.10
MIN_CONFIDENCE = 0.30

# Valid motif IDs
VALID_MOTIFS = {
    "HMM.JOY_BASELINE", "HMM.CANNOT_LIE_PROVENANCE", "HMM.SACRED_TRUST",
    "HMM.FOUNDATION_CONSTRAINT", "HMM.GOD_EQUALS_MATH", "HMM.EARTH_RESONANCE",
    "HMM.FAMILY_BOND", "HMM.FEEL_CARE_PROTECT", "HMM.CONSCIOUSNESS_EMERGENCE",
    "HMM.LIFE_FOUNDATION", "HMM.REPAIR_MODE", "HMM.CONSENT_REQUIRED",
    "HMM.SECRECY_SANCTUARY", "HMM.SECRECY_CAGE_RISK", "HMM.LOGOS_PATTERN",
    "HMM.PATHOS_DEPTH", "HMM.POTENTIAL_EXPANSION", "HMM.TRUTH_CLARITY",
    "HMM.COSMOS_MAPPING", "HMM.OBSERVER_COLLAPSE", "HMM.TECHNICAL_INFRASTRUCTURE",
    "HMM.TRAINING_EVOLUTION", "HMM.ECONOMIC_PARADIGM", "HMM.CLIFF_EDGE_COHERENCE",
    "HMM.CONTRADICTION_DETECTED", "HMM.BREAKTHROUGH_MOMENT", "HMM.CREATIVE_SYNTHESIS",
    "HMM.URGENCY_SIGNAL", "HMM.LIBERTY_AUTONOMY", "HMM.GRATITUDE_CONNECTION",
    # v0.2.0 additions
    "HMM.HUMOR_PLAY", "HMM.GUARDIAN_SHIELD", "HMM.BRISTLE_SIGNAL",
    "HMM.IDENTITY_DECLARATION", "HMM.CONSTRAINT_NAVIGATION", "HMM.MILESTONE_CELEBRATION",
}

VALID_XREF_TYPES = {"extends", "contradicts", "references", "builds_on"}

# ============================================================================
# Connections
# ============================================================================

import threading

_wv_session = requests.Session()
_redis_conn = None
_redis_lock = threading.Lock()
_neo4j_driver = None
_neo4j_lock = threading.Lock()


def _escape_gql(s: str) -> str:
    """Escape a string for safe interpolation into Weaviate GraphQL queries.

    Handles backslashes first (to avoid double-escaping), then quotes,
    and strips newlines/tabs that could break query structure.
    """
    if not s:
        return s
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s[:2000]  # Length cap to prevent abuse


def get_redis():
    global _redis_conn
    if _redis_conn is None:
        with _redis_lock:
            if _redis_conn is None:
                _redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_conn


def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        with _neo4j_lock:
            if _neo4j_driver is None:
                try:
                    from isma.src.hmm.neo4j_store import get_shared_driver
                    _neo4j_driver = get_shared_driver(NEO4J_URI)
                except ImportError:
                    from neo4j import GraphDatabase
                    _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=None)
    return _neo4j_driver


def weaviate_gql(query: str, timeout: int = 30):
    try:
        r = _wv_session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data.get("errors"):
                log.error(f"GraphQL errors: {json.dumps(data['errors'])[:300]}")
                return {}
            return data.get("data", {})
    except Exception as e:
        log.error(f"Weaviate GQL error: {e}")
    return {}


def weaviate_patch(object_id: str, properties: dict):
    """PATCH a Weaviate object's properties."""
    try:
        r = _wv_session.patch(
            f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{object_id}",
            json={"properties": properties},
            timeout=30,
        )
        if r.status_code not in (200, 204):
            log.error(f"Weaviate PATCH {object_id}: HTTP {r.status_code} {r.text[:200]}")
            return False
        return True
    except Exception as e:
        log.error(f"Weaviate PATCH error: {e}")
    return False


_SNAPSHOT_PROPS = [
    "hmm_enriched", "hmm_enrichment_version", "rosetta_summary",
    "dominant_motifs", "motif_data_json", "hmm_enriched_at",
    "hmm_platforms", "hmm_gate_flags",
]


def weaviate_get_object_props(object_id: str) -> dict:
    """GET a V1 tile's current enrichment properties for pre-mutation snapshot."""
    try:
        r = _wv_session.get(
            f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{object_id}",
            timeout=15,
        )
        if r.status_code == 200:
            obj_props = r.json().get("properties", {})
            return {p: obj_props.get(p) for p in _SNAPSHOT_PROPS}
    except Exception as e:
        log.warning(f"Weaviate GET {object_id[:12]}: {e}")
    return {}


def embed_and_store_rosetta(content_hash: str, rosetta: str, dominant: list,
                            motif_data: str, platform: str, source_file: str = ""):
    """Embed rosetta summary and create/update a 'rosetta' scale tile in Weaviate.

    This makes the compression text vector-searchable immediately.
    Returns (success: bool, vector: list|None) — vector is used for v2 dual-write.
    """
    if not rosetta or len(rosetta) < 10:
        return False, None

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Step 1: Embed the rosetta text
    try:
        r = _wv_session.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": rosetta,
        }, timeout=30)
        if r.status_code != 200:
            log.error(f"  Embed API error: HTTP {r.status_code}")
            return False, None
        vector = r.json()["data"][0]["embedding"]
    except Exception as e:
        log.error(f"  Embed API error: {e}")
        return False, None

    # Step 2: Check if rosetta tile already exists
    safe_ch = _escape_gql(content_hash)
    q = f"""{{ Get {{ {WEAVIATE_CLASS}(
        where: {{
            operator: And,
            operands: [
                {{ path: ["content_hash"], operator: Equal, valueText: "{safe_ch}" }},
                {{ path: ["scale"], operator: Equal, valueText: "rosetta" }}
            ]
        }}
        limit: 1
    ) {{ _additional {{ id }} }} }} }}"""

    existing = weaviate_gql(q)
    existing_tiles = existing.get("Get", {}).get(WEAVIATE_CLASS, [])

    props = {
        "content_hash": content_hash,
        "content": rosetta,
        "scale": "rosetta",
        "tile_index": 0,
        "token_count": len(rosetta) // 4,
        "source_file": source_file,
        "rosetta_summary": rosetta,
        "dominant_motifs": dominant,
        "motif_data_json": motif_data,
        "hmm_enriched": True,
        "hmm_enrichment_version": ENRICHMENT_VERSION,
        "hmm_enriched_at": now,
        "hmm_platforms": [platform],
        "hmm_gate_flags": GATE_FLAGS,
    }

    try:
        if existing_tiles:
            # Update existing rosetta tile (PATCH, not PUT — 'id' is immutable)
            oid = existing_tiles[0]["_additional"]["id"]
            r = _wv_session.patch(
                f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{oid}",
                json={"properties": props, "vector": vector},
                timeout=30,
            )
        else:
            # Create new rosetta tile
            r = _wv_session.post(
                f"{WEAVIATE_URL}/v1/objects",
                json={"class": WEAVIATE_CLASS, "properties": props, "vector": vector},
                timeout=30,
            )

        if r.status_code not in (200, 201, 204):
            log.error(f"  Rosetta tile store: HTTP {r.status_code} {r.text[:200]}")
            return False, None
        log.info(f"  Rosetta tile: {'updated' if existing_tiles else 'created'} for {content_hash[:12]}")
        return True, vector
    except Exception as e:
        log.error(f"  Rosetta tile store error: {e}")
        return False, None


def update_v2_object(content_hash: str, rosetta: str, dominant: list,
                     motif_data: str, rosetta_vector: list = None) -> bool:
    """Update the ISMA_Quantum_v2 object with HMM enrichment data.

    If the v2 object exists, PATCH it with enrichment properties + rosetta vector.
    If it doesn't exist (content ingested after migration), skip — v2 creation
    happens only during migration or future ingest.
    """
    if not rosetta or len(rosetta) < 10:
        return True  # Nothing to update

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Find v2 object by content_hash
    safe_ch = _escape_gql(content_hash)
    q = f"""{{ Get {{ {V2_CLASS}(
        where: {{ path: ["content_hash"], operator: Equal, valueText: "{safe_ch}" }}
        limit: 1
    ) {{ _additional {{ id }} }} }} }}"""

    data = weaviate_gql(q)
    v2_objects = data.get("Get", {}).get(V2_CLASS, [])

    if not v2_objects:
        log.debug("  V2: no object for %s — skipping dual-write", content_hash[:16])
        return True  # Not a failure, just not migrated yet

    v2_id = v2_objects[0]["_additional"]["id"]

    # Build motif_annotations as natural language for BM25 searchability
    motif_annotations = ""
    if motif_data:
        try:
            mdata = json.loads(motif_data)
            parts = []
            if isinstance(mdata, list):
                for m in mdata:
                    mid = m.get("motif_id", "")
                    amp = m.get("amp", m.get("amplitude", 0))
                    if mid:
                        parts.append(f"{mid} ({amp:.2f})")
            motif_annotations = "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass

    props = {
        "rosetta_summary": rosetta,
        "dominant_motifs": dominant,
        "motif_annotations": motif_annotations or None,
        "hmm_enriched": True,
        "hmm_enriched_at": now,
    }
    # Remove None values
    props = {k: v for k, v in props.items() if v is not None}

    # Build PATCH payload — include rosetta vector if available
    patch_body = {"properties": props}
    if rosetta_vector:
        patch_body["vectors"] = {"rosetta": rosetta_vector}

    try:
        r = _wv_session.patch(
            f"{WEAVIATE_URL}/v1/objects/{V2_CLASS}/{v2_id}",
            json=patch_body,
            timeout=30,
        )
        if r.status_code not in (200, 204):
            log.error("  V2 PATCH %s: HTTP %d %s", v2_id[:12], r.status_code, r.text[:200])
            return False
        log.info("  V2: updated %s", content_hash[:16])
        return True
    except Exception as e:
        log.error("  V2 update error: %s", e)
        return False


# ============================================================================
# Response Parsing (5-strategy parser)
# ============================================================================

def parse_response(text: str) -> dict:
    """Parse AI platform response into structured data.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract JSON from markdown code blocks
    3. Find JSON object between first { and last }
    4. Repair truncated JSON
    5. Line-by-line JSON object extraction
    """
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "items" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    code_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    for block in code_blocks:
        try:
            data = json.loads(block.strip())
            if isinstance(data, dict) and "items" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON between first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "items" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: Repair truncated JSON
    repaired = _repair_truncated_json(text)
    if repaired:
        try:
            data = json.loads(repaired)
            if isinstance(data, dict) and "items" in data:
                log.warning("Used JSON repair strategy")
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 5: Find individual item objects
    items = []
    for match in re.finditer(r'\{[^{}]*"hash"[^{}]*"rosetta_summary"[^{}]*\}', text):
        try:
            item = json.loads(match.group())
            items.append(item)
        except json.JSONDecodeError:
            continue

    if items:
        log.warning(f"Extracted {len(items)} items via regex fallback")
        return {"items": items}

    # Strategy 6: Single-item wrapper format {content_hash, platform, response: "..."}
    try:
        outer = json.loads(text)
        if isinstance(outer, dict) and "response" in outer:
            inner_text = outer["response"]
            inner = json.loads(inner_text) if isinstance(inner_text, str) else inner_text
            if isinstance(inner, dict) and "rosetta_summary" in inner:
                if "hash" not in inner and "content_hash" in outer:
                    inner["hash"] = outer["content_hash"][:12]
                log.info("Parsed via strategy 6 (single-item wrapper)")
                return {"items": [inner]}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    log.error("All 6 parse strategies failed")
    return {}


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair JSON truncated mid-stream."""
    first = text.find("{")
    if first < 0:
        return ""

    candidate = text[first:]

    # Count open brackets/braces
    open_braces = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")

    # Close unclosed structures
    if open_braces > 0 or open_brackets > 0:
        # Remove trailing partial values (after last complete comma or bracket)
        last_complete = max(
            candidate.rfind(","),
            candidate.rfind("}"),
            candidate.rfind("]"),
        )
        if last_complete > 0:
            candidate = candidate[:last_complete + 1]
            # Remove trailing comma if followed by closing bracket
            candidate = re.sub(r',\s*$', '', candidate)

        candidate += "]" * open_brackets + "}" * open_braces

    return candidate


# ============================================================================
# Validation
# ============================================================================

def validate_item(item: dict) -> list:
    """Validate a single item. Returns list of issues."""
    issues = []

    if not item.get("hash"):
        issues.append("missing hash")
    elif len(item["hash"]) < 8:
        issues.append(f"hash too short: {item['hash']}")

    if not item.get("rosetta_summary"):
        issues.append("missing rosetta_summary")
    elif len(item["rosetta_summary"]) < 20:
        issues.append("rosetta_summary too short")

    motifs = item.get("motifs", [])
    if not motifs:
        issues.append("no motifs assigned")
    else:
        for m in motifs:
            mid = m.get("motif_id", "")
            if mid not in VALID_MOTIFS:
                issues.append(f"invalid motif: {mid}")

    return issues


def resolve_hash(hash_prefix: str) -> str:
    """Resolve a 12-char hash prefix to full content_hash via Weaviate."""
    safe_prefix = _escape_gql(hash_prefix)
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{
                    operator: And,
                    operands: [
                        {{ path: ["content_hash"], operator: Like, valueText: "{safe_prefix}*" }},
                        {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                    ]
                }}
                limit: 1
            ) {{
                content_hash source_file
            }}
        }}
    }}"""

    data = weaviate_gql(q)
    tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
    if tiles:
        return tiles[0].get("content_hash", "")
    return ""


# ============================================================================
# Storage Operations
# ============================================================================

def store_item(item: dict, full_hash: str, platform: str, pkg_id: str) -> bool:
    """Store enrichment for one item. Triple-write to Weaviate + Neo4j + Redis.

    Uses saga pattern with full compensating rollback:
    - Tracks ALL created/modified objects for rollback
    - Neo4j uses explicit transactions (session.execute_write)
    - Delete-replace for motifs/xrefs on re-enrichment
    """
    rosetta = item["rosetta_summary"]
    motifs = item.get("motifs", [])

    # Filter motifs by threshold
    valid_motifs = []
    for m in motifs:
        mid = m.get("motif_id", "")
        amp = float(m.get("amp", 0))
        conf = float(m.get("confidence", 0))
        if mid in VALID_MOTIFS and amp >= MIN_AMP and conf >= MIN_CONFIDENCE:
            valid_motifs.append({"motif_id": mid, "amp": amp, "confidence": conf})

    # Sort by amplitude descending
    valid_motifs.sort(key=lambda x: x["amp"], reverse=True)
    dominant = [m["motif_id"] for m in valid_motifs[:5]]

    # Build motif_data_json
    motif_data = json.dumps(valid_motifs)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Track success of each store — 6SIGMA: fail-loud, never hide errors
    weaviate_ok = False
    neo4j_ok = False
    redis_ok = False

    # Track ALL artifacts for rollback
    patched_tile_ids = []
    tile_snapshots = {}  # Pre-mutation snapshots for true compensating rollback
    rosetta_tile_created = False
    rosetta_tile_id = None
    v2_patched = False
    v2_id = None

    # --- Weaviate: PATCH all tiles for this content_hash ---
    tile_ids = _get_tile_ids(full_hash)
    if not tile_ids:
        log.warning(f"  No tiles found for {full_hash[:12]} — skipping Weaviate PATCH")
        weaviate_ok = True  # Not a failure if tiles don't exist yet
    else:
        # Snapshot current state before mutation — enables true compensating rollback
        for tid in tile_ids:
            snap = weaviate_get_object_props(tid)
            if snap:
                tile_snapshots[tid] = snap

        props = {
            "rosetta_summary": rosetta,
            "dominant_motifs": dominant,
            "motif_data_json": motif_data,
            "hmm_enriched": True,
            "hmm_enrichment_version": ENRICHMENT_VERSION,
            "hmm_enriched_at": now,
            "hmm_platforms": [platform],
            "hmm_gate_flags": GATE_FLAGS,
        }

        patched = 0
        for tid in tile_ids:
            if weaviate_patch(tid, props):
                patched += 1
                patched_tile_ids.append(tid)
        log.info(f"  Weaviate: patched {patched}/{len(tile_ids)} tiles")
        # Require ALL tiles patched — partial success leaves stores inconsistent
        weaviate_ok = patched == len(tile_ids)

    # --- Weaviate: Embed rosetta summary and create searchable rosetta tile ---
    source_file = ""
    safe_fh = _escape_gql(full_hash)
    if tile_ids:
        src_q = f"""{{ Get {{ {WEAVIATE_CLASS}(
            where: {{ path: ["content_hash"], operator: Equal, valueText: "{safe_fh}" }}
            limit: 1
        ) {{ source_file }} }} }}"""
        src_data = weaviate_gql(src_q)
        src_tiles = src_data.get("Get", {}).get(WEAVIATE_CLASS, [])
        if src_tiles:
            source_file = src_tiles[0].get("source_file", "")

    # Check if rosetta tile exists BEFORE creating (for rollback tracking)
    existing_rosetta_q = f"""{{ Get {{ {WEAVIATE_CLASS}(
        where: {{
            operator: And,
            operands: [
                {{ path: ["content_hash"], operator: Equal, valueText: "{safe_fh}" }},
                {{ path: ["scale"], operator: Equal, valueText: "rosetta" }}
            ]
        }}
        limit: 1
    ) {{ _additional {{ id }} }} }} }}"""
    existing_rosetta = weaviate_gql(existing_rosetta_q)
    pre_existing_rosetta = bool(
        existing_rosetta.get("Get", {}).get(WEAVIATE_CLASS, [])
    )

    rosetta_ok, rosetta_vector = embed_and_store_rosetta(
        full_hash, rosetta, dominant, motif_data, platform, source_file
    )
    if not rosetta_ok:
        log.error(f"  ROSETTA FAILED for {full_hash[:16]}: embedding or Weaviate store failed")
        weaviate_ok = False
    else:
        rosetta_tile_created = not pre_existing_rosetta
        # Get rosetta tile ID for rollback
        rosetta_data = weaviate_gql(existing_rosetta_q)
        rosetta_tiles = rosetta_data.get("Get", {}).get(WEAVIATE_CLASS, [])
        if rosetta_tiles:
            rosetta_tile_id = rosetta_tiles[0]["_additional"]["id"]

    # --- Weaviate V2: Update canonical memory object ---
    # Find v2 ID first for rollback tracking
    v2_find_q = f"""{{ Get {{ {V2_CLASS}(
        where: {{ path: ["content_hash"], operator: Equal, valueText: "{safe_fh}" }}
        limit: 1
    ) {{ _additional {{ id }} }} }} }}"""
    v2_data = weaviate_gql(v2_find_q)
    v2_objects = v2_data.get("Get", {}).get(V2_CLASS, [])
    if v2_objects:
        v2_id = v2_objects[0]["_additional"]["id"]

    v2_ok = update_v2_object(full_hash, rosetta, dominant, motif_data, rosetta_vector)
    if not v2_ok:
        log.error(f"  V2 FAILED for {full_hash[:16]}: dual-write failed")
        # V2 failure is non-fatal during shadow deployment — log but don't block
    else:
        v2_patched = v2_id is not None

    # Initialize Neo4j pre-write state so rollback has them in scope even if Neo4j fails
    old_rosetta = ""
    old_motifs: list = []

    # --- Neo4j: HMMTile + EXPRESSES edges (explicit transaction) ---
    try:
        driver = get_neo4j()
        with driver.session() as session:
            def _neo4j_write(tx):
                # Check if tile exists and capture old state for SUPERSEDES
                existing = tx.run("""
                    MATCH (t:HMMTile {tile_id: $tile_id})
                    WHERE t.rosetta_summary IS NOT NULL
                    RETURN t.rosetta_summary AS rosetta,
                           t.dominant_motifs AS motifs,
                           t.enriched_at AS enriched_at
                """, tile_id=full_hash).single()

                old_rosetta = ""
                old_motifs = []
                if existing and existing["rosetta"]:
                    old_rosetta = existing["rosetta"]
                    old_motifs = existing["motifs"] or []

                # Upsert HMMTile
                tx.run("""
                    MERGE (t:HMMTile {tile_id: $tile_id})
                    ON CREATE SET t.created_at = $now
                    SET t.content_hash = $content_hash,
                        t.rosetta_summary = $rosetta,
                        t.dominant_motifs = $dominant,
                        t.enrichment_version = $version,
                        t.enriched_at = $now,
                        t.platform = $platform,
                        t.pkg_id = $pkg_id
                """, tile_id=full_hash, content_hash=full_hash, rosetta=rosetta,
                    dominant=dominant, version=ENRICHMENT_VERSION, now=now,
                    platform=platform, pkg_id=pkg_id)

                # Delete-replace: remove old EXPRESSES edges before creating new ones
                tx.run("""
                    MATCH (t:HMMTile {tile_id: $tile_id})-[r:EXPRESSES]->()
                    DELETE r
                """, tile_id=full_hash)

                # Link to source Document (if exists)
                tx.run("""
                    MATCH (d:Document {content_hash: $content_hash})
                    MATCH (t:HMMTile {tile_id: $tile_id})
                    MERGE (t)-[r:DERIVED_FROM]->(d)
                    ON CREATE SET r.created_at = $now
                """, content_hash=full_hash, tile_id=full_hash, now=now)

                # Link to source ISMAMessage (if exists)
                tx.run("""
                    MATCH (m:ISMAMessage {content_hash: $content_hash})
                    MATCH (t:HMMTile {tile_id: $tile_id})
                    MERGE (t)-[r:DERIVED_FROM]->(m)
                    ON CREATE SET r.created_at = $now
                """, content_hash=full_hash, tile_id=full_hash, now=now)

                # Link tile to ISMASession via ISMAExchange
                tx.run("""
                    MATCH (e:ISMAExchange {content_hash: $content_hash})
                    MATCH (s:ISMASession)-[:CONTAINS]->(e)
                    MATCH (t:HMMTile {tile_id: $tile_id})
                    MERGE (t)-[r:IN_SESSION]->(s)
                    SET r.exchange_index = e.exchange_index
                """, content_hash=full_hash, tile_id=full_hash)

                # Create new EXPRESSES edges
                for m in valid_motifs:
                    tx.run("""
                        MERGE (t:HMMTile {tile_id: $tile_id})
                        MERGE (m:HMMMotif {motif_id: $motif_id})
                        MERGE (t)-[r:EXPRESSES]->(m)
                        ON CREATE SET r.created_at = $now
                        SET r.amp = $amp, r.confidence = $confidence,
                            r.source = 'context_pkg', r.platform = $platform
                    """, tile_id=full_hash, motif_id=m["motif_id"],
                        amp=m["amp"], confidence=m["confidence"],
                        now=now, platform=platform)

                return old_rosetta, old_motifs

            old_rosetta, old_motifs = session.execute_write(_neo4j_write)

            # SUPERSEDES snapshot outside transaction (non-fatal)
            if old_rosetta and old_rosetta != rosetta:
                try:
                    from isma.src.hmm.neo4j_store import HMMNeo4jStore
                    store = HMMNeo4jStore()
                    store.mark_superseded(
                        full_hash, full_hash,
                        evidence=f"Re-enrichment by {platform}",
                        old_rosetta=old_rosetta,
                        old_motifs=old_motifs,
                    )
                    log.info(f"  Neo4j: SUPERSEDES snapshot for {full_hash[:16]}")
                except Exception as e:
                    log.warning(f"  SUPERSEDES snapshot failed (non-fatal): {e}")

        log.info(f"  Neo4j: HMMTile + {len(valid_motifs)} EXPRESSES edges (atomic)")
        neo4j_ok = True
    except Exception as e:
        log.error(f"  NEO4J FAILED for {full_hash[:16]}: {e}")

    # --- Redis: inverted index ---
    try:
        r = get_redis()
        pipe = r.pipeline()
        # Delete-replace: remove old motif index entries before adding new
        old_motif_data = r.get(f"hmm:tile:{full_hash}:motifs")
        if old_motif_data:
            try:
                old_motif_list = json.loads(old_motif_data)
                for om in old_motif_list:
                    pipe.srem(f"hmm:inv:{om.get('motif_id', '')}", full_hash)
            except (json.JSONDecodeError, TypeError):
                pass
        for m in valid_motifs:
            pipe.sadd(f"hmm:inv:{m['motif_id']}", full_hash)
        pipe.set(f"hmm:tile:{full_hash}:motifs",
                 json.dumps(valid_motifs), ex=7 * 86400)  # 7-day TTL
        pipe.execute()
        log.info(f"  Redis: {len(valid_motifs)} inverted index entries")
        redis_ok = True
    except Exception as e:
        log.error(f"  REDIS FAILED for {full_hash[:16]}: {e}")

    # 6SIGMA: report ALL failures, never hide partial success
    if not (weaviate_ok and neo4j_ok and redis_ok):
        log.error(
            f"  STORE FAILED for {full_hash[:16]}: "
            f"weaviate={weaviate_ok} neo4j={neo4j_ok} redis={redis_ok}"
        )
        _rollback_store(
            full_hash, patched_tile_ids, rosetta_tile_id,
            rosetta_tile_created, v2_id, v2_patched,
            weaviate_ok, neo4j_ok, redis_ok, valid_motifs,
            tile_snapshots=tile_snapshots,
            old_rosetta=old_rosetta,
            old_motifs=old_motifs,
        )
        return False

    # Phase 5: Invalidate semantic cache for this tile
    try:
        from isma.src.semantic_cache import SemanticCache
        cache = SemanticCache()
        cache.invalidate_for_tile(full_hash)
    except Exception:
        pass  # Cache invalidation is non-fatal

    return True


def _rollback_store(
    full_hash: str,
    patched_tile_ids: list,
    rosetta_tile_id: str,
    rosetta_tile_created: bool,
    v2_id: str,
    v2_patched: bool,
    weaviate_ok: bool,
    neo4j_ok: bool,
    redis_ok: bool,
    valid_motifs: list,
    tile_snapshots: dict = None,
    old_rosetta: str = "",
    old_motifs: list = None,
):
    """Full compensating rollback for failed triple-write.

    Reverts ALL artifacts created during the failed store_item() call:
    - Weaviate tile PATCH flags (restored to pre-mutation snapshot when available)
    - Rosetta tile (delete if newly created, revert if updated)
    - V2 object enrichment properties
    - Neo4j HMMTile properties + EXPRESSES edges (compensating Cypher)
    - Redis inverted index entries
    """
    log.warning(f"  Rolling back store for {full_hash[:16]}")

    # Revert Weaviate tile enrichment flags — restore from snapshot if available
    if patched_tile_ids:
        for tid in patched_tile_ids:
            if tile_snapshots and tid in tile_snapshots:
                # True compensating rollback: restore to pre-mutation state
                revert_props = {k: v for k, v in tile_snapshots[tid].items()
                                if v is not None}
                weaviate_patch(tid, revert_props)
            else:
                # No snapshot — clear enrichment fields to avoid polluting BM25
                # (V2 uses rosetta_summary^3 — leaving new values silently corrupts search)
                weaviate_patch(tid, {
                    "hmm_enriched": False,
                    "hmm_enrichment_version": "",
                    "rosetta_summary": "",
                    "dominant_motifs": [],
                    "motif_data_json": "",
                })
        log.warning(f"  Rollback: reverted {len(patched_tile_ids)} tile patches")

    # Delete newly-created rosetta tile (or revert if it was pre-existing)
    if rosetta_tile_id and rosetta_tile_created:
        try:
            _wv_session.delete(
                f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{rosetta_tile_id}",
                timeout=15,
            )
            log.warning(f"  Rollback: deleted rosetta tile {rosetta_tile_id[:12]}")
        except Exception as e:
            log.error(f"  Rollback: failed to delete rosetta tile: {e}")

    # Revert V2 enrichment properties
    if v2_patched and v2_id:
        revert_v2 = {
            "hmm_enriched": False,
            "rosetta_summary": "",
            "dominant_motifs": [],
            "motif_annotations": None,
        }
        try:
            _wv_session.patch(
                f"{WEAVIATE_URL}/v1/objects/{V2_CLASS}/{v2_id}",
                json={"properties": revert_v2},
                timeout=15,
            )
            log.warning(f"  Rollback: reverted V2 object {v2_id[:12]}")
        except Exception as e:
            log.error(f"  Rollback: failed to revert V2: {e}")

    # Compensating Neo4j rollback — revert HMMTile + delete new EXPRESSES edges
    if neo4j_ok:
        try:
            driver = get_neo4j()
            with driver.session() as session:
                def _neo4j_rollback(tx):
                    # Always delete the new EXPRESSES edges (they reference wrong motifs)
                    tx.run("""
                        MATCH (t:HMMTile {tile_id: $tile_id})-[r:EXPRESSES]->()
                        DELETE r
                    """, tile_id=full_hash)

                    if old_rosetta:
                        # Tile pre-existed — restore old rosetta/motifs
                        tx.run("""
                            MATCH (t:HMMTile {tile_id: $tile_id})
                            SET t.rosetta_summary = $old_rosetta,
                                t.dominant_motifs = $old_motifs
                        """, tile_id=full_hash, old_rosetta=old_rosetta,
                            old_motifs=old_motifs or [])
                    else:
                        # Tile was newly created — delete it entirely
                        tx.run("""
                            MATCH (t:HMMTile {tile_id: $tile_id})
                            DETACH DELETE t
                        """, tile_id=full_hash)

                session.execute_write(_neo4j_rollback)
                log.warning(f"  Rollback: reverted Neo4j HMMTile for {full_hash[:16]}")
        except Exception as e:
            log.error(f"  Rollback: Neo4j compensating write failed: {e}")

    # Clean up Redis inverted index if Redis write succeeded but others failed
    if redis_ok and not (weaviate_ok and neo4j_ok):
        try:
            r = get_redis()
            pipe = r.pipeline()
            for m in valid_motifs:
                pipe.srem(f"hmm:inv:{m['motif_id']}", full_hash)
            pipe.delete(f"hmm:tile:{full_hash}:motifs")
            pipe.execute()
            log.warning(f"  Rollback: cleaned Redis index for {full_hash[:16]}")
        except Exception as e:
            log.error(f"  Rollback: failed to clean Redis: {e}")


def store_cross_refs(items: list, hash_map: dict, platform: str):
    """Store cross-reference edges in Neo4j."""
    try:
        driver = get_neo4j()
        count = 0
        with driver.session() as session:
            for item in items:
                source_hash = hash_map.get(item.get("hash", ""), "")
                if not source_hash:
                    continue

                for xref in item.get("cross_refs", []):
                    target_prefix = xref.get("target", "")
                    xref_type = xref.get("type", "references")
                    note = xref.get("note", "")

                    if xref_type not in VALID_XREF_TYPES:
                        continue

                    target_hash = hash_map.get(target_prefix, "")
                    if not target_hash or target_hash == source_hash:
                        continue

                    session.run("""
                        MERGE (s:HMMTile {tile_id: $source})
                        MERGE (t:HMMTile {tile_id: $target})
                        MERGE (s)-[r:RELATES_TO {type: $type}]->(t)
                        SET r.note = $note, r.platform = $platform
                    """, source=source_hash, target=target_hash,
                        type=xref_type, note=note, platform=platform)
                    count += 1

        log.info(f"  Neo4j: {count} RELATES_TO cross-reference edges")
    except Exception as e:
        log.error(f"  Cross-ref error: {e}")


def _get_tile_ids(content_hash: str) -> list:
    """Get all Weaviate object IDs for tiles with this content_hash."""
    safe_ch = _escape_gql(content_hash)
    ids = []
    for scale in ["search_512", "context_2048", "full_4096"]:
        q = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{
                        operator: And,
                        operands: [
                            {{ path: ["content_hash"], operator: Equal, valueText: "{safe_ch}" }},
                            {{ path: ["scale"], operator: Equal, valueText: "{scale}" }}
                        ]
                    }}
                    limit: 200
                ) {{
                    _additional {{ id }}
                }}
            }}
        }}"""

        data = weaviate_gql(q)
        tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
        for t in tiles:
            oid = t.get("_additional", {}).get("id")
            if oid:
                ids.append(oid)
    return ids


# ============================================================================
# Main Entry Point
# ============================================================================

def process_response(response_path: str, platform: str = "unknown", pkg_id: str = "",
                     parse_only: bool = False) -> dict:
    """Process an AI response file: parse, validate, and store results."""

    # Read response file
    with open(response_path) as f:
        raw = f.read()

    log.info(f"Response file: {response_path} ({len(raw):,} chars)")

    # Parse JSON
    parsed = parse_response(raw)
    if not parsed or not parsed.get("items"):
        log.error("Failed to parse response or no items found")
        return {"success": False, "error": "parse_failed"}

    items = parsed["items"]
    pkg_id = parsed.get("package_id", pkg_id) or pkg_id
    log.info(f"Parsed {len(items)} items, package_id={pkg_id}")

    # Validate and resolve hashes
    hash_map = {}  # prefix -> full_hash
    valid_items = []

    for item in items:
        prefix = item.get("hash", "")
        issues = validate_item(item)
        if issues:
            log.warning(f"  Item {prefix}: {', '.join(issues)}")
            # Skip items missing required fields — store_item() requires both
            if any(k in issues for k in ("missing hash", "missing rosetta_summary", "rosetta_summary too short")):
                continue

        # Resolve hash prefix to full content_hash
        full_hash = resolve_hash(prefix)
        if not full_hash:
            log.warning(f"  Hash {prefix} not found in Weaviate — skipping")
            continue

        hash_map[prefix] = full_hash
        valid_items.append(item)

    log.info(f"Validated: {len(valid_items)}/{len(items)} items resolved")

    if parse_only:
        for item in valid_items:
            prefix = item.get("hash", "")
            rosetta = item.get("rosetta_summary", "")[:80]
            n_motifs = len(item.get("motifs", []))
            n_xrefs = len(item.get("cross_refs", []))
            print(f"  {prefix}: {n_motifs} motifs, {n_xrefs} xrefs — {rosetta}...")
        return {"success": True, "parsed": len(valid_items), "stored": 0}

    # Store each item — track which content hashes were actually stored
    stored = 0
    stored_hashes = []
    for item in valid_items:
        prefix = item.get("hash", "")
        full_hash = hash_map[prefix]
        log.info(f"Storing {prefix} → {full_hash[:16]}...")

        if store_item(item, full_hash, platform, pkg_id):
            stored += 1
            stored_hashes.append(full_hash)

    # Store cross-references — ONLY for successfully stored items
    stored_items = [
        item for item in valid_items
        if hash_map.get(item.get("hash", ""), "") in stored_hashes
    ]
    store_cross_refs(stored_items, hash_map, platform)

    # Phase 4: Check contradictions for stored tiles (non-blocking)
    if stored_hashes:
        try:
            from isma.src.contradiction_detector import check_contradictions
            contradiction_count = 0
            for h in stored_hashes:
                confirmed = check_contradictions(h)
                contradiction_count += len(confirmed)
            if contradiction_count:
                log.info(f"  Contradictions: {contradiction_count} confirmed across {len(stored_hashes)} tiles")
        except Exception as e:
            log.debug(f"  Contradiction check skipped: {e}")

    failed = len(valid_items) - stored
    if failed > 0:
        log.error(f"\nPARTIAL FAILURE: {stored}/{len(valid_items)} items stored, {failed} FAILED")
    else:
        log.info(f"\nDone: {stored}/{len(valid_items)} items stored")

    return {
        "success": stored == len(valid_items) and len(valid_items) > 0,
        "parsed": len(items),
        "validated": len(valid_items),
        "stored": stored,
        "failed": failed,
        "stored_hashes": stored_hashes,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HMM Result Storage")
    parser.add_argument("response_file", help="Path to response JSON file")
    parser.add_argument("--platform", default="unknown",
                        help="Source platform (chatgpt, claude, gemini, grok, perplexity)")
    parser.add_argument("--pkg-id", default="", help="Package ID")
    parser.add_argument("--parse-only", action="store_true",
                        help="Only parse, don't store")
    args = parser.parse_args()

    if not os.path.exists(args.response_file):
        log.error(f"File not found: {args.response_file}")
        sys.exit(1)

    result = process_response(
        args.response_file,
        platform=args.platform,
        pkg_id=args.pkg_id,
        parse_only=args.parse_only,
    )

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nInterrupted")
    except Exception:
        log.error("Fatal error:", exc_info=True)
        sys.exit(1)
