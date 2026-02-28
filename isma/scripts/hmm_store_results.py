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
}

VALID_XREF_TYPES = {"extends", "contradicts", "references", "builds_on"}

# ============================================================================
# Connections
# ============================================================================

_wv_session = requests.Session()
_redis_conn = None
_neo4j_driver = None


def get_redis():
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_conn


def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
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


def embed_and_store_rosetta(content_hash: str, rosetta: str, dominant: list,
                            motif_data: str, platform: str, source_file: str = ""):
    """Embed rosetta summary and create/update a 'rosetta' scale tile in Weaviate.

    This makes the compression text vector-searchable immediately.
    """
    if not rosetta or len(rosetta) < 10:
        return False

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Step 1: Embed the rosetta text
    try:
        r = _wv_session.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": rosetta,
        }, timeout=30)
        if r.status_code != 200:
            log.error(f"  Embed API error: HTTP {r.status_code}")
            return False
        vector = r.json()["data"][0]["embedding"]
    except Exception as e:
        log.error(f"  Embed API error: {e}")
        return False

    # Step 2: Check if rosetta tile already exists
    q = f"""{{ Get {{ {WEAVIATE_CLASS}(
        where: {{
            operator: And,
            operands: [
                {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }},
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
            # Update existing rosetta tile
            oid = existing_tiles[0]["_additional"]["id"]
            r = _wv_session.put(
                f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{oid}",
                json={"class": WEAVIATE_CLASS, "properties": props, "vector": vector},
                timeout=30,
            )
        else:
            # Create new rosetta tile
            r = _wv_session.post(
                f"{WEAVIATE_URL}/v1/objects",
                json={"class": WEAVIATE_CLASS, "properties": props, "vector": vector},
                timeout=30,
            )

        if r.status_code not in (200, 201):
            log.error(f"  Rosetta tile store: HTTP {r.status_code} {r.text[:200]}")
            return False
        log.info(f"  Rosetta tile: {'updated' if existing_tiles else 'created'} for {content_hash[:12]}")
        return True
    except Exception as e:
        log.error(f"  Rosetta tile store error: {e}")
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
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{
                    operator: And,
                    operands: [
                        {{ path: ["content_hash"], operator: Like, valueText: "{hash_prefix}*" }},
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
    """Store enrichment for one item. Triple-write to Weaviate + Neo4j + Redis."""
    rosetta = item["rosetta_summary"]
    motifs = item.get("motifs", [])
    xrefs = item.get("cross_refs", [])

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

    # --- Weaviate: PATCH all tiles for this content_hash ---
    tile_ids = _get_tile_ids(full_hash)
    if not tile_ids:
        log.warning(f"  No tiles found for {full_hash[:12]} — skipping Weaviate PATCH")
        weaviate_ok = True  # Not a failure if tiles don't exist yet
    else:
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
        log.info(f"  Weaviate: patched {patched}/{len(tile_ids)} tiles")
        weaviate_ok = patched > 0

    # --- Weaviate: Embed rosetta summary and create searchable rosetta tile ---
    source_file = ""
    # Try to get source_file from existing tiles
    if tile_ids:
        src_q = f"""{{ Get {{ {WEAVIATE_CLASS}(
            where: {{ path: ["content_hash"], operator: Equal, valueText: "{full_hash}" }}
            limit: 1
        ) {{ source_file }} }} }}"""
        src_data = weaviate_gql(src_q)
        src_tiles = src_data.get("Get", {}).get(WEAVIATE_CLASS, [])
        if src_tiles:
            source_file = src_tiles[0].get("source_file", "")
    embed_and_store_rosetta(full_hash, rosetta, dominant, motif_data, platform, source_file)

    # --- Neo4j: HMMTile + EXPRESSES edges ---
    try:
        driver = get_neo4j()
        with driver.session() as session:
            # Upsert HMMTile
            session.run("""
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

            # Link to source Document (if exists)
            session.run("""
                MATCH (d:Document {content_hash: $content_hash})
                MATCH (t:HMMTile {tile_id: $tile_id})
                MERGE (t)-[r:DERIVED_FROM]->(d)
                ON CREATE SET r.created_at = $now
            """, content_hash=full_hash, tile_id=full_hash, now=now)

            # Link to source ISMASession (if exists, via exchange hash)
            session.run("""
                MATCH (m:ISMAMessage {content_hash: $content_hash})
                MATCH (t:HMMTile {tile_id: $tile_id})
                MERGE (t)-[r:DERIVED_FROM]->(m)
                ON CREATE SET r.created_at = $now
            """, content_hash=full_hash, tile_id=full_hash, now=now)

            # EXPRESSES edges to HMMMotif
            for m in valid_motifs:
                session.run("""
                    MERGE (t:HMMTile {tile_id: $tile_id})
                    MERGE (m:HMMMotif {motif_id: $motif_id})
                    MERGE (t)-[r:EXPRESSES]->(m)
                    ON CREATE SET r.created_at = $now
                    SET r.amp = $amp, r.confidence = $confidence,
                        r.source = 'context_pkg', r.platform = $platform
                """, tile_id=full_hash, motif_id=m["motif_id"],
                    amp=m["amp"], confidence=m["confidence"],
                    now=now, platform=platform)

        log.info(f"  Neo4j: HMMTile + {len(valid_motifs)} EXPRESSES edges")
        neo4j_ok = True
    except Exception as e:
        log.error(f"  NEO4J FAILED for {full_hash[:16]}: {e}")

    # --- Redis: inverted index ---
    try:
        r = get_redis()
        pipe = r.pipeline()
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
        log.error(f"  STORE FAILED for {full_hash[:16]}: weaviate={weaviate_ok} neo4j={neo4j_ok} redis={redis_ok}")
        return False
    return True


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
    ids = []
    for scale in ["search_512", "context_2048", "full_4096"]:
        q = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{
                        operator: And,
                        operands: [
                            {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }},
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
            if "missing hash" in issues:
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

    # Store each item
    stored = 0
    for item in valid_items:
        prefix = item.get("hash", "")
        full_hash = hash_map[prefix]
        log.info(f"Storing {prefix} → {full_hash[:16]}...")

        if store_item(item, full_hash, platform, pkg_id):
            stored += 1

    # Store cross-references
    store_cross_refs(items, hash_map, platform)

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
