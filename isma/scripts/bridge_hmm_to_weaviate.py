#!/usr/bin/env python3
"""
Bridge HMM enrichment data to Weaviate ISMA_Quantum tiles.

Reads motif assignments, rosetta summaries, and gate scores from Neo4j/event log,
then patches matching Weaviate tiles across all scales (512, 2048, 4096).

Usage:
    bridge_hmm_to_weaviate.py --diagnose        # Verify join key, show stats
    bridge_hmm_to_weaviate.py --extend-schema   # Add HMM properties to schema
    bridge_hmm_to_weaviate.py --canary 5        # Enrich 5 tiles, validate
    bridge_hmm_to_weaviate.py --run             # Full bridge (idempotent, skips COMPLETED)
    bridge_hmm_to_weaviate.py --retry-failed    # Reset FAILED bridges for retry
    bridge_hmm_to_weaviate.py --validate        # Post-run verification
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hmm import HMMNeo4jStore, EventLog
from hmm.eventlog import Event
from hmm.motifs import V0_MOTIFS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")

WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_CLASS = "ISMA_Quantum"
EVENT_LOG_PATH = "/var/spark/isma/hmm_events.jsonl"
RESPONSE_DIR = "/var/spark/isma/hmm_responses"
ENRICHMENT_VERSION = "1.0.0"


# ============================================================================
# Schema Extension
# ============================================================================

HMM_PROPERTIES = [
    {
        "name": "dominant_motifs",
        "dataType": ["text[]"],
        "indexFilterable": True,
        "indexSearchable": False,
        "tokenization": "field",
    },
    {
        "name": "rosetta_summary",
        "dataType": ["text"],
        "indexFilterable": False,
        "indexSearchable": True,
        "tokenization": "word",
    },
    {
        "name": "motif_data_json",
        "dataType": ["text"],
        "indexFilterable": False,
        "indexSearchable": False,
        "tokenization": "field",
    },
    {
        "name": "hmm_phi",
        "dataType": ["number"],
        "indexFilterable": True,
        "indexRangeFilters": False,
        "indexSearchable": False,
    },
    {
        "name": "hmm_trust",
        "dataType": ["number"],
        "indexFilterable": True,
        "indexRangeFilters": False,
        "indexSearchable": False,
    },
    {
        "name": "hmm_platforms",
        "dataType": ["text[]"],
        "indexFilterable": True,
        "indexSearchable": False,
        "tokenization": "field",
    },
    {
        "name": "hmm_enriched",
        "dataType": ["boolean"],
        "indexFilterable": True,
        "indexSearchable": False,
    },
    {
        "name": "hmm_enrichment_version",
        "dataType": ["text"],
        "indexFilterable": True,
        "indexSearchable": False,
        "tokenization": "field",
    },
    {
        "name": "hmm_enriched_at",
        "dataType": ["text"],
        "indexFilterable": True,
        "indexSearchable": False,
        "tokenization": "field",
    },
    {
        "name": "hmm_consensus",
        "dataType": ["boolean"],
        "indexFilterable": True,
        "indexSearchable": False,
    },
    {
        "name": "hmm_gate_flags",
        "dataType": ["text[]"],
        "indexFilterable": True,
        "indexSearchable": False,
        "tokenization": "field",
    },
]


def extend_schema(session: requests.Session) -> bool:
    """Add HMM properties to ISMA_Quantum schema. Idempotent."""
    # Get current properties
    r = session.get(f"{WEAVIATE_URL}/v1/schema/{WEAVIATE_CLASS}", timeout=10)
    if r.status_code != 200:
        log.error(f"Failed to get schema: {r.status_code} {r.text}")
        return False

    existing = {p["name"] for p in r.json().get("properties", [])}
    log.info(f"Current schema has {len(existing)} properties")

    added = 0
    for prop in HMM_PROPERTIES:
        if prop["name"] in existing:
            log.info(f"  Skip (exists): {prop['name']}")
            continue

        r = session.post(
            f"{WEAVIATE_URL}/v1/schema/{WEAVIATE_CLASS}/properties",
            json=prop,
            timeout=10,
        )
        if r.status_code == 200:
            log.info(f"  Added: {prop['name']} ({prop['dataType'][0]})")
            added += 1
        else:
            log.error(f"  FAILED: {prop['name']} - {r.status_code} {r.text}")
            return False

    log.info(f"Schema extension complete: {added} new properties added")
    return True


# ============================================================================
# Data Collection
# ============================================================================

def get_tile_motifs(neo4j: HMMNeo4jStore, tile_id: str) -> List[Dict]:
    """Get motif assignments for a tile from Neo4j (amp >= 0.7 for dominant)."""
    return neo4j.get_tile_motifs(tile_id)


def get_rosetta_from_responses(content_hash: str) -> str:
    """Get rosetta summary from raw response files (full text, not truncated)."""
    response_dir = Path(RESPONSE_DIR)
    prefix = content_hash[:16]

    best_rosetta = ""
    for f in response_dir.glob(f"{prefix}_*.json"):
        try:
            data = json.loads(f.read_text())
            response_text = data.get("response", "")
            # Parse the JSON response to get rosetta
            parsed = _quick_parse_rosetta(response_text)
            if parsed and len(parsed) > len(best_rosetta):
                best_rosetta = parsed
        except (json.JSONDecodeError, OSError):
            continue

    return best_rosetta


def _quick_parse_rosetta(response_text: str) -> str:
    """Extract rosetta_summary from a response text."""
    import re
    if not response_text:
        return ""

    # Try JSON in code block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    text = json_match.group(1) if json_match else response_text

    # Try parsing
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data.get("rosetta_summary", "")
        except json.JSONDecodeError:
            pass

    return ""


def get_gate_scores_from_events(event_log: EventLog, tile_id: str) -> Dict:
    """Get latest gate scores and flags for a tile from event log."""
    best = {"phi": 0.0, "trust": 0.0, "flags": [], "platforms": [], "is_consensus": False}

    for event in event_log.iter_all():
        if event.type != "MOTIFS_ASSIGNED":
            continue
        if event.refs.get("tile_id") != tile_id:
            continue

        # Skip ghost events (empty motifs)
        motifs = event.payload.get("motifs", [])
        if not motifs:
            continue

        best["phi"] = event.gate.phi
        best["trust"] = event.gate.trust
        best["flags"] = event.gate.flags
        best["platforms"] = event.payload.get("platforms", [])

        # Detect consensus: source_platform starts with "family_"
        source = event.refs.get("source_platform", "")
        if source.startswith("family_"):
            best["is_consensus"] = True

        # Also get rosetta from event log as fallback
        rosetta = event.payload.get("rosetta_summary", "")
        if rosetta:
            best["rosetta_fallback"] = rosetta

    return best


def find_weaviate_uuids(session: requests.Session, content_hash: str) -> List[str]:
    """Find ALL Weaviate tile UUIDs where content_hash matches (across all scales)."""
    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }}
                limit: 10
            ) {{
                scale
                _additional {{ id }}
            }}
        }}
    }}"""

    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
        return [obj["_additional"]["id"] for obj in data]
    except Exception as e:
        log.error(f"Weaviate query failed for {content_hash[:16]}: {e}")
        return []


# ============================================================================
# Enrichment
# ============================================================================

def build_enrichment_payload(
    motifs: List[Dict],
    rosetta: str,
    gate_info: Dict,
) -> Dict:
    """Build the Weaviate PATCH payload for a tile."""
    # dominant_motifs: only those with amp >= 0.7
    dominant = [m["motif_id"] for m in motifs if m.get("amp", 0) >= 0.7]

    # motif_data_json: full detail blob
    motif_blob = json.dumps([{
        "motif_id": m["motif_id"],
        "amp": m.get("amp", 0),
        "confidence": m.get("confidence", 0),
        "band": m.get("band", ""),
        "source": m.get("source", ""),
    } for m in motifs])

    return {
        "dominant_motifs": dominant,
        "rosetta_summary": rosetta,
        "motif_data_json": motif_blob,
        "hmm_phi": gate_info.get("phi", 0.0),
        "hmm_trust": gate_info.get("trust", 0.0),
        "hmm_platforms": gate_info.get("platforms", []),
        "hmm_enriched": True,
        "hmm_enrichment_version": ENRICHMENT_VERSION,
        "hmm_enriched_at": datetime.now(timezone.utc).isoformat(),
        "hmm_consensus": gate_info.get("is_consensus", False),
        "hmm_gate_flags": gate_info.get("flags", []),
    }


def enrich_tile(
    session: requests.Session,
    tile_id: str,
    neo4j: HMMNeo4jStore,
    event_log: EventLog,
) -> Tuple[int, str]:
    """Enrich a single tile. Returns (tiles_enriched, error_or_empty)."""
    # Normalize tile_id (strip hmm_tile_ prefix if present from P0 test data)
    clean_id = tile_id
    if clean_id.startswith("hmm_tile_"):
        clean_id = clean_id[len("hmm_tile_"):]

    # 1. Get motif assignments from Neo4j
    motifs = get_tile_motifs(neo4j, tile_id)
    if not motifs:
        return 0, f"No motifs for {tile_id[:16]}"

    # 2. Get rosetta from raw response files, fallback to event log
    rosetta = get_rosetta_from_responses(clean_id)

    # 3. Get gate scores from event log
    gate_info = get_gate_scores_from_events(event_log, tile_id)

    # Fallback rosetta from event log
    if not rosetta and gate_info.get("rosetta_fallback"):
        rosetta = gate_info["rosetta_fallback"]

    # 4. Find ALL Weaviate UUIDs for this content_hash
    uuids = find_weaviate_uuids(session, clean_id)
    if not uuids:
        return 0, f"No Weaviate tiles for {clean_id[:16]}"

    # 5. Build payload and PATCH each UUID
    payload = build_enrichment_payload(motifs, rosetta, gate_info)
    enriched = 0

    for uuid in uuids:
        r = session.patch(
            f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{uuid}",
            json={"properties": payload},
            timeout=10,
        )
        if r.status_code in (200, 204):
            enriched += 1
        else:
            log.warning(f"  PATCH failed for UUID {uuid}: {r.status_code} {r.text[:200]}")

    return enriched, ""


# ============================================================================
# Diagnose
# ============================================================================

def diagnose(neo4j: HMMNeo4jStore, session: requests.Session):
    """Verify join key and show stats."""
    log.info("=== DIAGNOSE ===")

    # Neo4j HMM stats
    counts = neo4j.count_nodes()
    log.info(f"Neo4j HMM nodes: {counts}")

    # Get all HMMTile tile_ids
    with neo4j.driver.session() as s:
        result = s.run("MATCH (t:HMMTile) RETURN t.tile_id AS tid LIMIT 500")
        tile_ids = [r["tid"] for r in result]
    log.info(f"HMMTile count: {len(tile_ids)}")

    if not tile_ids:
        log.warning("No HMMTiles found. Run reprocess_hmm_responses.py first.")
        return

    # Sample join verification
    sample = tile_ids[:10]
    matched = 0
    for tid in sample:
        clean_id = tid
        if clean_id.startswith("hmm_tile_"):
            clean_id = clean_id[len("hmm_tile_"):]
        uuids = find_weaviate_uuids(session, clean_id)
        if uuids:
            matched += 1
            log.info(f"  {clean_id[:16]}: {len(uuids)} Weaviate tiles")
        else:
            log.warning(f"  {clean_id[:16]}: NO Weaviate match")

    log.info(f"Join key verification: {matched}/{len(sample)} matched")

    # Weaviate stats
    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={
            "query": "{ Aggregate { ISMA_Quantum { meta { count } } } }"
        }, timeout=10)
        total = r.json()["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]
        log.info(f"Weaviate total tiles: {total}")
    except Exception as e:
        log.error(f"Weaviate aggregate failed: {e}")

    # Check if any already enriched
    gql = f"""{{
        Aggregate {{
            {WEAVIATE_CLASS}(
                where: {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
            ) {{ meta {{ count }} }}
        }}
    }}"""
    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=10)
        enriched = r.json()["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]
        log.info(f"Already enriched: {enriched}")
    except Exception:
        log.info("Cannot check enriched count (property may not exist yet)")

    # Bridge status
    bridge_stats = neo4j.get_bridge_stats()
    log.info(f"Bridge status: {bridge_stats}")


# ============================================================================
# Validate
# ============================================================================

def validate(neo4j: HMMNeo4jStore, session: requests.Session):
    """Post-run validation."""
    log.info("=== VALIDATE ===")

    bridge_stats = neo4j.get_bridge_stats()
    log.info(f"Bridge stats: {bridge_stats}")

    completed = bridge_stats.get("COMPLETED", 0)
    failed = bridge_stats.get("FAILED", 0)
    total = completed + failed

    if total == 0:
        log.warning("No bridge records found. Run --run first.")
        return

    log.info(f"Completed: {completed}, Failed: {failed}, Total: {total}")

    # Spot-check: fetch 3 enriched tiles from Weaviate
    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
                limit: 3
            ) {{
                content_hash
                dominant_motifs
                rosetta_summary
                hmm_phi
                hmm_trust
                hmm_platforms
                hmm_enriched
                hmm_consensus
                hmm_enrichment_version
                hmm_enriched_at
                hmm_gate_flags
                scale
                _additional {{ id }}
            }}
        }}
    }}"""

    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=10)
        if r.status_code != 200:
            log.error(f"Validation query failed: {r.status_code}")
            return

        data = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
        log.info(f"Spot-check: {len(data)} enriched tiles")
        for obj in data:
            ch = obj.get("content_hash", "")[:16]
            dm = obj.get("dominant_motifs", [])
            rs = (obj.get("rosetta_summary", "") or "")[:80]
            phi = obj.get("hmm_phi", 0)
            trust = obj.get("hmm_trust", 0)
            scale = obj.get("scale", "")
            consensus = obj.get("hmm_consensus", False)
            log.info(f"  {ch} [{scale}] phi={phi:.2f} trust={trust:.2f} consensus={consensus}")
            log.info(f"    motifs: {dm}")
            log.info(f"    rosetta: {rs}...")
    except Exception as e:
        log.error(f"Validation failed: {e}")

    # Count enriched by scale
    for scale in ["search_512", "context_2048", "full_4096"]:
        gql = f"""{{
            Aggregate {{
                {WEAVIATE_CLASS}(
                    where: {{ operator: And, operands: [
                        {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }},
                        {{ path: ["scale"], operator: Equal, valueText: "{scale}" }}
                    ] }}
                ) {{ meta {{ count }} }}
            }}
        }}"""
        try:
            r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=10)
            count = r.json()["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]
            log.info(f"  Enriched {scale}: {count}")
        except Exception:
            pass


# ============================================================================
# Main Bridge
# ============================================================================

def retry_failed(neo4j: HMMNeo4jStore):
    """Reset FAILED WeaviateBridge nodes (with 'No Weaviate tiles' error) to allow re-processing."""
    query = """
    MATCH (b:WeaviateBridge)
    WHERE b.status = 'FAILED' AND b.error_message CONTAINS 'No Weaviate tiles'
    WITH b, b.content_hash AS ch
    SET b.status = 'PENDING', b.updated_at = datetime()
    RETURN count(b) AS reset_count
    """
    with neo4j.driver.session() as s:
        result = s.run(query)
        count = result.single()["reset_count"]
    log.info(f"Reset {count} FAILED bridges to PENDING for retry")
    return count


def _get_completed_hashes(neo4j: HMMNeo4jStore) -> Set[str]:
    """Get set of content_hashes that already have COMPLETED bridges."""
    query = """
    MATCH (b:WeaviateBridge {status: 'COMPLETED'})
    RETURN b.content_hash AS ch
    """
    with neo4j.driver.session() as s:
        result = s.run(query)
        return {r["ch"] for r in result}


def run_bridge(neo4j: HMMNeo4jStore, event_log: EventLog,
               session: requests.Session, limit: int = 0):
    """Run the full bridge: enrich all HMMTiles in Weaviate.

    Idempotent: skips tiles that already have COMPLETED bridges.
    """
    # Get all HMMTile tile_ids
    with neo4j.driver.session() as s:
        result = s.run("MATCH (t:HMMTile) RETURN t.tile_id AS tid")
        all_tiles = [r["tid"] for r in result]

    log.info(f"Total HMMTiles: {len(all_tiles)}")

    # Skip already-completed tiles (idempotent)
    completed = _get_completed_hashes(neo4j)
    if completed:
        before = len(all_tiles)
        all_tiles = [t for t in all_tiles if t not in completed
                     and (t[len("hmm_tile_"):] if t.startswith("hmm_tile_") else t) not in completed]
        log.info(f"Skipping {before - len(all_tiles)} already-completed tiles")

    if limit > 0:
        all_tiles = all_tiles[:limit]
        log.info(f"Canary mode: processing {limit} tiles")

    if not all_tiles:
        log.info("No tiles to process")
        return 0

    total_enriched = 0
    total_failed = 0
    total_skipped = 0

    for i, tile_id in enumerate(all_tiles):
        enriched, error = enrich_tile(session, tile_id, neo4j, event_log)

        if error:
            log.debug(f"  [{i+1}/{len(all_tiles)}] {tile_id[:16]}: {error}")
            neo4j.upsert_bridge_status(
                tile_id, "FAILED", tiles_enriched=0,
                version=ENRICHMENT_VERSION, error=error,
            )
            if "No Weaviate tiles" in error:
                total_skipped += 1
            else:
                total_failed += 1
        else:
            log.info(f"  [{i+1}/{len(all_tiles)}] {tile_id[:16]}: enriched {enriched} tiles")
            neo4j.upsert_bridge_status(
                tile_id, "COMPLETED", tiles_enriched=enriched,
                version=ENRICHMENT_VERSION,
            )
            total_enriched += enriched

        # Progress every 20
        if (i + 1) % 20 == 0:
            log.info(f"Progress: {i+1}/{len(all_tiles)} processed, {total_enriched} enriched")

    log.info(
        f"Bridge complete: {total_enriched} Weaviate tiles enriched, "
        f"{total_failed} failed, {total_skipped} skipped (no Weaviate match)"
    )

    return total_enriched


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bridge HMM enrichment to Weaviate")
    parser.add_argument("--diagnose", action="store_true", help="Verify join key, show stats")
    parser.add_argument("--extend-schema", action="store_true", help="Add HMM properties to schema")
    parser.add_argument("--canary", type=int, metavar="N", help="Enrich N tiles as canary test")
    parser.add_argument("--run", action="store_true", help="Full bridge run")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reset FAILED bridges (No Weaviate tiles) to PENDING for retry")
    parser.add_argument("--validate", action="store_true", help="Post-run verification")
    args = parser.parse_args()

    if not any([args.diagnose, args.extend_schema, args.canary, args.run,
                args.retry_failed, args.validate]):
        parser.print_help()
        return

    session = requests.Session()
    neo4j = HMMNeo4jStore()
    event_log = EventLog(EVENT_LOG_PATH)

    try:
        # Verify Weaviate is up
        r = session.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
        if r.status_code != 200:
            log.error(f"Weaviate not healthy: {r.status_code}")
            return
        log.info(f"Weaviate healthy: {r.json().get('version', 'unknown')}")

        if args.retry_failed:
            retry_failed(neo4j)

        if args.diagnose:
            diagnose(neo4j, session)

        if args.extend_schema:
            if not extend_schema(session):
                log.error("Schema extension failed, aborting")
                return

        if args.canary:
            n = run_bridge(neo4j, event_log, session, limit=args.canary)
            log.info(f"Canary complete: {n} tiles enriched")
            # Auto-validate after canary
            validate(neo4j, session)

        if args.run:
            n = run_bridge(neo4j, event_log, session)
            log.info(f"Full bridge complete: {n} tiles enriched")

        if args.validate:
            validate(neo4j, session)

    finally:
        neo4j.close()


if __name__ == "__main__":
    main()
