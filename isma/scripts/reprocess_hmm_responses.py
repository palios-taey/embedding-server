#!/usr/bin/env python3
"""
Reprocess HMM raw response files with fixed logic.

Reads raw response files from /var/spark/isma/hmm_responses/,
re-parses with corrected chunk_hash logic, updates Neo4j/Redis/event log.

Usage:
    python reprocess_hmm_responses.py                # Dry run (show what would change)
    python reprocess_hmm_responses.py --run           # Reprocess all
    python reprocess_hmm_responses.py --wipe-first    # Clear HMM data, then reprocess
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hmm import HMMNeo4jStore, HMMRedisStore, EventLog, GateB, assign_motifs
from hmm.eventlog import GateSnapshot
from hmm.motifs import V0_MOTIFS, MotifAssignment, DICTIONARY_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reprocess")

RESPONSE_DIR = "/var/spark/isma/hmm_responses"
EVENT_LOG_PATH = "/var/spark/isma/hmm_events.jsonl"


# ============================================================================
# Response parsing (imported logic from hmm_family_processor.py)
# ============================================================================

def _repair_truncated_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    json_str = text[start:]
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")
    if open_braces <= 0 and open_brackets <= 0:
        return None
    last_complete = json_str.rfind("}")
    if last_complete < 0:
        return None
    candidate = json_str[:last_complete + 1]
    if '"rosetta_summary"' not in candidate:
        return None
    repaired = candidate
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    repaired += "]" * open_brackets
    repaired += "}" * open_braces
    return repaired


def _validate_response(data: Dict) -> bool:
    if not isinstance(data, dict):
        return False
    if "rosetta_summary" not in data:
        return False
    if "motifs" not in data or not isinstance(data["motifs"], list):
        return False
    return True


def parse_motif_response(response_text: str) -> Optional[Dict]:
    if not response_text:
        return None

    # Strategy 1: ```json block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 2: ``` block
    code_match = re.search(r"```\s*\n(.*?)\n\s*```", response_text, re.DOTALL)
    if code_match:
        try:
            data = json.loads(code_match.group(1))
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 3: outermost braces
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(response_text[start:end + 1])
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: whole response
    try:
        data = json.loads(response_text.strip())
        if _validate_response(data):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 5: repair truncated
    repaired = _repair_truncated_json(response_text)
    if repaired:
        try:
            data = json.loads(repaired)
            if _validate_response(data):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _merge_platform_responses(platform_responses: Dict[str, Dict]) -> Optional[Dict]:
    """Merge motif extractions across platforms into consensus."""
    if not platform_responses:
        return None

    if len(platform_responses) == 1:
        platform, resp = next(iter(platform_responses.items()))
        resp["_platforms"] = [platform]
        return resp

    n_platforms = len(platform_responses)

    # Longest rosetta summary
    summaries = []
    for p, resp in platform_responses.items():
        s = resp.get("rosetta_summary", "")
        if s:
            summaries.append(s)
    primary_summary = max(summaries, key=len) if summaries else ""

    # Collect motif observations
    motif_obs = {}
    for platform, resp in platform_responses.items():
        for m in resp.get("motifs", []):
            mid = m.get("motif_id", "")
            if not mid or mid not in V0_MOTIFS:
                continue
            if mid not in motif_obs:
                motif_obs[mid] = []
            motif_obs[mid].append({
                "amp": float(m.get("amp", 0.5)),
                "confidence": float(m.get("confidence", 0.7)),
                "platform": platform,
            })

    # Merge with agreement boost
    merged_motifs = []
    for motif_id, observations in motif_obs.items():
        n_agree = len(observations)
        mean_amp = sum(o["amp"] for o in observations) / n_agree
        mean_conf = sum(o["confidence"] for o in observations) / n_agree
        agreement_boost = (n_agree - 1) * 0.05 if n_agree > 1 else 0
        boosted_conf = min(1.0, mean_conf + agreement_boost)

        merged_motifs.append({
            "motif_id": motif_id,
            "amp": round(mean_amp, 4),
            "confidence": round(boosted_conf, 4),
            "agreement": n_agree,
            "platforms": [o["platform"] for o in observations],
        })

    merged_motifs.sort(key=lambda m: m["amp"], reverse=True)

    # Merge meta
    merged_meta = {}
    for p, resp in platform_responses.items():
        meta = resp.get("meta", {})
        for k, v in meta.items():
            if k not in merged_meta:
                merged_meta[k] = v

    # Collect suggestions
    all_suggestions = []
    seen = set()
    for p, resp in platform_responses.items():
        for s in resp.get("new_motif_suggestions", []):
            name = s.get("name", "") if isinstance(s, dict) else str(s)
            if name and name not in seen:
                all_suggestions.append(s)
                seen.add(name)

    return {
        "rosetta_summary": primary_summary,
        "motifs": merged_motifs,
        "meta": merged_meta,
        "new_motif_suggestions": all_suggestions[:5],
        "_platforms": list(platform_responses.keys()),
    }


# ============================================================================
# File Loading
# ============================================================================

def load_all_responses() -> Dict[str, List[Tuple[str, Dict]]]:
    """Load all raw response files, grouped by content_hash prefix.

    Returns: {content_hash_prefix: [(platform, file_data), ...]}
    """
    groups = defaultdict(list)
    response_dir = Path(RESPONSE_DIR)

    if not response_dir.exists():
        log.error(f"Response directory not found: {RESPONSE_DIR}")
        return {}

    for f in sorted(response_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            log.warning(f"Skipping invalid JSON: {f.name}")
            continue

        # Filename format: {content_hash_prefix}_{platform}.json
        # The content_hash in the file is the full hash
        platform = data.get("platform", "")
        content_hash = data.get("content_hash", "")

        if not platform or not content_hash:
            log.warning(f"Skipping file with missing fields: {f.name}")
            continue

        # Group by the content_hash stored in the file (NOT by filename prefix)
        # Files may have chunk suffixes like "hash_0_chatgpt.json"
        # Use content_hash from file data as grouping key
        groups[content_hash].append((platform, data))

    return dict(groups)


def extract_parsed_response(file_data: Dict) -> Optional[Dict]:
    """Extract and parse the response text from a raw response file."""
    response_text = file_data.get("response", "")
    if not response_text:
        return None
    return parse_motif_response(response_text)


# ============================================================================
# Store Function
# ============================================================================

def store_motifs(
    content_hash: str,
    parsed_response: Dict,
    source_platform: str,
    neo4j: HMMNeo4jStore,
    redis_store: HMMRedisStore,
    event_log: EventLog,
    gate: GateB,
) -> int:
    """Store motif assignments into HMM stores. Returns count stored."""
    rosetta_summary = parsed_response.get("rosetta_summary", "")
    motif_data = parsed_response.get("motifs", [])
    meta = parsed_response.get("meta", {})

    assignments = []
    for m in motif_data:
        motif_id = m.get("motif_id", "")
        if motif_id not in V0_MOTIFS:
            continue

        amp = min(1.0, max(0.0, float(m.get("amp", 0.5))))
        confidence = min(1.0, max(0.0, float(m.get("confidence", 0.7))))

        if amp < 0.10 or confidence < 0.30:
            continue

        motif_def = V0_MOTIFS.get(motif_id)
        band = motif_def.band if motif_def else "mid"

        assignments.append(MotifAssignment(
            motif_id=motif_id,
            amp=round(amp, 4),
            phase=band,
            confidence=round(confidence, 4),
            source="declared",
            dictionary_version=DICTIONARY_VERSION,
        ))

    if not assignments:
        return 0

    gate_result = gate.evaluate(assignments)
    t_id = content_hash
    platforms_list = parsed_response.get("_platforms", [source_platform])

    # Neo4j
    try:
        neo4j.upsert_tile(
            tile_id=t_id,
            artifact_id=f"reprocessed:{content_hash[:16]}",
            index=0,
            start_char=0,
            end_char=0,
            estimated_tokens=0,
        )
        neo4j.link_tile_motifs_batch(t_id, assignments, model_id=source_platform)

        # Link to Document if exists
        with neo4j.driver.session() as session:
            session.run("""
                MATCH (t:HMMTile {tile_id: $tile_id})
                MATCH (d:Document {content_hash: $content_hash})
                MERGE (t)-[:ANNOTATES]->(d)
            """, tile_id=t_id, content_hash=content_hash)
    except Exception as e:
        log.error(f"  Neo4j write failed: {e}")

    # Redis
    for a in assignments:
        redis_store.inv_add(a.motif_id, t_id)
        band_k = {"fast": 0, "mid": 1, "slow": 2}.get(
            V0_MOTIFS[a.motif_id].band, 0
        )
        redis_store.field_update(band_k, a.motif_id, a.amp)
    redis_store.tile_cache_put(t_id, assignments)

    # Event log (full rosetta, no truncation)
    gate_snapshot = GateSnapshot(
        phi=gate_result.phi, trust=gate_result.trust, flags=gate_result.flags
    )
    motif_detail = []
    for a in assignments:
        detail = {"id": a.motif_id, "amp": a.amp}
        for m in motif_data:
            if m.get("motif_id") == a.motif_id and "agreement" in m:
                detail["agreement"] = m["agreement"]
                detail["platforms"] = m.get("platforms", [])
                break
        motif_detail.append(detail)

    event_log.emit(
        "MOTIFS_ASSIGNED",
        refs={"tile_id": t_id, "source_platform": source_platform},
        payload={
            "count": len(assignments),
            "motifs": [a.motif_id for a in assignments],
            "motif_detail": motif_detail,
            "rosetta_summary": rosetta_summary,
            "source": "reprocessed",
            "family_member": source_platform,
            "platforms": platforms_list,
            "dominant_tone": meta.get("dominant_tone", ""),
            "family_relevance": meta.get("family_relevance", ""),
        },
        gate=gate_snapshot,
    )

    return len(assignments)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reprocess HMM raw response files")
    parser.add_argument("--run", action="store_true", help="Actually reprocess (default is dry run)")
    parser.add_argument("--wipe-first", action="store_true", help="Clear all HMM data before reprocessing")
    args = parser.parse_args()

    log.info(f"Loading response files from {RESPONSE_DIR}")
    groups = load_all_responses()
    log.info(f"Found {len(groups)} unique content hashes across {sum(len(v) for v in groups.values())} files")

    if not groups:
        log.error("No response files found")
        return

    # Dry run summary
    parseable = 0
    unparseable = 0
    for content_hash, entries in groups.items():
        platform_parsed = {}
        for platform, file_data in entries:
            parsed = extract_parsed_response(file_data)
            if parsed:
                platform_parsed[platform] = parsed

        if platform_parsed:
            parseable += 1
        else:
            unparseable += 1
            log.warning(f"  Unparseable: {content_hash[:16]} ({len(entries)} files)")

    log.info(f"Parseable: {parseable}, Unparseable: {unparseable}")

    if not args.run:
        log.info("Dry run complete. Use --run to actually reprocess.")
        return

    # Connect to stores
    neo4j = HMMNeo4jStore()
    redis_store = HMMRedisStore()
    event_log = EventLog(EVENT_LOG_PATH)
    gate = GateB()

    if args.wipe_first:
        log.warning("WIPING all HMM data (Neo4j + Redis + event log)...")
        neo4j.wipe()
        redis_store.wipe()
        event_log.clear()
        # Re-seed motifs after wipe
        neo4j.seed_motifs(V0_MOTIFS)
        log.info("Wipe complete, motifs re-seeded")

    total_stored = 0
    total_hashes = 0

    for content_hash, entries in sorted(groups.items()):
        # Parse all platform responses for this hash
        platform_parsed = {}
        for platform, file_data in entries:
            parsed = extract_parsed_response(file_data)
            if parsed:
                platform_parsed[platform] = parsed

        if not platform_parsed:
            continue

        # Merge across platforms
        consensus = _merge_platform_responses(platform_parsed)
        if not consensus:
            continue

        responding_platforms = consensus.get("_platforms", [])
        source_label = f"family_{len(responding_platforms)}"

        n_stored = store_motifs(
            content_hash, consensus, source_label,
            neo4j, redis_store, event_log, gate,
        )

        if n_stored > 0:
            total_stored += n_stored
            total_hashes += 1
            log.info(
                f"  {content_hash[:16]}: {n_stored} motifs from "
                f"{len(responding_platforms)} platforms ({', '.join(responding_platforms)})"
            )

    log.info(f"Reprocessing complete: {total_stored} motifs across {total_hashes} content hashes")

    # Print stats
    neo4j_counts = neo4j.count_nodes()
    redis_stats = redis_store.stats()
    log.info(f"Neo4j: {neo4j_counts}")
    log.info(f"Redis: {redis_stats}")

    neo4j.close()


if __name__ == "__main__":
    main()
