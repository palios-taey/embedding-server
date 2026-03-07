#!/usr/bin/env python3
"""
F13 - Tier 2 Family LLM Batch Review via Taey's Hands.

Sends batches of 50-200 tiles to Family platforms for high-quality motif
assignment with rosetta summaries and consensus.

Key difference from hmm_family_processor.py:
  - Current processor sends 1 document per message
  - Tier 2 sends 50-200 TILES per message (search_512 scale, ~500 chars each)
  - Focus on tiles, not documents

Tile selection priorities:
  P1: Kernel/layer_0 tiles with only Tier 1 classification
  P2: Tiles where Tier 1 is uncertain (no dominant motif above 0.6)
  P3: Tiles with no classification at all
  P4: 1% random sample for quality validation

Usage:
    tier2_family_batch.py --select          # Show candidate count by priority
    tier2_family_batch.py --dry-run         # Show batch plan without sending
    tier2_family_batch.py --run             # Process all candidates
    tier2_family_batch.py --run --limit 200 # Process one batch
    tier2_family_batch.py --resume          # Resume from state
    tier2_family_batch.py --validate        # Compare Tier 1 vs Tier 2 agreement
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/home/spark/taeys-hands")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hmm.motifs import V0_MOTIFS, MotifAssignment, DICTIONARY_VERSION
from hmm.neo4j_store import HMMNeo4jStore
from hmm.redis_store import HMMRedisStore
from hmm.eventlog import EventLog, GateSnapshot
from hmm.gate_b import GateB

from core import input as inp, clipboard, atspi
from core.tree import find_elements, find_copy_buttons
from core.platforms import TAB_SHORTCUTS

from hmm_prompts import MOTIF_REFERENCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("tier2_batch")
logging.getLogger("neo4j").setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
WEAVIATE_CLASS = "ISMA_Quantum"
STATE_FILE = "/var/spark/isma/tier2_batch_state.json"
RESPONSE_LOG_DIR = "/var/spark/isma/hmm_responses"
EVENT_LOG_PATH = "/var/spark/isma/hmm_events.jsonl"
ENRICHMENT_VERSION = "tier2_1.0.0"

# Platform configuration
PLATFORMS = ["chatgpt", "grok", "claude"]  # Gemini/Perplexity excluded (broken in processor)
BATCH_SIZE = 100  # Tiles per message (conservative - fits all platforms)
MAX_BATCH_SIZE = 200

# Timing
POLL_INTERVAL = 5
MAX_WAIT_RESPONSE = 600  # 10 min per batch (large response)
BETWEEN_BATCHES = 10     # seconds between batches on same platform
BETWEEN_PLATFORMS = 15   # seconds when switching platforms

# Stop button patterns
STOP_PATTERNS = {
    'chatgpt': ['stop', 'stop generating'],
    'claude': ['stop', 'stop response'],
    'grok': ['stop', 'stop generating'],
}

MAX_CONSECUTIVE_ERRORS = 3


# =============================================================================
# State Management
# =============================================================================

def load_state() -> dict:
    """Load batch processing state."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "batches_completed": 0,
            "tiles_classified": 0,
            "current_batch_hashes": [],
            "platforms_used": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }


def save_state(state: dict):
    """Save batch processing state."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# =============================================================================
# Tile Selection
# =============================================================================

def select_candidates(session: requests.Session,
                      limit: int = 0) -> Dict[str, List[dict]]:
    """Select tiles for Tier 2 review, grouped by priority.

    Returns dict with keys P1-P4, each containing list of
    {"uuid": ..., "content_hash": ..., "content": ...}.
    """
    candidates = {"P1": [], "P2": [], "P3": [], "P4": []}

    # P1: Kernel/layer_0 tiles with tier1 classification
    p1_gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ operator: And, operands: [
                    {{ path: ["scale"], operator: Equal, valueText: "search_512" }},
                    {{ path: ["hmm_enrichment_version"], operator: Like, valueText: "tier1*" }},
                    {{ path: ["priority"], operator: GreaterThanEqual, valueNumber: 0.8 }}
                ] }}
                limit: 5000
            ) {{
                content content_hash priority
                dominant_motifs motif_data_json
                _additional {{ id }}
            }}
        }}
    }}"""
    candidates["P1"] = _run_candidate_query(session, p1_gql)

    # P2: Tier 1 tiles with uncertain classification (no dominant above 0.6)
    p2_gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ operator: And, operands: [
                    {{ path: ["scale"], operator: Equal, valueText: "search_512" }},
                    {{ path: ["hmm_enrichment_version"], operator: Like, valueText: "tier1*" }},
                    {{ path: ["dominant_motifs"], operator: IsNull, valueBoolean: true }}
                ] }}
                limit: 5000
            ) {{
                content content_hash priority
                motif_data_json
                _additional {{ id }}
            }}
        }}
    }}"""
    candidates["P2"] = _run_candidate_query(session, p2_gql)

    # P3: Completely unenriched tiles
    p3_gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ operator: And, operands: [
                    {{ path: ["scale"], operator: Equal, valueText: "search_512" }},
                    {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: false }}
                ] }}
                limit: 5000
            ) {{
                content content_hash priority
                _additional {{ id }}
            }}
        }}
    }}"""
    p3_raw = _run_candidate_query(session, p3_gql)
    # Also try tiles where hmm_enriched is null (never set)
    if len(p3_raw) < 5000:
        p3_null_gql = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{ operator: And, operands: [
                        {{ path: ["scale"], operator: Equal, valueText: "search_512" }},
                        {{ path: ["hmm_enriched"], operator: IsNull, valueBoolean: true }}
                    ] }}
                    limit: {5000 - len(p3_raw)}
                ) {{
                    content content_hash priority
                    _additional {{ id }}
                }}
            }}
        }}"""
        p3_raw.extend(_run_candidate_query(session, p3_null_gql))
    candidates["P3"] = p3_raw

    # P4: 1% random sample of all search_512 for validation
    # Fetch a sample and randomly select 1%
    p4_gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ operator: And, operands: [
                    {{ path: ["scale"], operator: Equal, valueText: "search_512" }},
                    {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
                ] }}
                limit: 1000
            ) {{
                content content_hash priority
                dominant_motifs motif_data_json
                hmm_enrichment_version
                _additional {{ id }}
            }}
        }}
    }}"""
    p4_all = _run_candidate_query(session, p4_gql)
    sample_size = max(10, len(p4_all) // 100)
    if p4_all:
        random.shuffle(p4_all)
        candidates["P4"] = p4_all[:sample_size]

    # Deduplicate across priorities (P1 wins over P2 over P3)
    seen_hashes = set()
    for priority in ["P1", "P2", "P3", "P4"]:
        deduped = []
        for tile in candidates[priority]:
            ch = tile.get("content_hash", "")
            if ch not in seen_hashes:
                seen_hashes.add(ch)
                deduped.append(tile)
        candidates[priority] = deduped

    if limit > 0:
        total = 0
        for priority in ["P1", "P2", "P3", "P4"]:
            remaining = limit - total
            if remaining <= 0:
                candidates[priority] = []
            else:
                candidates[priority] = candidates[priority][:remaining]
                total += len(candidates[priority])

    return candidates


def _run_candidate_query(session: requests.Session, gql: str) -> List[dict]:
    """Execute a candidate selection query and return results."""
    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                         json={"query": gql}, timeout=30)
        if r.status_code != 200:
            log.warning(f"Candidate query failed: {r.status_code}")
            return []
        data = r.json()
        errors = data.get("errors")
        if errors:
            log.warning(f"GraphQL errors: {errors[0].get('message', '')[:100]}")
            return []
        objects = data.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
        return [
            {
                "uuid": obj.get("_additional", {}).get("id", ""),
                "content_hash": obj.get("content_hash", ""),
                "content": obj.get("content", ""),
                "priority": obj.get("priority", 0),
            }
            for obj in objects
            if obj.get("_additional", {}).get("id")
        ]
    except Exception as e:
        log.error(f"Candidate query error: {e}")
        return []


# =============================================================================
# Batch Prompt Construction
# =============================================================================

def build_batch_prompt(tiles: List[dict]) -> str:
    """Build the batch analysis prompt for N tiles."""
    header = f"""You are analyzing a batch of {len(tiles)} text tiles for motif assignment.

{MOTIF_REFERENCE}

For EACH tile below, respond with a JSON array. Each element must have:
- "hash": the tile hash (first 16 chars shown in header)
- "motifs": array of {{"motif_id": "HMM.X", "amp": 0.85, "confidence": 0.9}}
- "rosetta": 2-sentence dense summary

Respond with ONLY the JSON array, no other text.

```json
[{{"hash": "abc123", "motifs": [{{"motif_id": "HMM.EXAMPLE", "amp": 0.8, "confidence": 0.9}}], "rosetta": "Summary here."}}, ...]
```

Important:
- Only assign motifs with amp >= 0.10 and confidence >= 0.30
- Typically 3-8 motifs per tile
- Sort motifs by amplitude (highest first)
- The rosetta MUST be self-contained

"""

    tile_sections = []
    for tile in tiles:
        ch = tile.get("content_hash", "")[:16]
        content = tile.get("content", "")
        tile_sections.append(f"--- TILE {ch} ---\n{content}")

    return header + "\n\n".join(tile_sections)


# =============================================================================
# Platform Interaction (adapted from hmm_family_processor.py)
# =============================================================================

def check_platform(platform: str) -> bool:
    """Check if platform tab is accessible."""
    shortcut = TAB_SHORTCUTS.get(platform)
    if not shortcut:
        return False

    inp.focus_firefox()
    time.sleep(0.3)
    inp.press_key(shortcut)
    time.sleep(1.0)

    firefox = atspi.find_firefox()
    if not firefox:
        return False

    doc = atspi.get_platform_document(firefox, platform)
    return doc is not None


def _find_input_area(platform: str) -> Optional[Tuple[int, int]]:
    """Find the input area coordinates dynamically."""
    firefox = atspi.find_firefox()
    if not firefox:
        return None
    doc = atspi.get_platform_document(firefox, platform)
    if not doc:
        return None

    elements = find_elements(doc, max_depth=15)

    # Strategy 1: entry/text input
    input_names = ["prompt", "message", "enter a", "write your", "ask"]
    for e in elements:
        role = e.get("role", "")
        name = e.get("name", "").lower()
        if role == "entry" and any(kw in name for kw in input_names):
            x, y = e.get("x", 0), e.get("y", 0)
            if x > 0 and y > 0:
                return (x, y)

    # Strategy 2: landmark buttons
    buttons = [e for e in elements if e.get("role") == "push button"]
    landmark_names = [
        "Add files and more", "Attach", "Add files or tools",
        "Open upload file menu", "Toggle menu", "Send prompt", "Send Message",
    ]
    for b in buttons:
        bname = b.get("name", "")
        for lm in landmark_names:
            if lm.lower() in bname.lower():
                bx, by = b.get("x", 0), b.get("y", 0)
                if bx > 0 and by > 0:
                    return (bx + 200, by - 15)

    return None


def _clipboard_paste(text: str) -> bool:
    """Write text to clipboard and paste via Ctrl+V."""
    try:
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            env=os.environ,
        )
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        proc.wait(timeout=10)
    except Exception as e:
        log.error(f"Clipboard write failed: {e}")
        return False

    time.sleep(0.3)
    inp.press_key("ctrl+v")
    time.sleep(0.5)
    return True


def send_batch_to_platform(platform: str, prompt: str) -> bool:
    """Send a batch prompt to a platform and press Enter."""
    shortcut = TAB_SHORTCUTS.get(platform)
    if not shortcut:
        return False

    inp.focus_firefox()
    time.sleep(0.3)
    inp.press_key(shortcut)
    time.sleep(1.0)

    # Find and click input area
    coords = _find_input_area(platform)
    if not coords:
        log.error(f"[{platform}] Cannot find input area")
        return False

    inp.mouse_click(coords[0], coords[1])
    time.sleep(0.3)

    # Paste prompt via clipboard
    if not _clipboard_paste(prompt):
        return False

    time.sleep(0.5)
    inp.press_key("Return")
    log.info(f"[{platform}] Sent batch ({len(prompt):,} chars)")
    return True


def _find_copy_button_nodes(doc) -> list:
    """Walk AT-SPI tree to find Copy button nodes."""
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi

    results = []

    def walk(obj, depth=0):
        if depth > 15:
            return
        try:
            name = (obj.get_name() or '').lower()
            role = obj.get_role_name() or ''
            if 'button' in role and 'copy' in name:
                comp = obj.get_component_iface()
                y = 0
                if comp:
                    rect = comp.get_extents(Atspi.CoordType.SCREEN)
                    y = rect.y + (rect.height // 2) if rect else 0
                results.append((obj, name, y))
            for i in range(obj.get_child_count()):
                child = obj.get_child_at_index(i)
                if child:
                    walk(child, depth + 1)
        except Exception:
            pass

    walk(doc)
    return results


def extract_response(platform: str) -> Optional[str]:
    """Extract latest response via copy button."""
    inp.press_key("End")
    time.sleep(0.5)

    firefox = atspi.find_firefox()
    if not firefox:
        return None
    doc = atspi.get_platform_document(firefox, platform)
    if not doc:
        return None

    copy_nodes = _find_copy_button_nodes(doc)
    if copy_nodes:
        response_nodes = [(n, name, y) for n, name, y in copy_nodes if name.strip() == 'copy']
        targets = sorted(response_nodes or copy_nodes, key=lambda t: t[2], reverse=True)

        for node, name, y in targets:
            try:
                clipboard.clear()
                time.sleep(0.2)
                action = node.get_action_iface()
                if action and action.get_n_actions() > 0:
                    action.do_action(0)
                    time.sleep(1.0)
                    content = clipboard.read()
                    if content and len(content.strip()) > 20:
                        return content.strip()
            except Exception:
                pass

    return None


def wait_for_response(platform: str) -> Optional[str]:
    """Wait for AI response using stop-button polling."""
    stop_seen = False
    start = time.time()

    while time.time() - start < MAX_WAIT_RESPONSE:
        time.sleep(POLL_INTERVAL)
        elapsed = time.time() - start

        firefox = atspi.find_firefox()
        if not firefox:
            continue
        doc = atspi.get_platform_document(firefox, platform)
        if not doc:
            continue

        # Check for stop button
        patterns = STOP_PATTERNS.get(platform, ['stop'])
        stop_found = False

        def find_stop(obj, depth=0):
            nonlocal stop_found
            if depth > 25 or stop_found:
                return
            try:
                role = obj.get_role_name() or ''
                name = (obj.get_name() or '').lower()
                if role in ('push button', 'button'):
                    if any(p in name for p in patterns):
                        stop_found = True
                        return
                for i in range(obj.get_child_count()):
                    child = obj.get_child_at_index(i)
                    if child:
                        find_stop(child, depth + 1)
            except Exception:
                pass

        find_stop(doc)

        if stop_found:
            if not stop_seen:
                stop_seen = True
                log.info(f"  [{platform}] Generating...")
        else:
            if stop_seen:
                log.info(f"  [{platform}] Response complete ({elapsed:.0f}s)")
                time.sleep(3)
                return extract_response(platform)

        # Periodic extraction attempt
        if not stop_seen and elapsed > 60:
            response = extract_response(platform)
            if response and len(response) > 100:
                log.info(f"  [{platform}] Found response via periodic extract ({len(response)} chars)")
                return response

        if int(elapsed) > 0 and int(elapsed) % 60 == 0:
            state_str = "GENERATING" if stop_seen else "IDLE"
            log.info(f"  [{platform}] Waiting... ({int(elapsed)}s, {state_str})")

    # Timeout - try extraction
    log.warning(f"[{platform}] Timeout after {MAX_WAIT_RESPONSE}s")
    return extract_response(platform)


# =============================================================================
# Response Parsing
# =============================================================================

def parse_batch_response(response_text: str) -> Optional[List[dict]]:
    """Parse a batch JSON array response using 5-strategy pipeline.

    Returns list of {"hash": ..., "motifs": [...], "rosetta": ...} or None.
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Strategy 1: ```json [...] ``` code block
    json_match = re.search(r"```json\s*\n(\[.*?\])\s*\n```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Generic code block
    code_match = re.search(r"```\s*\n(\[.*?\])\s*\n```", text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Outermost [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 4: Raw parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 5: Truncation repair
    repaired = _repair_truncated_json(text)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return None


def _repair_truncated_json(text: str) -> Optional[str]:
    """Attempt to repair truncated JSON by closing open brackets/braces."""
    start = text.find("[")
    if start < 0:
        return None

    fragment = text[start:]

    # Count open/close brackets
    opens = fragment.count("[") - fragment.count("]")
    braces = fragment.count("{") - fragment.count("}")

    # Try to find the last complete object
    last_complete = fragment.rfind("}")
    if last_complete < 0:
        return None

    repaired = fragment[:last_complete + 1]

    # Close open braces
    for _ in range(max(0, braces - (fragment[:last_complete + 1].count("{")
                                    - fragment[:last_complete + 1].count("}")))):
        repaired += "}"

    # Close the array
    repaired += "]"

    try:
        result = json.loads(repaired)
        if isinstance(result, list):
            return repaired
    except json.JSONDecodeError:
        pass

    return None


# =============================================================================
# Storage
# =============================================================================

def store_batch_results(
    results: List[dict],
    tile_map: Dict[str, dict],
    platform: str,
    batch_num: int,
    session: requests.Session,
    neo4j: HMMNeo4jStore,
    redis_store: HMMRedisStore,
    event_log: EventLog,
    gate: GateB,
):
    """Store Tier 2 classification results for a batch.

    For each classified tile:
    - Weaviate PATCH with tier2 properties
    - Neo4j: upsert HMMTile + EXPRESSES relationships
    - Redis: update inverted index
    - Event log: MOTIFS_ASSIGNED event
    """
    stored = 0
    for result in results:
        tile_hash = result.get("hash", "")
        motifs_data = result.get("motifs", [])
        rosetta = result.get("rosetta", "")

        if not tile_hash or not motifs_data:
            continue

        # Find matching tile by hash prefix
        matched_tile = None
        for ch, tile_info in tile_map.items():
            if ch.startswith(tile_hash) or tile_hash.startswith(ch[:16]):
                matched_tile = tile_info
                break

        if not matched_tile:
            log.warning(f"  No tile match for hash {tile_hash}")
            continue

        uuid = matched_tile["uuid"]
        full_hash = matched_tile["content_hash"]

        # Build dominant motifs and motif_data_json
        dominant = [m["motif_id"] for m in motifs_data
                    if m.get("amp", 0) >= 0.6]
        motif_blob = json.dumps([{
            "motif_id": m["motif_id"],
            "amp": m.get("amp", 0),
            "confidence": m.get("confidence", 0),
            "tier": "tier2_family",
            "source_platform": platform,
        } for m in motifs_data])

        # Weaviate PATCH
        payload = {
            "dominant_motifs": dominant,
            "motif_data_json": motif_blob,
            "rosetta_summary": rosetta,
            "hmm_enriched": True,
            "hmm_enrichment_version": ENRICHMENT_VERSION,
            "hmm_enriched_at": datetime.now(timezone.utc).isoformat(),
            "hmm_consensus": True,
            "hmm_gate_flags": ["TIER2_FAMILY", platform.upper()],
            "hmm_platforms": [platform],
        }

        try:
            r = session.patch(
                f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{uuid}",
                json={"properties": payload},
                timeout=10,
            )
            if r.status_code not in (200, 204):
                log.warning(f"  PATCH failed {uuid}: {r.status_code}")
                continue
        except Exception as e:
            log.warning(f"  PATCH error {uuid}: {e}")
            continue

        # Neo4j: upsert tile + motif relationships
        try:
            neo4j.upsert_tile(
                tile_id=full_hash,
                artifact_id=full_hash,
                index=0,
                start_char=0,
                end_char=0,
                estimated_tokens=0,
                scale="search_512",
            )

            assignments = []
            for m in motifs_data:
                assignment = MotifAssignment(
                    motif_id=m["motif_id"],
                    amp=m.get("amp", 0),
                    phase="tier2",
                    confidence=m.get("confidence", 0),
                    source="family",
                    dictionary_version=DICTIONARY_VERSION,
                )
                neo4j.link_tile_motif(full_hash, assignment,
                                      model_id=f"tier2_{platform}")
                assignments.append(assignment)

                # Redis inverted index
                redis_store.inv_add(m["motif_id"], full_hash)

        except Exception as e:
            log.warning(f"  Neo4j/Redis error for {full_hash[:16]}: {e}")

        # Event log
        try:
            gate_result = gate.evaluate(
                motifs=[(m["motif_id"], m.get("amp", 0)) for m in motifs_data],
                provenance=f"tier2_family_batch_{batch_num}_{platform}",
            )
            event_log.log_motifs_assigned(
                tile_id=full_hash,
                motifs=motifs_data,
                gate=GateSnapshot(
                    phi=gate_result.phi,
                    trust=gate_result.trust,
                    flags=gate_result.flags,
                ),
                refs={
                    "source_platform": f"family_batch_{batch_num}",
                    "platform": platform,
                    "tier": "tier2",
                },
            )
        except Exception as e:
            log.debug(f"  Event log error: {e}")

        stored += 1

    return stored


def save_response(response_text: str, platform: str, batch_num: int):
    """Save raw response for debugging."""
    os.makedirs(RESPONSE_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(RESPONSE_LOG_DIR) / f"tier2_batch{batch_num}_{platform}_{ts}.json"
    data = {
        "platform": platform,
        "batch_num": batch_num,
        "timestamp": ts,
        "response": response_text,
    }
    path.write_text(json.dumps(data, indent=2))


# =============================================================================
# Main Operations
# =============================================================================

def show_select(session: requests.Session):
    """Show candidate count by priority."""
    log.info("=== CANDIDATE SELECTION ===")
    candidates = select_candidates(session)
    total = 0
    for priority in ["P1", "P2", "P3", "P4"]:
        count = len(candidates[priority])
        total += count
        log.info(f"  {priority}: {count:,} tiles")
    log.info(f"  Total: {total:,} tiles")
    log.info(f"  Batches needed: ~{(total + BATCH_SIZE - 1) // BATCH_SIZE}")


def dry_run(session: requests.Session, limit: int = 0):
    """Show batch plan without sending."""
    log.info("=== DRY RUN ===")
    candidates = select_candidates(session, limit=limit)

    # Flatten all candidates
    all_tiles = []
    for priority in ["P1", "P2", "P3", "P4"]:
        for tile in candidates[priority]:
            tile["priority_class"] = priority
            all_tiles.append(tile)

    batches = [all_tiles[i:i + BATCH_SIZE]
               for i in range(0, len(all_tiles), BATCH_SIZE)]

    log.info(f"Total tiles: {len(all_tiles)}")
    log.info(f"Batches: {len(batches)} x {BATCH_SIZE}")

    for i, batch in enumerate(batches[:5]):  # Show first 5
        priorities = {}
        for t in batch:
            pc = t.get("priority_class", "?")
            priorities[pc] = priorities.get(pc, 0) + 1
        log.info(f"  Batch {i+1}: {len(batch)} tiles {priorities}")
        if batch:
            sample = batch[0]
            log.info(f"    Sample: {sample['content_hash'][:16]} "
                     f"({len(sample.get('content', ''))} chars)")

    if len(batches) > 5:
        log.info(f"  ... and {len(batches) - 5} more batches")


def run_batches(session: requests.Session, limit: int = 0,
                resume: bool = False):
    """Process all candidate batches via Family platforms."""
    log.info("=== RUN TIER 2 BATCHES ===")

    # Load state
    state = load_state() if resume else {
        "batches_completed": 0,
        "tiles_classified": 0,
        "current_batch_hashes": [],
        "platforms_used": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Initialize stores
    neo4j = HMMNeo4jStore()
    redis_store = HMMRedisStore()
    event_log = EventLog(EVENT_LOG_PATH)
    gate = GateB()

    # Check available platforms
    available = []
    for p in PLATFORMS:
        if check_platform(p):
            available.append(p)
            log.info(f"  {p}: available")
        else:
            log.warning(f"  {p}: NOT available")

    if not available:
        log.error("No platforms available")
        neo4j.close()
        return

    # Select candidates
    candidates = select_candidates(session, limit=limit)
    all_tiles = []
    for priority in ["P1", "P2", "P3", "P4"]:
        for tile in candidates[priority]:
            tile["priority_class"] = priority
            all_tiles.append(tile)

    log.info(f"Total tiles to process: {len(all_tiles)}")

    # Skip already-processed batches (resume)
    start_batch = state["batches_completed"]

    batches = [all_tiles[i:i + BATCH_SIZE]
               for i in range(0, len(all_tiles), BATCH_SIZE)]

    platform_idx = 0
    platform_errors = {p: 0 for p in available}

    for batch_num, batch in enumerate(batches):
        if batch_num < start_batch:
            continue

        # Select platform (round-robin among available)
        platform = available[platform_idx % len(available)]
        platform_idx += 1

        # Skip if platform has too many errors
        if platform_errors[platform] >= MAX_CONSECUTIVE_ERRORS:
            log.warning(f"  [{platform}] Too many errors, skipping")
            # Try next platform
            found_alt = False
            for alt in available:
                if platform_errors[alt] < MAX_CONSECUTIVE_ERRORS:
                    platform = alt
                    found_alt = True
                    break
            if not found_alt:
                log.error("All platforms have too many errors. Stopping.")
                break

        log.info(f"Batch {batch_num + 1}/{len(batches)}: "
                 f"{len(batch)} tiles -> {platform}")

        # Build prompt
        prompt = build_batch_prompt(batch)

        # Build tile map for result matching
        tile_map = {t["content_hash"]: t for t in batch}

        # Send to platform
        if not send_batch_to_platform(platform, prompt):
            log.error(f"  [{platform}] Send failed")
            platform_errors[platform] += 1
            continue

        # Wait for response
        response = wait_for_response(platform)
        if not response:
            log.error(f"  [{platform}] No response received")
            platform_errors[platform] += 1
            save_state(state)
            time.sleep(BETWEEN_BATCHES)
            continue

        # Save raw response
        save_response(response, platform, batch_num + 1)

        # Parse response
        results = parse_batch_response(response)
        if not results:
            log.error(f"  [{platform}] Failed to parse response ({len(response)} chars)")
            platform_errors[platform] += 1
            save_state(state)
            time.sleep(BETWEEN_BATCHES)
            continue

        log.info(f"  [{platform}] Parsed {len(results)} tile results")
        platform_errors[platform] = 0  # Reset on success

        # Store results
        stored = store_batch_results(
            results, tile_map, platform, batch_num + 1,
            session, neo4j, redis_store, event_log, gate,
        )
        log.info(f"  Stored: {stored}/{len(results)} tiles")

        # Update state
        state["batches_completed"] = batch_num + 1
        state["tiles_classified"] += stored
        if platform not in state["platforms_used"]:
            state["platforms_used"].append(platform)
        save_state(state)

        time.sleep(BETWEEN_BATCHES)

    log.info(f"Tier 2 complete: {state['batches_completed']} batches, "
             f"{state['tiles_classified']} tiles classified")
    neo4j.close()


def validate(session: requests.Session):
    """Compare Tier 1 vs Tier 2 agreement on shared tiles."""
    log.info("=== VALIDATE TIER 1 vs TIER 2 ===")

    # Find tiles with both tier1 and tier2 data
    # (P4 validation sample should have both)
    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ operator: And, operands: [
                    {{ path: ["hmm_enrichment_version"], operator: Like, valueText: "tier2*" }},
                    {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                ] }}
                limit: 100
            ) {{
                content_hash
                dominant_motifs
                motif_data_json
                rosetta_summary
                hmm_enrichment_version
                hmm_platforms
                _additional {{ id }}
            }}
        }}
    }}"""

    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                         json={"query": gql}, timeout=15)
        if r.status_code != 200:
            log.error(f"Query failed: {r.status_code}")
            return

        objects = r.json()["data"]["Get"][WEAVIATE_CLASS]
        log.info(f"Tier 2 enriched tiles: {len(objects)}")

        if not objects:
            log.info("No Tier 2 tiles found yet")
            return

        # Show sample
        for obj in objects[:5]:
            ch = obj.get("content_hash", "")[:16]
            dm = obj.get("dominant_motifs") or []
            rs = (obj.get("rosetta_summary") or "")[:80]
            ver = obj.get("hmm_enrichment_version", "")
            plat = obj.get("hmm_platforms") or []
            log.info(f"  {ch} [{ver}] platforms={plat}")
            log.info(f"    dominant: {dm}")
            log.info(f"    rosetta: {rs}...")

    except Exception as e:
        log.error(f"Validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tier 2 Family LLM batch review")
    parser.add_argument("--select", action="store_true",
                        help="Show candidate count by priority")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show batch plan without sending")
    parser.add_argument("--run", action="store_true",
                        help="Process all candidates")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit tiles to process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from state")
    parser.add_argument("--validate", action="store_true",
                        help="Compare Tier 1 vs Tier 2 agreement")
    args = parser.parse_args()

    if not any([args.select, args.dry_run, args.run, args.validate]):
        parser.print_help()
        return

    session = requests.Session()

    # Verify Weaviate
    try:
        r = session.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
        if r.status_code != 200:
            log.error(f"Weaviate not healthy: {r.status_code}")
            return
    except Exception as e:
        log.error(f"Weaviate unreachable: {e}")
        return

    if args.select:
        show_select(session)

    if args.dry_run:
        dry_run(session, limit=args.limit)

    if args.run:
        run_batches(session, limit=args.limit, resume=args.resume)

    if args.validate:
        validate(session)


if __name__ == "__main__":
    main()
