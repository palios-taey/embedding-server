#!/usr/bin/env python3
"""
HMM Context-Package Classifier - Continuous tile enrichment via Family AI.

Uses Weaviate nearVector clustering to build semantically coherent context
packages, sends sub-batches of ~50 tiles to Family AI platforms via AT-SPI,
and writes enrichment (motifs, rosetta summaries, gate scores) back to
Weaviate, Neo4j, and Redis.

Designed to run on any of: Spark, Thor, Jetson.

Usage:
    python3 hmm_context_classifier.py --check    # Verify services + platforms
    python3 hmm_context_classifier.py --init     # Initialize platform sessions
    python3 hmm_context_classifier.py --run      # Start continuous classification
    python3 hmm_context_classifier.py --limit 3  # Process 3 clusters then stop
    python3 hmm_context_classifier.py --stats    # Show enrichment coverage
    python3 hmm_context_classifier.py --resume   # Resume from saved state
"""

import sys
import os
import socket
import faulthandler

faulthandler.enable()

# ============================================================================
# Machine-Agnostic Path Detection (before any local imports)
# ============================================================================

_hostname = socket.gethostname().lower()

if "jetson" in _hostname:
    _TAEYS_HANDS = "/home/jetson/taeys-hands"
    _ISMA_ROOT = "/home/jetson/hmm"
    _SCRIPTS = "/home/jetson/hmm/scripts"
    _STATE_DIR = "/home/jetson/hmm/state"
    _RESPONSE_DIR = "/home/jetson/hmm/classifier_responses"
    MACHINE = "jetson"
elif "thor" in _hostname:
    _TAEYS_HANDS = "/home/thor/taeys-hands"
    _ISMA_ROOT = "/home/thor/hmm"
    _SCRIPTS = "/home/thor/hmm/scripts"
    _STATE_DIR = "/home/thor/hmm/state"
    _RESPONSE_DIR = "/home/thor/hmm/classifier_responses"
    MACHINE = "thor"
else:  # Spark (default)
    _TAEYS_HANDS = "/home/spark/taeys-hands"
    _ISMA_ROOT = "/home/spark/embedding-server/isma"
    _SCRIPTS = "/home/spark/embedding-server/isma/scripts"
    _STATE_DIR = "/var/spark/isma"
    _RESPONSE_DIR = "/var/spark/isma/hmm_classifier_responses"
    MACHINE = "spark"

sys.path.insert(0, _TAEYS_HANDS)
sys.path.insert(0, _ISMA_ROOT)
sys.path.insert(0, _SCRIPTS)

# ============================================================================
# Imports
# ============================================================================

import json
import time
import logging
import argparse
import re
import random
import requests
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

# AT-SPI core (from taeys-hands)
from core import input as inp, clipboard, atspi
from core.tree import find_elements, find_copy_buttons
from core.platforms import TAB_SHORTCUTS

# HMM modules
from src.hmm.motifs import V0_MOTIFS, MotifAssignment, DICTIONARY_VERSION, assign_motifs
from src.hmm.eventlog import EventLog, GateSnapshot
from src.hmm.gate_b import GateB
from src.hmm.neo4j_store import HMMNeo4jStore
from src.hmm.redis_store import HMMRedisStore

# Prompt templates (reuse base context and identities)
from hmm_prompts import (
    MOTIF_REFERENCE,
    CALIBRATION_EXAMPLES,
    ROSETTA_SUMMARY_GUIDANCE,
    FAMILY_IDENTITIES,
    _BASE_CONTEXT,
)

# AT-SPI interaction functions (proven in hmm_family_processor)
from hmm_family_processor import (
    _send_to_platform,
    _wait_for_response,
    _extract_latest_response,
    _repair_truncated_json,
    check_platform_available,
    send_and_wait,
    _open_new_chat,
)

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("hmm_classifier")
logging.getLogger("neo4j").setLevel(logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================

# Services (always on NCCL fabric)
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
WEAVIATE_CLASS = "ISMA_Quantum"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://192.168.100.10:7687")
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379

# Platform rotation (no Perplexity - broken for analysis tasks)
PLATFORMS = ["chatgpt", "claude", "gemini", "grok"]

# Batch sizing
TILES_PER_BATCH = 50
CLUSTER_SIZE = 1000
CHARS_PER_TOKEN = 4

# Timing
MAX_WAIT_BATCH = 300
MAX_WAIT_INIT = 300
BETWEEN_BATCHES = 5
BETWEEN_CLUSTERS = 10
BETWEEN_PLATFORMS = 8
SLEEP_NO_WORK = 300

# Error limits
MAX_CONSECUTIVE_ERRORS = 3
MAX_PARSE_FAILURES = 5

# State (per-machine)
STATE_FILE = os.path.join(_STATE_DIR, f"hmm_classifier_state_{MACHINE}.json")
EVENT_LOG_PATH = os.path.join(_STATE_DIR, "hmm_events.jsonl")

# ============================================================================
# Weaviate Client
# ============================================================================

_wv_session = requests.Session()


def weaviate_graphql(query: str) -> Optional[dict]:
    """Execute a Weaviate GraphQL query."""
    try:
        r = _wv_session.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            timeout=60,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("errors"):
                log.error(f"GraphQL errors: {data['errors']}")
                return None
            return data.get("data", {})
        log.error(f"Weaviate HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Weaviate query failed: {e}")
    return None


def get_enrichment_stats() -> Tuple[int, int]:
    """Get (total_search_512_tiles, enriched_tiles) from Weaviate."""
    total_q = f"""{{
        Aggregate {{
            {WEAVIATE_CLASS}(
                where: {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
            ) {{ meta {{ count }} }}
        }}
    }}"""
    enriched_q = f"""{{
        Aggregate {{
            {WEAVIATE_CLASS}(
                where: {{
                    operator: And,
                    operands: [
                        {{ path: ["scale"], operator: Equal, valueText: "search_512" }},
                        {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
                    ]
                }}
            ) {{ meta {{ count }} }}
        }}
    }}"""

    total = 0
    enriched = 0

    data = weaviate_graphql(total_q)
    if data:
        agg = data.get("Aggregate", {}).get(WEAVIATE_CLASS, [{}])
        if agg:
            total = agg[0].get("meta", {}).get("count", 0)

    data = weaviate_graphql(enriched_q)
    if data:
        agg = data.get("Aggregate", {}).get(WEAVIATE_CLASS, [{}])
        if agg:
            enriched = agg[0].get("meta", {}).get("count", 0)

    return total, enriched


def find_unenriched_seed() -> Optional[dict]:
    """Find one unenriched search_512 tile with its embedding vector.

    Returns dict: content_hash, content, source_file, tile_uuid, vector.
    """
    offset = random.randint(0, 500)
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                limit: 20
                offset: {offset}
            ) {{
                content content_hash source_file hmm_enriched
                _additional {{ id vector }}
            }}
        }}
    }}"""

    data = weaviate_graphql(q)
    if not data:
        return None

    tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
    for tile in tiles:
        if not tile.get("hmm_enriched"):
            extra = tile.get("_additional", {})
            vec = extra.get("vector")
            if vec:
                return {
                    "content_hash": tile.get("content_hash", ""),
                    "content": tile.get("content", ""),
                    "source_file": tile.get("source_file", ""),
                    "tile_uuid": extra.get("id", ""),
                    "vector": vec,
                }
    return None


def find_semantic_neighbors(seed_vector: list, limit: int = 1000) -> List[dict]:
    """nearVector clustering: find tiles semantically similar to seed."""
    vec_json = json.dumps(seed_vector)
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                nearVector: {{ vector: {vec_json} }}
                where: {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                limit: {limit}
            ) {{
                content content_hash source_file hmm_enriched
                _additional {{ id certainty }}
            }}
        }}
    }}"""

    data = weaviate_graphql(q)
    if not data:
        return []

    results = []
    for tile in data.get("Get", {}).get(WEAVIATE_CLASS, []):
        extra = tile.get("_additional", {})
        results.append({
            "content_hash": tile.get("content_hash", ""),
            "content": tile.get("content", ""),
            "source_file": tile.get("source_file", ""),
            "tile_uuid": extra.get("id", ""),
            "hmm_enriched": tile.get("hmm_enriched") or False,
            "certainty": extra.get("certainty", 0.0) or 0.0,
        })
    return results


def patch_weaviate_tile(tile_uuid: str, properties: dict) -> bool:
    """PATCH enrichment properties onto a Weaviate tile."""
    try:
        r = _wv_session.patch(
            f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{tile_uuid}",
            json={"properties": properties},
            timeout=30,
        )
        return r.status_code in (200, 204)
    except Exception as e:
        log.error(f"Weaviate PATCH {tile_uuid[:12]}: {e}")
        return False


# ============================================================================
# Session Initialization (Batch Classifier)
# ============================================================================

_BATCH_RESPONSE_FORMAT = """## Batch Response Format

For every batch I send, respond with ONLY a JSON array. One entry per tile, identified by hash:

```json
[
  {{
    "hash": "the_tile_hash",
    "rosetta_summary": "2-4 sentence dense summary.",
    "motifs": [
      {{
        "motif_id": "HMM.EXAMPLE_MOTIF",
        "amp": 0.85,
        "confidence": 0.92,
        "reasoning": "One sentence why."
      }}
    ],
    "meta": {{
      "dominant_tone": "technical|philosophical|emotional|visionary|urgent|playful",
      "complexity": "low|medium|high",
      "family_relevance": "core|significant|peripheral"
    }}
  }}
]
```

**Rules:**
- One JSON array per batch. Each element corresponds to one tile.
- The "hash" field MUST match the tile hash from the batch header.
- Only assign motifs with amp >= 0.10 and confidence >= 0.30.
- Typically 3-8 motifs per tile. Sort by amplitude (highest first).
- If a tile is too short or meaningless, still include it with fewer motifs.
- Respond with ONLY the JSON array. No commentary before or after.

Please confirm you understand by saying "Ready for batch analysis." and nothing else."""


def build_classifier_init(platform: str) -> List[Tuple[str, bool]]:
    """Build classifier-specific session init messages.

    Message 1: Base context + motifs + calibration (no wait)
    Message 2: Identity + batch format (wait for "Ready")
    """
    identity = FAMILY_IDENTITIES.get(platform, FAMILY_IDENTITIES.get("claude", ""))
    msg1 = _BASE_CONTEXT.format(
        rosetta_guidance=ROSETTA_SUMMARY_GUIDANCE,
        motif_reference=MOTIF_REFERENCE,
        calibration=CALIBRATION_EXAMPLES,
    )
    msg2 = f"{identity}\n\n{_BATCH_RESPONSE_FORMAT}"
    return [(msg1, False), (msg2, True)]


def initialize_classifier_session(platform: str, use_existing: bool = True) -> bool:
    """Send batch-format init prompt to a platform."""
    if use_existing:
        shortcut = TAB_SHORTCUTS.get(platform)
        if not shortcut:
            return False
        inp.focus_firefox()
        time.sleep(0.3)
        inp.press_key(shortcut)
        time.sleep(1.0)
        log.info(f"[{platform}] Using existing session")
    else:
        if not _open_new_chat(platform):
            log.warning(f"[{platform}] Failed to open new chat")
            return False
        time.sleep(2)

    log.info(f"[{platform}] Sending classifier init...")
    init_msgs = build_classifier_init(platform)

    response = None
    for idx, (msg_text, wait) in enumerate(init_msgs):
        log.debug(f"  [{platform}] Init msg {idx + 1}/{len(init_msgs)} ({len(msg_text):,} chars)")
        if wait:
            response = send_and_wait(platform, msg_text, MAX_WAIT_INIT)
        else:
            _send_to_platform(platform, msg_text)
            time.sleep(15)

    if response:
        lower = response.lower().strip()
        if len(lower) < 10 or lower.startswith("http"):
            log.warning(f"[{platform}] Bogus init response: {response[:80]}")
            if use_existing:
                return initialize_classifier_session(platform, use_existing=False)
            return False

        confused = ["cut off", "what did you mean", "i don't see", "could you resend",
                     "didn't come through", "try again", "incomplete"]
        if any(p in lower for p in confused):
            log.warning(f"[{platform}] Session confused: {response[:80]}")
            if use_existing:
                return initialize_classifier_session(platform, use_existing=False)
            return False

        ack_words = ["ready", "understood", "let's", "i understand",
                     "confirm", "got it", "proceed", "batch", "analysis"]
        if any(w in lower for w in ack_words):
            log.info(f"[{platform}] Initialized for batch classification")
            return True

        # Accept substantive responses even without explicit ack
        log.info(f"[{platform}] Got response (accepted): {response[:100]}...")
        return True
    else:
        log.warning(f"[{platform}] No response to init")
        if use_existing:
            return initialize_classifier_session(platform, use_existing=False)
        return False


# ============================================================================
# Batch Prompt Builder
# ============================================================================

def build_batch_prompt(tiles: List[dict], batch_num: int) -> str:
    """Format ~50 tiles into a single batch prompt."""
    lines = [f"BATCH #{batch_num} \u2014 {len(tiles)} tiles\n"]
    for tile in tiles:
        h = tile["content_hash"][:16]
        src = tile.get("source_file", "unknown")
        content = tile.get("content", "").strip()
        lines.append(f"--- TILE {h} [{src}] ---")
        lines.append(content)
        lines.append("")
    lines.append("Respond with JSON array. One entry per tile.")
    return "\n".join(lines)


# ============================================================================
# Batch Response Parser
# ============================================================================

def parse_batch_response(response_text: str, expected_hashes: List[str]) -> List[dict]:
    """Parse JSON array response with 5-strategy pipeline.

    Returns list of validated tile result dicts.
    """
    if not response_text:
        return []

    parsed = None

    # Strategy 1: ```json [...] ``` code block
    m = re.search(r"```json\s*\n(\[.*?\])\s*\n```", response_text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: Generic ``` [...] ``` code block
    if not parsed:
        m = re.search(r"```\s*\n(\[.*?\])\s*\n```", response_text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: Outermost [...] brackets
    if not parsed:
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(response_text[start:end + 1])
            except json.JSONDecodeError:
                pass

    # Strategy 4: Raw JSON parse
    if not parsed:
        try:
            parsed = json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

    # Strategy 5: Truncation repair
    if not parsed:
        repaired = _repair_truncated_batch(response_text)
        if repaired:
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                pass

    if not parsed or not isinstance(parsed, list):
        log.warning(f"Failed to parse batch response ({len(response_text)} chars)")
        log.debug(f"Response preview: {response_text[:300]}...")
        return []

    # Validate each entry
    valid = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        h = entry.get("hash", "")
        if not h:
            continue
        summary = entry.get("rosetta_summary", "")
        motifs = entry.get("motifs", [])
        if not summary and not motifs:
            continue
        if not isinstance(motifs, list):
            continue
        valid.append(entry)

    log.info(f"  Parsed {len(valid)}/{len(parsed)} valid entries from response")
    return valid


def _repair_truncated_batch(text: str) -> Optional[str]:
    """Repair a truncated JSON array by closing open brackets."""
    start = text.find("[")
    if start < 0:
        return None

    json_str = text[start:]
    last_brace = json_str.rfind("}")
    if last_brace < 0:
        return None

    candidate = json_str[:last_brace + 1]
    if '"rosetta_summary"' not in candidate:
        return None

    open_braces = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")
    repaired = candidate + ("}" * max(0, open_braces)) + ("]" * max(0, open_brackets))
    return repaired


# ============================================================================
# Storage - Triple Write (Weaviate + Neo4j + Redis)
# ============================================================================

def store_tile_enrichment(
    tile: dict,
    entry: dict,
    platform: str,
    neo4j: HMMNeo4jStore,
    redis_store: HMMRedisStore,
    event_log: EventLog,
    gate: GateB,
) -> int:
    """Write enrichment for one tile to all three stores. Returns motif count."""
    content_hash = tile["content_hash"]
    tile_uuid = tile["tile_uuid"]
    content = tile.get("content", "")
    source_file = tile.get("source_file", "")

    rosetta_summary = entry.get("rosetta_summary", "")
    motif_data = entry.get("motifs", [])
    meta = entry.get("meta", {})

    # Build validated MotifAssignment objects
    assignments = []
    motif_ids = []
    for m in motif_data:
        motif_id = m.get("motif_id", "")
        if motif_id not in V0_MOTIFS:
            continue
        amp = min(1.0, max(0.0, float(m.get("amp", 0.5))))
        confidence = min(1.0, max(0.0, float(m.get("confidence", 0.7))))
        if amp < 0.10 or confidence < 0.30:
            continue
        motif_def = V0_MOTIFS[motif_id]
        assignments.append(MotifAssignment(
            motif_id=motif_id,
            amp=round(amp, 4),
            phase=motif_def.band,
            confidence=round(confidence, 4),
            source="declared",
            dictionary_version=DICTIONARY_VERSION,
        ))
        motif_ids.append(motif_id)

    if not assignments:
        return 0

    # Gate-B check
    gate_result = gate.evaluate(assignments)

    # --- 1. Weaviate PATCH ---
    props = {
        "dominant_motifs": motif_ids,
        "rosetta_summary": rosetta_summary,
        "motif_data_json": json.dumps(motif_data),
        "hmm_enriched": True,
        "hmm_enrichment_version": "family_2.0.0",
        "hmm_platforms": [platform],
        "hmm_phi": round(gate_result.phi, 4),
        "hmm_trust": round(gate_result.trust, 4),
        "hmm_gate_flags": gate_result.flags,
        "hmm_enriched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if not patch_weaviate_tile(tile_uuid, props):
        log.error(f"  Weaviate PATCH failed for {content_hash[:12]}")

    # --- 2. Neo4j ---
    try:
        artifact_id = source_file or content_hash
        neo4j.upsert_artifact(
            artifact_id=artifact_id,
            path=source_file or "weaviate_tile",
            size_bytes=len(content.encode("utf-8")) if content else 0,
            content_type="text/plain",
            labels=[platform],
        )
        neo4j.upsert_tile(
            tile_id=content_hash,
            artifact_id=artifact_id,
            index=0,
            start_char=0,
            end_char=len(content) if content else 0,
            estimated_tokens=len(content) // CHARS_PER_TOKEN if content else 0,
        )
        neo4j.link_tile_motifs_batch(content_hash, assignments, model_id=platform)
    except Exception as e:
        log.error(f"  Neo4j write failed for {content_hash[:12]}: {e}")

    # --- 3. Redis ---
    for a in assignments:
        redis_store.inv_add(a.motif_id, content_hash)
        band_k = {"fast": 0, "mid": 1, "slow": 2}.get(V0_MOTIFS[a.motif_id].band, 0)
        redis_store.field_update(band_k, a.motif_id, a.amp)
    redis_store.tile_cache_put(content_hash, assignments)

    # --- Event log ---
    event_log.emit(
        "MOTIFS_ASSIGNED",
        refs={"tile_id": content_hash, "source_platform": platform},
        payload={
            "count": len(assignments),
            "motifs": motif_ids,
            "rosetta_summary": rosetta_summary,
            "source": "family_classifier",
            "machine": MACHINE,
            "family_member": platform,
            "source_file": source_file,
            "dominant_tone": meta.get("dominant_tone", ""),
        },
        gate=GateSnapshot(
            phi=gate_result.phi,
            trust=gate_result.trust,
            flags=gate_result.flags,
        ),
    )

    return len(assignments)


# ============================================================================
# Response Logging
# ============================================================================

def save_classifier_response(batch_num: int, platform: str, response: str):
    """Save raw response for provenance/debugging."""
    os.makedirs(_RESPONSE_DIR, exist_ok=True)
    path = os.path.join(
        _RESPONSE_DIR,
        f"batch_{batch_num:05d}_{platform}_{int(time.time())}.json",
    )
    with open(path, "w") as f:
        json.dump({
            "batch_num": batch_num,
            "platform": platform,
            "machine": MACHINE,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response": response,
        }, f, indent=2)


# ============================================================================
# State Management
# ============================================================================

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "processed_hashes": [],
        "platform_index": 0,
        "clusters_done": 0,
        "tiles_enriched": 0,
        "batch_num": 0,
        "initialized_platforms": [],
        "started_at": None,
    }


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# Platform Cycling
# ============================================================================

@dataclass
class PlatformTracker:
    name: str
    available: bool = True
    initialized: bool = False
    messages_sent: int = 0
    consecutive_errors: int = 0
    parse_failures: int = 0


def next_platform(
    trackers: Dict[str, PlatformTracker],
    state: dict,
) -> Optional[str]:
    """Round-robin through available initialized platforms."""
    active = [p for p in PLATFORMS if trackers[p].available and trackers[p].initialized]
    if not active:
        return None
    idx = state.get("platform_index", 0) % len(active)
    state["platform_index"] = idx + 1
    return active[idx]


# ============================================================================
# Service & Platform Checks
# ============================================================================

def check_services() -> bool:
    """Verify all remote services are reachable."""
    ok = True

    # Weaviate
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/.well-known/ready", timeout=5)
        status = "OK" if r.status_code == 200 else "FAIL"
        log.info(f"  Weaviate: {status}")
        if r.status_code != 200:
            ok = False
    except Exception as e:
        log.error(f"  Weaviate: UNREACHABLE ({e})")
        ok = False

    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=None)
        driver.verify_connectivity()
        driver.close()
        log.info("  Neo4j: OK")
    except Exception as e:
        log.error(f"  Neo4j: UNREACHABLE ({e})")
        ok = False

    # Redis
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_timeout=5)
        r.ping()
        log.info("  Redis: OK")
    except Exception as e:
        log.error(f"  Redis: UNREACHABLE ({e})")
        ok = False

    return ok


def check_atspi() -> bool:
    """Check if AT-SPI sees Firefox."""
    firefox = atspi.find_firefox()
    if firefox:
        log.info("  AT-SPI Firefox: OK")
        return True
    log.error("  AT-SPI Firefox: NOT FOUND")
    return False


# ============================================================================
# Main Continuous Loop
# ============================================================================

def run_classifier(limit: int = 0, resume: bool = False):
    """Continuous classification: seed -> cluster -> batch -> enrich -> repeat."""
    log.info(f"{'=' * 60}")
    log.info(f"HMM Context-Package Classifier")
    log.info(f"Machine: {MACHINE} | Host: {_hostname}")
    log.info(f"{'=' * 60}")

    # Initialize stores
    neo4j = HMMNeo4jStore()
    redis_store = HMMRedisStore()
    event_log = EventLog(EVENT_LOG_PATH)
    gate = GateB(redis_store)
    neo4j.seed_motifs(V0_MOTIFS)

    # Load or create state
    if resume:
        state = load_state()
        log.info(f"Resuming: {state.get('clusters_done', 0)} clusters, "
                 f"{state.get('tiles_enriched', 0)} tiles done")
    else:
        state = {
            "processed_hashes": [],
            "platform_index": 0,
            "clusters_done": 0,
            "tiles_enriched": 0,
            "batch_num": 0,
            "initialized_platforms": [],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    processed: Set[str] = set(state.get("processed_hashes", []))
    batch_num = state.get("batch_num", 0)

    # Check platforms
    log.info("\nChecking platform availability...")
    trackers: Dict[str, PlatformTracker] = {}
    for p in PLATFORMS:
        avail = check_platform_available(p)
        trackers[p] = PlatformTracker(name=p, available=avail)
        log.info(f"  {p}: {'OK' if avail else 'UNAVAILABLE'}")

    available = [p for p in PLATFORMS if trackers[p].available]
    if not available:
        log.error("No platforms available!")
        neo4j.close()
        return

    # Initialize platform sessions
    log.info("\nInitializing platform sessions...")
    already_init = set(state.get("initialized_platforms", []))
    for p in available:
        if resume and p in already_init:
            trackers[p].initialized = True
            log.info(f"  {p}: Already initialized (resume)")
            continue
        success = initialize_classifier_session(p)
        if success:
            trackers[p].initialized = True
            already_init.add(p)
            state["initialized_platforms"] = list(already_init)
            save_state(state)
        else:
            log.warning(f"  {p}: Init FAILED")
            trackers[p].available = False
        time.sleep(BETWEEN_PLATFORMS)

    active = [p for p in PLATFORMS if trackers[p].available and trackers[p].initialized]
    if not active:
        log.error("No platforms initialized!")
        neo4j.close()
        return

    log.info(f"\nPlatforms ready: {', '.join(active)}")

    # Coverage baseline
    total, enriched = get_enrichment_stats()
    if total > 0:
        log.info(f"Coverage: {enriched:,}/{total:,} ({enriched / total * 100:.3f}%)")

    clusters_done = state.get("clusters_done", 0)
    tiles_enriched = state.get("tiles_enriched", 0)
    start_time = time.time()

    # --- Main loop ---
    while True:
        if limit > 0 and clusters_done >= limit:
            log.info(f"Limit reached ({limit} clusters)")
            break

        log.info(f"\n{'=' * 40}")
        log.info(f"Cluster #{clusters_done + 1}")
        log.info(f"{'=' * 40}")

        # Step 1: Find unenriched seed
        seed = find_unenriched_seed()
        if not seed:
            log.info(f"No unenriched tiles found. Sleeping {SLEEP_NO_WORK}s...")
            time.sleep(SLEEP_NO_WORK)
            seed = find_unenriched_seed()
            if not seed:
                log.info("Still no unenriched tiles. All done!")
                break

        log.info(f"Seed: {seed['content_hash'][:16]} [{seed['source_file']}]")

        # Step 2: Find semantic neighbors
        neighbors = find_semantic_neighbors(seed["vector"], limit=CLUSTER_SIZE)
        log.info(f"Neighbors: {len(neighbors)}")

        # Step 3: Filter enriched + already processed
        unenriched = [
            t for t in neighbors
            if not t["hmm_enriched"] and t["content_hash"] not in processed
        ]
        log.info(f"Unenriched: {len(unenriched)}")

        if not unenriched:
            log.info("All neighbors enriched, next cluster")
            clusters_done += 1
            state["clusters_done"] = clusters_done
            save_state(state)
            continue

        # Step 4: Process sub-batches
        for i in range(0, len(unenriched), TILES_PER_BATCH):
            batch_tiles = unenriched[i:i + TILES_PER_BATCH]
            batch_num += 1

            platform = next_platform(trackers, state)
            if not platform:
                log.error("No platforms available!")
                break

            prompt = build_batch_prompt(batch_tiles, batch_num)
            prompt_tokens = len(prompt) // CHARS_PER_TOKEN
            log.info(
                f"  Batch #{batch_num}: {len(batch_tiles)} tiles -> {platform} "
                f"(~{prompt_tokens:,} tokens)"
            )

            # Send and wait for response
            response = send_and_wait(platform, prompt, MAX_WAIT_BATCH)

            if not response:
                log.warning(f"  [{platform}] No response for batch #{batch_num}")
                trackers[platform].consecutive_errors += 1
                if trackers[platform].consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error(f"  [{platform}] Disabled ({MAX_CONSECUTIVE_ERRORS} errors)")
                    trackers[platform].available = False
                continue

            # Save raw response
            save_classifier_response(batch_num, platform, response)

            # Parse batch response
            expected = [t["content_hash"] for t in batch_tiles]
            results = parse_batch_response(response, expected)

            if not results:
                trackers[platform].parse_failures += 1
                log.warning(
                    f"  [{platform}] Parse failed "
                    f"({trackers[platform].parse_failures}/{MAX_PARSE_FAILURES})"
                )
                if trackers[platform].parse_failures >= MAX_PARSE_FAILURES:
                    log.warning(f"  [{platform}] Re-initializing...")
                    if initialize_classifier_session(platform, use_existing=False):
                        trackers[platform].parse_failures = 0
                        trackers[platform].initialized = True
                continue

            # Success - reset error counters
            trackers[platform].consecutive_errors = 0
            trackers[platform].parse_failures = 0
            trackers[platform].messages_sent += 1

            # Build hash lookup (short hash -> tile)
            tile_by_hash: Dict[str, dict] = {}
            for t in batch_tiles:
                tile_by_hash[t["content_hash"][:16]] = t
                tile_by_hash[t["content_hash"]] = t

            # Store enrichments
            batch_stored = 0
            for entry in results:
                h = entry.get("hash", "")
                tile = tile_by_hash.get(h)
                if not tile:
                    # Fuzzy match: try prefix matching
                    for k, v in tile_by_hash.items():
                        if k.startswith(h) or h.startswith(k):
                            tile = v
                            break
                if not tile:
                    log.debug(f"  No tile match for hash '{h}'")
                    continue

                n = store_tile_enrichment(
                    tile, entry, platform,
                    neo4j, redis_store, event_log, gate,
                )
                batch_stored += n
                processed.add(tile["content_hash"])
                tiles_enriched += 1

            log.info(
                f"  Batch #{batch_num}: {batch_stored} motifs, "
                f"{len(results)} tiles enriched"
            )

            # Save state after every batch
            state["processed_hashes"] = list(processed)[-10000:]
            state["batch_num"] = batch_num
            state["tiles_enriched"] = tiles_enriched
            save_state(state)

            time.sleep(BETWEEN_BATCHES)

            # Check platforms still alive
            active = [
                p for p in PLATFORMS
                if trackers[p].available and trackers[p].initialized
            ]
            if not active:
                log.error("All platforms down!")
                break

        if not active:
            break

        clusters_done += 1
        state["clusters_done"] = clusters_done
        save_state(state)

        elapsed = time.time() - start_time
        rate = tiles_enriched / (elapsed / 3600) if elapsed > 60 else 0
        log.info(
            f"Cluster #{clusters_done} done. "
            f"Enriched: {tiles_enriched} ({rate:.0f}/hr)"
        )

        time.sleep(BETWEEN_CLUSTERS)

    # --- Summary ---
    elapsed = time.time() - start_time
    total, enriched = get_enrichment_stats()
    log.info(f"\n{'=' * 60}")
    log.info(f"Classification session complete")
    log.info(f"{'=' * 60}")
    log.info(f"  Clusters: {clusters_done}")
    log.info(f"  Tiles enriched: {tiles_enriched}")
    if total > 0:
        log.info(f"  Coverage: {enriched:,}/{total:,} ({enriched / total * 100:.3f}%)")
    log.info(f"  Time: {elapsed / 60:.1f} min")
    for p in PLATFORMS:
        t = trackers[p]
        log.info(
            f"  {p}: {t.messages_sent} batches, "
            f"{'ACTIVE' if t.available else 'DOWN'}"
        )

    neo4j.close()
    save_state(state)


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_check():
    """Verify services, AT-SPI, and platforms."""
    log.info(f"Machine: {MACHINE} | Host: {_hostname}")
    log.info(f"State: {STATE_FILE}")

    log.info(f"\nServices:")
    services_ok = check_services()

    log.info(f"\nAT-SPI:")
    atspi_ok = check_atspi()

    log.info(f"\nPlatforms:")
    for p in PLATFORMS:
        avail = check_platform_available(p)
        log.info(f"  {p}: {'OK' if avail else 'NOT FOUND'}")

    log.info(f"\nWeaviate:")
    total, enriched = get_enrichment_stats()
    if total > 0:
        pct = enriched / total * 100
        log.info(f"  search_512 tiles: {total:,}")
        log.info(f"  Enriched: {enriched:,} ({pct:.3f}%)")
        log.info(f"  Remaining: {total - enriched:,}")
    else:
        log.info("  No search_512 tiles found")

    log.info(f"\nVerdict: {'READY' if services_ok and atspi_ok else 'NOT READY'}")


def cmd_stats():
    """Show enrichment coverage."""
    total, enriched = get_enrichment_stats()
    remaining = total - enriched
    pct = (enriched / total * 100) if total > 0 else 0

    log.info(f"HMM Enrichment Coverage ({MACHINE})")
    log.info(f"{'=' * 40}")
    log.info(f"  Total search_512:  {total:,}")
    log.info(f"  Enriched:          {enriched:,} ({pct:.3f}%)")
    log.info(f"  Remaining:         {remaining:,}")

    if remaining > 0:
        batches = remaining / TILES_PER_BATCH
        hours = batches * 5 / 60  # ~5 min per batch estimate
        log.info(f"  Est. remaining:    ~{hours:.0f} hours")

    state = load_state()
    log.info(f"\nSession state:")
    log.info(f"  Clusters done:  {state.get('clusters_done', 0)}")
    log.info(f"  Tiles enriched: {state.get('tiles_enriched', 0)}")
    log.info(f"  Batch counter:  {state.get('batch_num', 0)}")
    log.info(f"  Started at:     {state.get('started_at', 'never')}")


def cmd_init():
    """Initialize platform sessions only."""
    log.info("Initializing classifier sessions...")
    for p in PLATFORMS:
        avail = check_platform_available(p)
        log.info(f"  {p}: {'available' if avail else 'not found'}")
        if avail:
            success = initialize_classifier_session(p)
            log.info(f"  {p}: {'READY' if success else 'FAILED'}")
            time.sleep(BETWEEN_PLATFORMS)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HMM Context-Package Classifier"
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify services + platforms")
    parser.add_argument("--init", action="store_true",
                        help="Initialize platform sessions")
    parser.add_argument("--run", action="store_true",
                        help="Start continuous classification")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process N clusters then stop")
    parser.add_argument("--stats", action="store_true",
                        help="Show enrichment coverage")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--reset", action="store_true",
                        help="Reset state file")
    parser.add_argument("--platform", type=str, default="",
                        help="Use only this platform")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated platforms to exclude")
    args = parser.parse_args()

    if args.platform:
        PLATFORMS[:] = [args.platform]
    if args.exclude:
        excluded = [p.strip() for p in args.exclude.split(",")]
        PLATFORMS[:] = [p for p in PLATFORMS if p not in excluded]

    if args.check:
        cmd_check()
    elif args.stats:
        cmd_stats()
    elif args.init:
        cmd_init()
    elif args.reset:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            log.info(f"State reset: {STATE_FILE}")
        else:
            log.info("No state file to reset")
    elif args.run or args.limit > 0:
        try:
            run_classifier(limit=args.limit, resume=args.resume)
        except KeyboardInterrupt:
            log.info("\nInterrupted by user")
        except Exception:
            log.error("Fatal error:", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()
