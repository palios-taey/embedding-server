#!/usr/bin/env python3
"""6B-3: ColBERT Pilot — Create ISMA_ColBERT_Pilot class and ingest top 20K enriched tiles.

Uses jinaai/jina-colbert-v2 (560M, 64-dim Matryoshka late interaction).
Stores multi-vector per-token embeddings in Weaviate with MaxSim aggregation.

Usage:
    python3 colbert_pilot_ingest.py [--create-class] [--ingest] [--limit 20000]
    python3 colbert_pilot_ingest.py --create-class         # Create class only
    python3 colbert_pilot_ingest.py --ingest --limit 5000  # Ingest N tiles
    python3 colbert_pilot_ingest.py --create-class --ingest # Full pilot setup
    python3 colbert_pilot_ingest.py --stats                # Show ingest status
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
PILOT_CLASS = "ISMA_ColBERT_Pilot"
SOURCE_CLASS = "ISMA_Quantum"

COLBERT_MODEL = "jinaai/jina-colbert-v2"
COLBERT_DIM = 64       # Matryoshka: use first 64 dims for storage efficiency
MAX_DOC_TOKENS = 512   # Max tokens per passage (ColBERT recommendation)
MAX_QUERY_TOKENS = 64  # Max tokens per query
BATCH_SIZE = 32        # Reduced from 256 — GB10 OOM-kills Weaviate with 256-tile batch writes (32K vectors/batch)

# Redis for checkpoint
CHECKPOINT_KEY = "isma:colbert_pilot:checkpoint"

# ── Alert / circuit-breaker config ────────────────────────────────────────────
SHARD_LABEL    = os.environ.get("COLBERT_SHARD_LABEL", "s?")   # e.g. s0, s1
SUPERVISOR     = os.environ.get("TMUX_SUPERVISOR", "weaver")
MAX_CONSEC_FAIL = 3      # consecutive batch failures before hard stop
STALL_SECS      = 1200   # 20 min with no successful writes → alarm + exit

# Shared mutable state (main thread writes, io threads set stop event)
_consecutive_failures = [0]
_stop_event = threading.Event()


def send_alert(msg: str):
    """Send critical alert to supervisor tmux session and log it.

    Tries tmux-send first (works if binary is in container/PATH).
    Falls back to Redis PUBLISH so host-side watchers can pick it up.
    Always writes to an alert file so watch scripts can detect it.
    """
    full_msg = f"COLBERT-{SHARD_LABEL}: {msg}"
    log.error(f"ALERT: {full_msg}")

    # Always write to file — watch script checks this
    try:
        with open(f"/tmp/colbert_{SHARD_LABEL}.alert", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {full_msg}\n")
    except Exception:
        pass

    # Try tmux-send (available on host, may not be in container)
    try:
        subprocess.run(
            ["tmux-send", SUPERVISOR, full_msg],
            timeout=5, capture_output=True
        )
        return
    except Exception:
        pass

    # Try Redis PUBLISH as fallback
    try:
        import redis as _redis
        r = _redis.Redis(
            host=os.environ.get("REDIS_HOST", "192.168.100.10"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True
        )
        r.publish("colbert:alerts", full_msg)
    except Exception:
        pass


def write_exit_code(code: int):
    """Write exit code file so watch scripts can detect process termination."""
    try:
        with open(f"/tmp/colbert_{SHARD_LABEL}.exitcode", "w") as f:
            f.write(str(code))
    except Exception:
        pass


# =============================================================================
# COLBERT MODEL
# =============================================================================

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    log.info(f"Loading {COLBERT_MODEL}...")
    _tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL)
    _model = AutoModel.from_pretrained(COLBERT_MODEL, trust_remote_code=True)
    _model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    log.info(f"Model loaded on {device}")
    return _model, _tokenizer


def encode_passages(texts: List[str], model=None, tokenizer=None) -> List[List[List[float]]]:
    """Encode a batch of passages with BF16 autocast. Returns list of multi-vectors."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    device = next(model.parameters()).device

    prefixed = [f"[unused1]{t}" for t in texts]
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        max_length=MAX_DOC_TOKENS,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**inputs)

    token_embs = outputs.last_hidden_state.float()  # back to FP32 for normalize
    token_embs = token_embs[:, :, :COLBERT_DIM]
    token_embs = F.normalize(token_embs, p=2, dim=-1)

    attention_mask = inputs["attention_mask"]
    result = []
    for i in range(len(texts)):
        mask = attention_mask[i].bool()
        vecs = token_embs[i][mask]
        result.append(vecs.cpu().tolist())
    return result


def encode_query(text: str, model=None, tokenizer=None) -> List[List[float]]:
    """Encode a single query. Returns multi-vector."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    device = next(model.parameters()).device

    # Query prefix (ColBERT: [unused0] for queries)
    prefixed = f"[unused0]{text}"
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        max_length=MAX_QUERY_TOKENS,
        truncation=True,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    token_embs = outputs.last_hidden_state[:, :, :COLBERT_DIM]  # [1, seq_len, dim]
    token_embs = F.normalize(token_embs, p=2, dim=-1)

    mask = inputs["attention_mask"][0].bool()
    return token_embs[0][mask].cpu().tolist()


# =============================================================================
# WEAVIATE CLASS
# =============================================================================

def class_exists() -> bool:
    r = requests.get(f"{WEAVIATE_URL}/v1/schema/{PILOT_CLASS}", timeout=10)
    return r.status_code == 200


def create_class():
    """Create ISMA_ColBERT_Pilot with multi-vector ColBERT config."""
    if class_exists():
        log.info(f"{PILOT_CLASS} already exists — skipping create")
        return

    schema = {
        "class": PILOT_CLASS,
        "description": (
            "ColBERT pilot: 20K enriched tiles with per-token 64-dim vectors "
            "for MaxSim late interaction retrieval. Part of ISMA Phase 6B-3."
        ),
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
            "cleanupIntervalSeconds": 60,
            "stopwords": {"preset": "en"},
            "usingBlockMaxWAND": True,
            "indexTimestamps": True,
        },
        "properties": [
            {"name": "content_hash", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
            {"name": "content", "dataType": ["text"],
             "tokenization": "word", "indexSearchable": True, "indexFilterable": False},
            {"name": "rosetta_summary", "dataType": ["text"],
             "tokenization": "word", "indexSearchable": True, "indexFilterable": False},
            {"name": "source_file", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
            {"name": "platform", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
            {"name": "dominant_motifs", "dataType": ["text[]"],
             "indexFilterable": True, "indexSearchable": True},
            {"name": "n_tokens", "dataType": ["int"],
             "indexFilterable": True, "indexRangeFilters": True},
            {"name": "source_uuid", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
        ],
        "vectorConfig": {
            "colbert": {
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "dot",
                    "multivector": {
                        "enabled": True,
                        "aggregation": "maxSim",
                    },
                },
                "vectorizer": {"none": {}},
            }
        },
    }

    r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, timeout=30)
    if r.status_code in (200, 201):
        log.info(f"Created class {PILOT_CLASS}")
    else:
        log.error(f"Failed to create class: {r.status_code} {r.text[:300]}")
        sys.exit(1)


# =============================================================================
# STREAMING INGEST — cursor → encode → write with no accumulation
# =============================================================================

def _tile_from_obj(obj: dict) -> dict | None:
    """Extract tile dict from Weaviate REST object. Returns None if text too short."""
    props = obj.get("properties", {})
    text = (props.get("rosetta_summary") or "").strip()
    if len(text) < 50:
        text = (text + " " + (props.get("content") or "")).strip()[:2000]
    if len(text) < 20:
        return None
    return {
        "uuid": obj["id"],
        "content_hash": props.get("content_hash", ""),
        "text": text,
        "source_file": props.get("source_file", ""),
        "platform": props.get("platform", ""),
        "dominant_motifs": props.get("dominant_motifs") or [],
        "rosetta_summary": props.get("rosetta_summary", ""),
        "scale": props.get("scale", ""),
    }


def run_streaming_ingest(model, tokenizer, scale: str, limit: int,
                         existing_uuids: set, cursor_start: str | None = None,
                         shard_id: int = 0, num_shards: int = 1):
    """Pipelined ingest: fetch and write overlap with GPU encode.

    Uses REST /v1/objects cursor (O(1) per page regardless of depth).
    Filters by scale in Python. ThreadPoolExecutor overlaps fetch/write with encode.

    shard_id / num_shards: hash-based sharding so multiple nodes can run
    simultaneously on the same scale without coordination or duplicates.
    Each node writes only tiles where int(uuid_hex, 16) % num_shards == shard_id.
    """
    log.info(f"Pipelined ingest: scale={scale} limit={limit} "
             f"shard={shard_id}/{num_shards} cursor_start={cursor_start or 'beginning'}")

    cursor = cursor_start
    t0 = time.monotonic()
    success = failed = found = scanned = 0

    # Stall detection: track when we last made write progress
    last_ok_time = time.monotonic()
    last_ok_count = 0

    # Thread pool for IO operations (fetch + write)
    io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="io")

    # --- Fetcher: produces batches of tiles ---
    fetch_buf: list = []
    fetch_exhausted = False

    def fetch_page(cur):
        """Fetch one page from Weaviate (runs in thread)."""
        url = f"{WEAVIATE_URL}/v1/objects?class={SOURCE_CLASS}&limit=500"
        if cur:
            url += f"&after={cur}"
        r = requests.get(url, timeout=30)
        return r.json().get("objects", [])

    def fill_encode_buffer():
        """Fill buffer to BATCH_SIZE from fetched pages."""
        nonlocal cursor, fetch_exhausted, found, scanned
        buf = []
        while len(buf) < BATCH_SIZE and not fetch_exhausted:
            if not fetch_buf:
                try:
                    objects = fetch_page(cursor)
                except Exception as e:
                    log.error(f"Fetch error: {e}")
                    fetch_exhausted = True
                    break
                if not objects:
                    fetch_exhausted = True
                    break
                cursor = objects[-1]["id"]
                scanned += len(objects)
                if len(objects) < 500:
                    fetch_exhausted = True
                fetch_buf.extend(objects)

            while fetch_buf and len(buf) < BATCH_SIZE:
                obj = fetch_buf.pop(0)
                if obj.get("properties", {}).get("scale") != scale:
                    continue
                if num_shards > 1 and int(obj["id"].replace("-", ""), 16) % num_shards != shard_id:
                    continue
                if obj["id"] in existing_uuids:
                    continue
                tile = _tile_from_obj(obj)
                if tile is None:
                    continue
                buf.append(tile)
                found += 1
                if found >= limit:
                    break
            if found >= limit:
                break
        return buf

    def write_batch_async(tiles, vectors):
        """Submit batch write to thread pool, return future."""
        return io_pool.submit(ingest_batch, tiles, vectors)

    # --- Prefetch first page (with retry) ---
    for attempt in range(6):
        try:
            first_objects = fetch_page(cursor)
            if first_objects:
                cursor = first_objects[-1]["id"]
                scanned += len(first_objects)
                if len(first_objects) < 500:
                    fetch_exhausted = True
                fetch_buf.extend(first_objects)
            break
        except Exception as e:
            wait = 15 * (2 ** attempt)
            log.warning(f"Initial fetch error (attempt {attempt+1}/6, retry in {wait}s): {e}")
            if attempt < 5:
                time.sleep(wait)
            else:
                msg = f"Initial fetch failed after 6 attempts — Weaviate unreachable. Aborting."
                send_alert(msg)
                io_pool.shutdown(wait=False)
                write_exit_code(2)
                sys.exit(2)

    # --- Main pipeline loop ---
    pending_write: Future | None = None

    while found < limit:
        # ── Circuit breaker / stall checks ──────────────────────────────────
        if _stop_event.is_set():
            send_alert(
                f"Circuit breaker triggered after {MAX_CONSEC_FAIL} consecutive "
                f"batch failures. Stopping. ok={success} found={found} scanned={scanned}"
            )
            if pending_write is not None:
                pending_write.cancel()
            io_pool.shutdown(wait=False)
            write_exit_code(2)
            sys.exit(2)

        now = time.monotonic()
        if success > last_ok_count:
            last_ok_time = now
            last_ok_count = success
        elif now - last_ok_time > STALL_SECS and found > 0:
            send_alert(
                f"STALL: no successful writes in {STALL_SECS//60}min. "
                f"ok={success} found={found} scanned={scanned} — Weaviate down?"
            )
            if pending_write is not None:
                pending_write.cancel()
            io_pool.shutdown(wait=False)
            write_exit_code(2)
            sys.exit(2)

        # 1. Fill encode buffer (may fetch more pages)
        batch_tiles = fill_encode_buffer()
        if not batch_tiles:
            break

        # 2. Encode on GPU (this is the long pole)
        try:
            vectors = encode_passages(
                [t["text"] for t in batch_tiles], model, tokenizer
            )
        except Exception as e:
            log.error(f"Encode error: {e}")
            failed += len(batch_tiles)
            continue

        # 3. Wait for previous write to complete (if any)
        if pending_write is not None:
            try:
                ok, bad = pending_write.result(timeout=360)
                success += ok
                failed += bad
            except Exception as e:
                # Future itself threw (timeout, thread crash) — count as dropped
                failed += BATCH_SIZE
                _consecutive_failures[0] += 1
                send_alert(
                    f"Write future exception (consec={_consecutive_failures[0]}): {e}"
                )
                if _consecutive_failures[0] >= MAX_CONSEC_FAIL:
                    _stop_event.set()

        # 4. Submit this batch's write asynchronously
        pending_write = write_batch_async(batch_tiles, vectors)

        elapsed = time.monotonic() - t0
        rate = found / elapsed if elapsed > 0 else 0
        log.info(f"Progress: {found} | {rate:.1f} tiles/s | "
                 f"scanned={scanned} ok={success} failed={failed}")

    # Drain final write
    if pending_write is not None and not _stop_event.is_set():
        try:
            ok, bad = pending_write.result(timeout=360)
            success += ok
            failed += bad
        except Exception as e:
            failed += BATCH_SIZE
            send_alert(f"Final write error: {e}")

    io_pool.shutdown(wait=True)
    elapsed = time.monotonic() - t0
    log.info(f"Done in {elapsed:.1f}s — {success} ingested, {failed} failed | "
             f"scanned {scanned} objects to find {found} {scale} tiles")

    if failed > 0 and success == 0:
        send_alert(
            f"ZERO successful writes. {failed} tiles dropped. "
            f"Weaviate was unreachable the entire run."
        )


# =============================================================================
# CHECKPOINT / DEDUP
# =============================================================================

def get_redis():
    try:
        import redis
        redis_host = os.environ.get("REDIS_HOST", "192.168.100.10")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()
        return r
    except Exception:
        log.warning("Redis unavailable — checkpointing disabled")
        return None


def get_existing_source_uuids() -> set:
    """Return set of source_uuids already ingested (dedup by source tile UUID, not content_hash).

    Uses cursor-based REST pagination (O(1) per page) — NOT offset pagination
    which is O(N) per page and causes Weaviate timeouts on 193K+ records.
    """
    all_uuids = set()
    cursor = None
    page_size = 2000
    while True:
        url = f"{WEAVIATE_URL}/v1/objects?class={PILOT_CLASS}&limit={page_size}"
        if cursor:
            url += f"&after={cursor}"
        fetched = None
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=60)
                fetched = r.json().get("objects", []) or []
                break
            except Exception as e:
                wait = 20 * (2 ** attempt)
                log.warning(f"Could not fetch existing UUIDs at cursor={cursor} (attempt {attempt+1}/4, retry in {wait}s): {e}")
                if attempt < 3:
                    time.sleep(wait)
        if fetched is None:
            log.warning(f"UUID fetch failed — using partial dedup set ({len(all_uuids)} uuids)")
            break
        if not fetched:
            break
        for obj in fetched:
            uuid = obj.get("properties", {}).get("source_uuid")
            if uuid:
                all_uuids.add(uuid)
        if len(fetched) < page_size:
            break
        cursor = fetched[-1]["id"]
        if len(all_uuids) % 20000 == 0 and len(all_uuids) > 0:
            log.info(f"  Dedup scan: {len(all_uuids)} uuids loaded...")
    log.info(f"Found {len(all_uuids)} already-ingested source tiles")
    return all_uuids


def ingest_batch(tiles: list, vectors_list: List[List[List[float]]]) -> tuple[int, int]:
    """Batch-write tiles to Weaviate using /v1/batch/objects. Returns (success, failed)."""
    objects = []
    for tile, vectors in zip(tiles, vectors_list):
        if len(vectors) == 0:
            continue
        objects.append({
            "class": PILOT_CLASS,
            "properties": {
                "content_hash": tile["content_hash"],
                "content": (tile.get("rosetta_summary") or "")[:2000],
                "rosetta_summary": (tile.get("rosetta_summary") or ""),
                "source_file": tile["source_file"],
                "platform": tile["platform"],
                "dominant_motifs": tile["dominant_motifs"][:10],
                "n_tokens": len(vectors),
                "source_uuid": tile["uuid"],
            },
            "vectors": {"colbert": vectors},
        })
    if not objects:
        return 0, len(tiles)
    last_exc = None
    for attempt in range(5):
        try:
            r = requests.post(
                f"{WEAVIATE_URL}/v1/batch/objects",
                json={"objects": objects},
                timeout=300,
            )
            if r.status_code not in (200, 201):
                log.error(f"Batch write failed: {r.status_code} {r.text[:200]}")
                return 0, len(objects)
            results = r.json()
            if isinstance(results, list):
                ok = sum(1 for res in results if res.get("result", {}).get("status") == "SUCCESS")
                bad = len(results) - ok
            else:
                ok = len(objects)
                bad = 0
            # Success — reset consecutive failure counter
            _consecutive_failures[0] = 0
            return ok, bad
        except Exception as e:
            last_exc = e
            wait = 10 * (2 ** attempt)
            log.warning(f"Batch write error (attempt {attempt+1}/5, retry in {wait}s): {e}")
            if attempt < 4:
                time.sleep(wait)

    # All 5 retries exhausted — this is a hard failure
    _consecutive_failures[0] += 1
    msg = (
        f"Batch DROPPED after 5 retries (consec={_consecutive_failures[0]}). "
        f"{len(objects)} tiles lost. Last error: {last_exc}"
    )
    send_alert(msg)
    if _consecutive_failures[0] >= MAX_CONSEC_FAIL:
        # Signal main loop to stop — don't call sys.exit() from a thread
        _stop_event.set()

    return 0, len(objects)


# =============================================================================
# STATS
# =============================================================================

def show_stats():
    r = requests.post(
        f"{WEAVIATE_URL}/v1/graphql",
        json={"query": f"{{ Aggregate {{ {PILOT_CLASS} {{ meta {{ count }} }} }} }}"},
        timeout=15,
    )
    count = (
        r.json().get("data", {}).get("Aggregate", {})
        .get(PILOT_CLASS, [{}])[0]
        .get("meta", {}).get("count", "?")
    )
    print(f"{PILOT_CLASS}: {count} tiles ingested")
    print(f"Class exists: {class_exists()}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-class", action="store_true",
                        help="Create ISMA_ColBERT_Pilot Weaviate class")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest tiles into the class")
    parser.add_argument("--limit", type=int, default=20000,
                        help="Number of tiles to ingest (default: 20000)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index (0-based). With --num-shards, each node writes a distinct subset.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards. Splits load across nodes without coordination.")
    parser.add_argument("--scale", default="search_512",
                        choices=["rosetta", "search_512", "context_2048", "full_4096"],
                        help="Source scale to ingest (default: search_512 for full passage coverage)")
    parser.add_argument("--stats", action="store_true",
                        help="Show current ingest stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if not args.create_class and not args.ingest:
        parser.print_help()
        sys.exit(1)

    if args.create_class:
        create_class()

    if not args.ingest:
        return

    # Load ColBERT model
    model, tokenizer = load_model()

    # UUID dedup (fast for empty class, needed for safe resume)
    existing_uuids = get_existing_source_uuids()

    # Stream: cursor → encode → write, no accumulation
    run_streaming_ingest(model, tokenizer, scale=args.scale,
                         limit=args.limit, existing_uuids=existing_uuids,
                         shard_id=args.shard_id, num_shards=args.num_shards)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
        write_exit_code(0)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        write_exit_code(exit_code)
        sys.exit(exit_code)
    except Exception as e:
        exit_code = 1
        send_alert(f"CRASH: unhandled exception: {e}")
        write_exit_code(exit_code)
        log.exception("Unhandled exception in main")
        sys.exit(exit_code)
