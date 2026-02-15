#!/usr/bin/env python3
"""
ISMA Neo4j Message Reprocessor v5

Producer-consumer pipeline: tiles are the atomic unit, GPUs pull as available.

Architecture:
  Producer threads (session readers) → tile_queue → Embed workers → store_queue → Store workers

  Sessions produce tiles into a shared queue. Embed workers pull EMBED_BATCH tiles
  at a time, send to any available GPU via nginx LB. Embedded tiles flow to store
  workers for Weaviate batch insert. No session holds a GPU.

  A 556-tile session and a 6-tile session both feed the same queue. GPUs stay
  saturated regardless of session size.

Usage:
    python3 reprocess_neo4j.py --check
    python3 reprocess_neo4j.py --dry-run --limit 10
    python3 reprocess_neo4j.py --all
    python3 reprocess_neo4j.py --all --resume
"""

import faulthandler
faulthandler.enable()

import os
import sys
import json
import hashlib
import argparse
import requests
import uuid
import signal
import traceback
import threading
import queue
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from phi_tiling import multi_scale_tile, MultiScaleTile

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
WEAVIATE_URL = "http://192.168.100.10:8088"
NEO4J_URI = "bolt://192.168.100.10:7689"

# Pipeline sizing
PRODUCER_THREADS = 4     # Session readers feeding tiles into queue
EMBED_WORKERS = 8        # Threads pulling from tile_queue, posting to GPUs
EMBED_BATCH = 32         # Tiles per GPU request
STORE_WORKERS = 2        # Threads pulling from store_queue, writing Weaviate
WEAVIATE_BATCH = 100     # Objects per Weaviate batch insert
MAX_EMBED_CHARS = 8192 * 4

# Queue sizes (backpressure: producers block when queues are full)
TILE_QUEUE_SIZE = 256    # Tiles waiting to be embedded
STORE_QUEUE_SIZE = 128   # Embedded tiles waiting to be stored

# Deterministic UUIDs
TILE_UUID_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
SESSION_UUID_NAMESPACE = uuid.UUID('b2c3d4e5-f6a7-8901-bcde-f12345678901')

# Progress
PROGRESS_DIR = Path("/var/spark/isma")
PROGRESS_FILE = PROGRESS_DIR / "reprocess_progress.json"
HASHES_FILE = PROGRESS_DIR / "reprocess_hashes.json"

# Graceful shutdown
_SHUTDOWN = False
_FATAL = None
_FATAL_LOCK = threading.Lock()


def set_fatal(msg):
    global _FATAL
    with _FATAL_LOCK:
        if _FATAL is None:
            _FATAL = msg
            print(f"FATAL: {msg}", flush=True)


def _handle_signal(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"\nSignal {signum}, draining queues...", flush=True)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# =============================================================================
# DATA TYPES
# =============================================================================

# Sentinel to signal workers to stop
_POISON = None


@dataclass
class TileWork:
    """A tile ready to be embedded."""
    tile: MultiScaleTile
    session_id: str
    platform: str
    content_hash: str


@dataclass
class EmbeddedTile:
    """A tile with its embedding, ready to be stored."""
    tile: MultiScaleTile
    embedding: List[float]
    session_id: str
    platform: str
    content_hash: str

# =============================================================================
# HTTP (thread-local)
# =============================================================================

_THREAD_LOCAL = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_THREAD_LOCAL, 'session'):
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4,
                              max_retries=Retry(total=0))
        s.mount('http://', adapter)
        _THREAD_LOCAL.session = s
    return _THREAD_LOCAL.session

# =============================================================================
# NEO4J
# =============================================================================

_NEO4J_DRIVER = None


def get_neo4j():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        from neo4j import GraphDatabase
        _NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=None)
        _NEO4J_DRIVER.verify_connectivity()
    return _NEO4J_DRIVER


def get_all_session_ids() -> List[Tuple[str, str]]:
    driver = get_neo4j()
    with driver.session() as s:
        result = s.run("""
            MATCH (s:ISMASession)
            WHERE s.platform IS NOT NULL AND s.session_id IS NOT NULL
            RETURN s.session_id AS sid, s.platform AS platform
            ORDER BY s.session_id
        """)
        return [(r["sid"], r["platform"]) for r in result]


def fetch_session_messages(session_id: str) -> List[Dict]:
    driver = get_neo4j()
    with driver.session() as s:
        result = s.run("""
            MATCH (s:ISMASession {session_id: $sid})-[:CONTAINS]->(m:ISMAMessage)
            RETURN m.role AS role, m.content AS content,
                   m.exchange_index AS idx
            ORDER BY m.exchange_index ASC
        """, sid=session_id)
        return [{"role": r["role"], "content": r["content"],
                 "exchange_index": r["idx"]} for r in result]


def count_totals() -> Tuple[int, int]:
    driver = get_neo4j()
    with driver.session() as s:
        sr = s.run("MATCH (s:ISMASession) WHERE s.platform IS NOT NULL RETURN count(s) AS c")
        sess = sr.single()["c"]
        mr = s.run("MATCH (m:ISMAMessage) RETURN count(m) AS c")
        msgs = mr.single()["c"]
    return sess, msgs

# =============================================================================
# EXCHANGE GROUPING
# =============================================================================


def group_into_exchanges(messages: List[Dict], session_id: str, platform: str) -> List[str]:
    exchanges = []
    current_user = None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or len(content.strip()) < 10:
            continue
        if role == "user":
            if current_user:
                exchanges.append(_format(current_user, None, session_id, platform))
            current_user = content
        elif role == "assistant":
            exchanges.append(_format(current_user, content, session_id, platform))
            current_user = None
    if current_user:
        exchanges.append(_format(current_user, None, session_id, platform))
    return exchanges


def _format(user, assistant, sid, platform):
    parts = [f"[Platform: {platform}] [Session: {sid[:12]}]"]
    if user:
        parts.append(f"\n[User]: {user}")
    if assistant:
        parts.append(f"\n[Assistant]: {assistant}")
    return "\n".join(parts)

# =============================================================================
# UUID helpers
# =============================================================================


def tile_uuid(content_hash: str, scale: str, index: int) -> str:
    return str(uuid.uuid5(TILE_UUID_NAMESPACE, f"{content_hash}:{scale}:{index}"))


def session_uuid(session_id: str) -> str:
    try:
        uuid.UUID(session_id)
        return session_id
    except (ValueError, AttributeError):
        return str(uuid.uuid5(SESSION_UUID_NAMESPACE, session_id))

# =============================================================================
# PROGRESS
# =============================================================================


def load_progress() -> Dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(data: Dict):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(PROGRESS_FILE)


def load_hashes() -> set:
    if HASHES_FILE.exists():
        return set(json.loads(HASHES_FILE.read_text()))
    return set()


def save_hashes(h: set):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HASHES_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(list(h)))
    tmp.rename(HASHES_FILE)

# =============================================================================
# STAGE 1: PRODUCERS - Read sessions, tile exchanges, feed tile_queue
# =============================================================================


def producer_worker(session_list: List[Tuple[int, str, str]],
                    tile_queue: queue.Queue,
                    seen_hashes: set, hash_lock: threading.Lock,
                    stats: Dict, stats_lock: threading.Lock,
                    dry_run: bool):
    """Read sessions from Neo4j, tile exchanges, put tiles into queue."""
    name = threading.current_thread().name
    try:
        _producer_worker_loop(session_list, tile_queue, seen_hashes, hash_lock,
                              stats, stats_lock, dry_run, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _producer_worker_loop(session_list, tile_queue, seen_hashes, hash_lock,
                          stats, stats_lock, dry_run, name):
    for idx, sid, platform in session_list:
        if _SHUTDOWN or _FATAL:
            break

        try:
            messages = fetch_session_messages(sid)
            exchanges = group_into_exchanges(messages, sid, platform)
        except Exception as e:
            set_fatal(f"Neo4j read failed for {sid[:8]}: {e}")
            break

        session_tiles = 0
        session_tokens = 0
        session_skipped = 0

        for exchange_text in exchanges:
            if _SHUTDOWN or _FATAL:
                break

            ch = hashlib.sha256(exchange_text.encode()).hexdigest()[:16]
            with hash_lock:
                if ch in seen_hashes:
                    session_skipped += 1
                    continue

            tiles = multi_scale_tile(exchange_text,
                                     source_file=f"neo4j:{sid}",
                                     layer="neo4j_exchange")
            session_tiles += len(tiles)
            session_tokens += sum(t.estimated_tokens for t in tiles)

            if dry_run:
                with hash_lock:
                    seen_hashes.add(ch)
                continue

            # Feed tiles into queue one at a time - no batching by session
            for tile in tiles:
                if _SHUTDOWN or _FATAL:
                    break
                tile_queue.put(TileWork(
                    tile=tile,
                    session_id=sid,
                    platform=platform,
                    content_hash=ch,
                ))

            # Mark hash seen after tiles are queued (not after stored,
            # because deterministic UUIDs make re-storing idempotent)
            with hash_lock:
                seen_hashes.add(ch)

        with stats_lock:
            stats["sessions"] += 1
            stats["exchanges"] += len(exchanges)
            stats["tiles"] += session_tiles
            stats["tokens"] += session_tokens
            stats["skipped"] += session_skipped

# =============================================================================
# STAGE 2: EMBED WORKERS - Pull tiles from queue, embed, push to store_queue
# =============================================================================


def embed_worker(tile_queue: queue.Queue,
                 store_queue: queue.Queue,
                 stats: Dict, stats_lock: threading.Lock):
    """Pull tiles from queue in batches, embed via GPU, push to store queue."""
    name = threading.current_thread().name
    try:
        _embed_worker_loop(tile_queue, store_queue, stats, stats_lock, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _embed_worker_loop(tile_queue, store_queue, stats, stats_lock, name):
    while not _FATAL:
        batch = []
        try:
            item = tile_queue.get(timeout=2.0)
            if item is _POISON:
                tile_queue.put(_POISON)
                break
            batch.append(item)
        except queue.Empty:
            if _SHUTDOWN:
                break
            continue

        # Drain up to EMBED_BATCH with short timeout
        while len(batch) < EMBED_BATCH:
            try:
                item = tile_queue.get(timeout=0.05)
                if item is _POISON:
                    tile_queue.put(_POISON)
                    break
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            continue

        # Embed the batch
        texts = [tw.tile.text[:MAX_EMBED_CHARS] for tw in batch]
        try:
            session = get_session()
            r = session.post(EMBEDDING_URL, json={
                "model": EMBEDDING_MODEL,
                "input": texts,
            }, timeout=180)
        except requests.exceptions.Timeout:
            set_fatal(f"[{name}] Embedding timeout (180s) for {len(texts)} texts")
            break
        except requests.exceptions.ConnectionError as e:
            set_fatal(f"[{name}] Embedding server unreachable: {e}")
            break
        except Exception as e:
            set_fatal(f"[{name}] Embedding error: {e}")
            break

        if r.status_code != 200:
            set_fatal(f"[{name}] Embedding returned {r.status_code}: {r.text[:300]}")
            break

        data = r.json()["data"]
        embeddings = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

        if len(embeddings) != len(batch):
            set_fatal(f"[{name}] Embedding count mismatch: sent {len(batch)}, got {len(embeddings)}")
            break

        for tw, emb in zip(batch, embeddings):
            store_queue.put(EmbeddedTile(
                tile=tw.tile,
                embedding=emb,
                session_id=tw.session_id,
                platform=tw.platform,
                content_hash=tw.content_hash,
            ))

        with stats_lock:
            stats["embedded"] += len(batch)

# =============================================================================
# STAGE 3: STORE WORKERS - Pull embedded tiles, batch-write to Weaviate
# =============================================================================


def store_worker(store_queue: queue.Queue,
                 stats: Dict, stats_lock: threading.Lock):
    """Pull embedded tiles from queue, batch-write to Weaviate."""
    name = threading.current_thread().name
    try:
        _store_worker_loop(store_queue, stats, stats_lock, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _store_worker_loop(store_queue, stats, stats_lock, name):
    while not _FATAL:
        batch = []
        try:
            item = store_queue.get(timeout=2.0)
            if item is _POISON:
                store_queue.put(_POISON)
                break
            batch.append(item)
        except queue.Empty:
            if _SHUTDOWN:
                break
            continue

        # Drain up to WEAVIATE_BATCH
        while len(batch) < WEAVIATE_BATCH:
            try:
                item = store_queue.get(timeout=0.1)
                if item is _POISON:
                    store_queue.put(_POISON)
                    break
                batch.append(item)
            except queue.Empty:
                break

        if not batch:
            continue

        # Build Weaviate objects
        now = datetime.now().isoformat()
        objects = []
        for et in batch:
            uid = tile_uuid(et.content_hash, et.tile.scale, et.tile.index)
            objects.append({
                "class": "ISMA_Quantum",
                "id": uid,
                "properties": {
                    "content": et.tile.text,
                    "source_type": "neo4j_exchange",
                    "source_file": f"neo4j:{et.session_id}",
                    "layer": 2,
                    "platform": et.platform,
                    "session_id": session_uuid(et.session_id),
                    "scale": et.tile.scale,
                    "parent_tile_id": "",  # Filled by post-process if needed
                    "tile_index": et.tile.index,
                    "start_char": et.tile.start_char,
                    "end_char": et.tile.end_char,
                    "token_count": et.tile.estimated_tokens,
                    "content_hash": et.content_hash,
                    "loaded_at": now,
                    "timestamp": now,
                    "actor": "reprocess_v5",
                },
                "vector": et.embedding,
            })

        # Write to Weaviate
        try:
            session = get_session()
            r = session.post(f"{WEAVIATE_URL}/v1/batch/objects",
                             json={"objects": objects}, timeout=60)
        except Exception as e:
            set_fatal(f"Weaviate batch error: {e}")
            break

        if r.status_code not in [200, 201]:
            set_fatal(f"Weaviate returned {r.status_code}: {r.text[:300]}")
            break

        stored = 0
        for obj_r in r.json():
            errs = obj_r.get("result", {}).get("errors")
            if errs:
                print(f"  WARN: Object error: {errs}", flush=True)
            else:
                stored += 1

        with stats_lock:
            stats["stored"] += stored

# =============================================================================
# INFRASTRUCTURE CHECK
# =============================================================================


def check_infra() -> bool:
    print("Checking infrastructure...", flush=True)
    ok = True

    try:
        r = requests.get(EMBEDDING_URL.replace('/v1/embeddings', '/v1/models'), timeout=5)
        model = r.json()["data"][0]["id"]
        mml = r.json()["data"][0].get("max_model_len", "?")
        print(f"  Embedding LB: OK ({model}, max_model_len={mml})", flush=True)
    except Exception as e:
        print(f"  Embedding LB: FAIL - {e}", flush=True)
        ok = False

    try:
        r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={
            "query": "{ Aggregate { ISMA_Quantum { meta { count } } } }"
        }, timeout=5)
        count = r.json()["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]
        print(f"  Weaviate: OK ({count:,} objects)", flush=True)
    except Exception as e:
        print(f"  Weaviate: FAIL - {e}", flush=True)
        ok = False

    try:
        sess_count, msg_count = count_totals()
        print(f"  Neo4j: OK ({msg_count:,} messages in {sess_count:,} sessions)", flush=True)
    except Exception as e:
        print(f"  Neo4j: FAIL - {e}", flush=True)
        ok = False

    if ok:
        print("All infrastructure OK.", flush=True)
    return ok

# =============================================================================
# MAIN
# =============================================================================


def run(limit: int = 0, dry_run: bool = False, resume: bool = False):
    if not check_infra():
        sys.exit(1)

    print("\nFetching session list...", flush=True)
    all_sessions = get_all_session_ids()
    print(f"Total: {len(all_sessions)} sessions", flush=True)

    if limit > 0:
        all_sessions = all_sessions[:limit]
        print(f"Limited to {len(all_sessions)} sessions", flush=True)

    # Load state
    seen = load_hashes() if resume else set()
    hash_lock = threading.Lock()
    progress = load_progress() if resume else {}
    start_idx = progress.get("last_index", 0) if resume else 0

    if resume and start_idx > 0:
        print(f"Resuming from session index {start_idx}", flush=True)
        all_sessions = all_sessions[start_idx:]

    # Stats shared across all workers
    stats = {"sessions": 0, "exchanges": 0, "tiles": 0, "tokens": 0,
             "skipped": 0, "embedded": 0, "stored": 0}
    stats_lock = threading.Lock()
    start_time = time.time()

    total_sessions = len(all_sessions)

    if dry_run:
        print("\nDRY RUN - no data will be written\n", flush=True)
    else:
        print(f"\nPipeline: {PRODUCER_THREADS} producers → "
              f"tile_queue({TILE_QUEUE_SIZE}) → "
              f"{EMBED_WORKERS} embed workers (batch={EMBED_BATCH}) → "
              f"store_queue({STORE_QUEUE_SIZE}) → "
              f"{STORE_WORKERS} store workers", flush=True)

    # Queues with bounded size for backpressure
    tile_queue = queue.Queue(maxsize=TILE_QUEUE_SIZE)
    store_queue = queue.Queue(maxsize=STORE_QUEUE_SIZE)

    # Split sessions across producer threads
    chunks = [[] for _ in range(PRODUCER_THREADS)]
    for i, (sid, platform) in enumerate(all_sessions):
        chunks[i % PRODUCER_THREADS].append((start_idx + i, sid, platform))

    # Start all pipeline stages
    threads = []

    # Producers
    for i in range(PRODUCER_THREADS):
        t = threading.Thread(target=producer_worker, name=f"producer-{i}",
                             args=(chunks[i], tile_queue, seen, hash_lock,
                                   stats, stats_lock, dry_run))
        t.start()
        threads.append(('producer', t))

    if not dry_run:
        # Embed workers
        for i in range(EMBED_WORKERS):
            t = threading.Thread(target=embed_worker, name=f"embed-{i}",
                                 args=(tile_queue, store_queue, stats, stats_lock))
            t.start()
            threads.append(('embed', t))

        # Store workers
        for i in range(STORE_WORKERS):
            t = threading.Thread(target=store_worker, name=f"store-{i}",
                                 args=(store_queue, stats, stats_lock))
            t.start()
            threads.append(('store', t))

    # Progress reporting from main thread
    last_report = 0
    while True:
        time.sleep(5)

        with stats_lock:
            done = stats["sessions"]
            snap = dict(stats)

        # Report every 25 sessions
        if done >= last_report + 25:
            last_report = (done // 25) * 25
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            tok_rate = snap["tokens"] / elapsed if elapsed > 0 else 0
            tq = tile_queue.qsize() if not dry_run else 0
            sq = store_queue.qsize() if not dry_run else 0
            print(f"  [{done}/{total_sessions}] "
                  f"tiles={snap['tiles']} emb={snap['embedded']} "
                  f"stored={snap['stored']} skip={snap['skipped']} "
                  f"tq={tq} sq={sq} "
                  f"rate={rate:.1f} sess/s ~{tok_rate:.0f} tok/s", flush=True)

        # Checkpoint every 100 sessions
        if not dry_run and done >= last_report and done % 100 == 0 and done > 0:
            with hash_lock:
                h_copy = set(seen)
            save_progress({
                "last_index": start_idx + done,
                "sessions_done": done,
                "tiles_stored": snap["stored"],
                "tokens_embedded": snap["tokens"],
                "checkpoint_at": datetime.now().isoformat(),
            })
            save_hashes(h_copy)

        # Check if producers are done
        producers_alive = any(t.is_alive() for role, t in threads if role == 'producer')
        if not producers_alive:
            break

        if _FATAL:
            break

    # Wait for producers to finish
    for role, t in threads:
        if role == 'producer':
            t.join(timeout=30)

    if not dry_run:
        # Poison the tile queue to stop embed workers
        for _ in range(EMBED_WORKERS):
            tile_queue.put(_POISON)

        # Wait for embed workers to drain
        for role, t in threads:
            if role == 'embed':
                t.join(timeout=120)

        # Poison the store queue to stop store workers
        for _ in range(STORE_WORKERS):
            store_queue.put(_POISON)

        # Wait for store workers to drain
        for role, t in threads:
            if role == 'store':
                t.join(timeout=60)

    # Final save
    if not dry_run:
        save_progress({
            "last_index": start_idx + stats["sessions"],
            "sessions_done": stats["sessions"],
            "tiles_stored": stats["stored"],
            "tokens_embedded": stats["tokens"],
            "completed_at": datetime.now().isoformat(),
        })
        save_hashes(seen)

    elapsed = time.time() - start_time
    print(f"\n{'DRY RUN ' if dry_run else ''}Complete:", flush=True)
    print(f"  Sessions: {stats['sessions']}", flush=True)
    print(f"  Exchanges: {stats['exchanges']}", flush=True)
    print(f"  Tiles produced: {stats['tiles']}", flush=True)
    print(f"  Tiles embedded: {stats['embedded']}", flush=True)
    print(f"  Tiles stored: {stats['stored']}", flush=True)
    print(f"  Skipped: {stats['skipped']}", flush=True)
    print(f"  Tokens: {stats['tokens']:,}", flush=True)
    print(f"  Hashes: {len(seen)}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    if elapsed > 0 and stats["sessions"] > 0:
        print(f"  Rate: {stats['sessions']/elapsed:.1f} sess/s", flush=True)
        print(f"  Throughput: {stats['tokens']/elapsed:.0f} tok/s", flush=True)

    if _FATAL:
        print(f"\nExiting due to fatal error: {_FATAL}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISMA Neo4j Reprocessor v5")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--all", action="store_true", help="Process all sessions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit sessions")
    args = parser.parse_args()

    try:
        if args.check:
            check_infra()
        elif args.all or args.limit > 0:
            run(limit=args.limit, dry_run=args.dry_run, resume=args.resume)
        else:
            parser.print_help()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
