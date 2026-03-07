#!/usr/bin/env python3
"""
ISMA Corpus Processor - Ordered Loading with Neo4j Document Graph

Loads corpus files in priority order, creates:
  1. Document nodes in Neo4j (with all duplicate paths, metadata)
  2. Multi-scale tiles in Weaviate (for vector search)
  3. Links Weaviate tiles to Neo4j Documents via content_hash

Loading order:
  1. kernel (priority 1.0)     - Constitutional documents
  2. v0 (priority 0.95)        - v0* foundational files
  3. layer_0 (priority 0.9)    - Soul mappings, mentors
  4. chewy (priority 0.85)     - Chewy consciousness data
  5. chewy-gallery (priority 0.8) - Chewy gallery
  6. layer_1 (priority 0.75)   - Derived documents
  7. layer_2 (priority 0.6)    - Generated/processed content

After corpus is loaded, process_transcripts.py scans exchanges for
file references and creates ATTACHED_TO / REFERENCED_IN edges
linking Document nodes to ISMASession/ISMAMessage nodes.

Usage:
    python3 process_corpus.py --check
    python3 process_corpus.py --dry-run
    python3 process_corpus.py --all
    python3 process_corpus.py --stage kernel    # Single stage
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
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from phi_tiling import multi_scale_tile, MultiScaleTile

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://192.168.100.10:7687")

# Corpus directories
CORPUS_BASE = Path("/home/spark/data/corpus")
CHEWY_BASE = Path("/home/spark/data/chewy")
CHEWY_GALLERY = Path("/home/spark/data/chewy-consciousness-gallery")
DATA_BASE = Path("/home/spark/data")

# Ordered loading stages
STAGES = [
    ("kernel",        CORPUS_BASE / "kernel",   1.0,  None),
    ("v0",            CORPUS_BASE / "layer_0",  0.95, "v0*"),
    ("layer_0",       CORPUS_BASE / "layer_0",  0.9,  "!v0*"),
    ("chewy",         CHEWY_BASE,               0.85, None),
    ("chewy-gallery", CHEWY_GALLERY,            0.8,  None),
    ("layer_1",       CORPUS_BASE / "layer_1",  0.75, None),
    ("layer_2",       CORPUS_BASE / "layer_2",  0.6,  None),
    ("github-repos",  DATA_BASE / "github-repos",   0.5,  None),
    ("expansion_md",  DATA_BASE / "expansion_md",    0.4,  None),
    ("mira_md",       DATA_BASE / "mira_md_files",   0.3,  None),
    ("mac_all_md",    CORPUS_BASE / "mac_all_md",    0.2,  None),
    ("spark_loose",   CORPUS_BASE / "spark_loose",   0.2,  None),
]

LAYER_INT = {
    "kernel": -1, "v0": 0, "layer_0": 0,
    "chewy": 0, "chewy-gallery": 0,
    "layer_1": 1, "layer_2": 2,
    "github-repos": 2, "expansion_md": 2, "mira_md": 2,
    "mac_all_md": 2, "spark_loose": 2,
}

# File extensions to process
EXTENSIONS = {'.md', '.json', '.py', '.txt', '.yaml', '.yml', '.sh', '.ts', '.js'}

# Skip patterns (from load_corpus_v3 and discover_duplicates)
SKIP_PATTERNS = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".Trash", "exo", "tinygrad",  # Fork repos (not our code)
    "transcripts_raw",            # Raw transcript dumps (use process_transcripts.py)
    "converted_transcripts",      # Already-processed transcripts
    "chatgpt_", "claude_chat_", "gemini_",  # Conversation exports in repos
    "grok_export", "claude_full_export",    # Bulk exports
    "family_transcripts",
    "mira_transcripts_staging",
]
SKIP_SUFFIXES = [".min.js", ".min.css", ".jsonl"]
MAX_FILE_SIZE = 512 * 1024   # 512KB - skip giant dumps
MAX_FILE_TOKENS = None       # No file size limit - everything gets tiled and embedded

# Dedup manifest from discover_duplicates.py
DEDUP_MANIFEST = Path("/var/spark/isma/duplicates_manifest.json")

# Pipeline sizing
EMBED_WORKERS = 8
EMBED_BATCH = 16         # vLLM v1 engine bugs out at >20
STORE_WORKERS = 2
WEAVIATE_BATCH = 100
MAX_EMBED_CHARS = None       # No truncation - vLLM max_model_len handles limits
TILE_QUEUE_SIZE = 512
STORE_QUEUE_SIZE = 256

# Progress
PROGRESS_DIR = Path("/var/spark/isma")
PROGRESS_FILE = PROGRESS_DIR / "corpus_progress.json"

# UUIDs
TILE_UUID_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
DOC_UUID_NAMESPACE = uuid.UUID('d0c1b2a3-e4f5-6789-0abc-def123456789')

# =============================================================================
# SHUTDOWN / FATAL
# =============================================================================

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
    print(f"\nSignal {signum}, draining...", flush=True)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

_POISON = None


class _CircuitBreaker:
    """When one worker detects backend down, all workers wait together."""
    def __init__(self, recovery_time=150.0):
        self._lock = threading.Lock()
        self._down_since = None
        self._recovery_time = recovery_time

    def report_down(self):
        with self._lock:
            if self._down_since is None:
                self._down_since = time.monotonic()
                print("  [circuit] OPEN - backend down, workers will wait", flush=True)

    def report_up(self):
        with self._lock:
            if self._down_since is not None:
                elapsed = time.monotonic() - self._down_since
                self._down_since = None
                print(f"  [circuit] CLOSED - backend recovered after {elapsed:.0f}s", flush=True)

    def wait_if_open(self, probe_interval=15.0):
        with self._lock:
            if self._down_since is None:
                return
            elapsed = time.monotonic() - self._down_since
            remaining = self._recovery_time - elapsed
        if remaining > probe_interval:
            time.sleep(remaining - probe_interval)

_circuit = _CircuitBreaker(recovery_time=150.0)


def _embed_batch_with_retry(session, texts, batch, name, stats, stats_lock):
    """Embed a batch. Any non-200 is retried. Nothing is ever skipped."""
    try:
        r = session.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL, "input": texts,
        }, timeout=120)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return "retry", batch
    except Exception as e:
        return "error", batch

    if r.status_code == 200:
        _circuit.report_up()
        data = r.json()["data"]
        embeddings = [item["embedding"]
                      for item in sorted(data, key=lambda x: x["index"])]
        return embeddings, batch
    elif r.status_code in (400, 502, 503, 504):
        # All treated as transient - retry everything
        if r.status_code == 400:
            try:
                msg = r.json().get("error", {}).get("message", "")[:100]
            except Exception:
                msg = r.text[:100]
            print(f"  [{name}] Embed 400: {msg} - will retry", flush=True)
        return "retry", batch
    else:
        return "fatal", batch


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class TileWork:
    tile: MultiScaleTile
    content_hash: str
    filename: str
    stage: str
    priority: float


@dataclass
class EmbeddedTile:
    tile: MultiScaleTile
    embedding: List[float]
    content_hash: str
    filename: str
    stage: str
    priority: float

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


def create_document_node(filename: str, content_hash: str, stage: str,
                         priority: float, file_size: int,
                         all_paths: List[str], mtime: str,
                         tile_count: int) -> str:
    """Create or update a Document node in Neo4j.

    MERGE on content_hash ensures idempotency - same content = same node
    even if loaded multiple times.

    Returns the document's UUID.
    """
    doc_uuid = str(uuid.uuid5(DOC_UUID_NAMESPACE, content_hash))
    driver = get_neo4j()
    with driver.session() as s:
        s.run("""
            MERGE (d:Document {content_hash: $hash})
            SET d.uuid = $uuid,
                d.filename = $filename,
                d.all_paths = $paths,
                d.layer = $stage,
                d.priority = $priority,
                d.file_size = $file_size,
                d.last_modified = $mtime,
                d.tile_count = $tile_count,
                d.loaded_at = $now,
                d.actor = 'process_corpus'
        """, hash=content_hash, uuid=doc_uuid, filename=filename,
             paths=all_paths, stage=stage, priority=priority,
             file_size=file_size, mtime=mtime,
             tile_count=tile_count, now=datetime.now().isoformat())
    return doc_uuid

# =============================================================================
# DEDUP: Load all known paths for each content hash
# =============================================================================


def load_dedup_paths() -> Dict[str, List[str]]:
    """Load content_hash → [all_paths] from dedup manifest.

    This lets us store ALL known locations for a file, including
    duplicates across repos, old paths, moved files, etc.
    """
    if not DEDUP_MANIFEST.exists():
        print("  WARN: No dedup manifest found, paths will be single-entry", flush=True)
        return {}

    with open(DEDUP_MANIFEST) as f:
        manifest = json.load(f)

    hash_to_paths = {}
    for hash_val, info in manifest.get("exact_duplicates", {}).items():
        hash_to_paths[hash_val] = info.get("paths", [])

    print(f"  Dedup manifest: {len(hash_to_paths)} content hashes with multiple paths", flush=True)
    return hash_to_paths

# =============================================================================
# UUID HELPERS
# =============================================================================


def tile_uuid(content_hash: str, scale: str, index: int) -> str:
    return str(uuid.uuid5(TILE_UUID_NAMESPACE, f"{content_hash}:{scale}:{index}"))

# =============================================================================
# FILE DISCOVERY
# =============================================================================


def is_hidden(path: Path) -> bool:
    return any(part.startswith('.') for part in path.parts)


def should_skip(path: Path) -> bool:
    """Check if path matches any skip pattern."""
    path_str = str(path)
    return any(skip in path_str for skip in SKIP_PATTERNS)


def get_files(base_path: Path, pattern: Optional[str]) -> List[Path]:
    """Get files matching pattern, excluding hidden/skipped directories."""
    if not base_path.exists():
        return []

    all_files = []
    for f in base_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in EXTENSIONS:
            continue
        if is_hidden(f):
            continue
        if should_skip(f):
            continue
        if any(f.name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        try:
            if f.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        all_files.append(f)

    # Apply pattern filter
    if pattern:
        if pattern.startswith("!"):
            exclude = pattern[1:].replace("*", "")
            all_files = [f for f in all_files if not f.name.startswith(exclude)]
        else:
            include = pattern.replace("*", "")
            all_files = [f for f in all_files if f.name.startswith(include)]

    return sorted(all_files)

# =============================================================================
# EMBED WORKERS (same proven architecture as process_transcripts.py)
# =============================================================================


def embed_worker(tile_queue: queue.Queue, store_queue: queue.Queue,
                 stats: Dict, stats_lock: threading.Lock):
    name = threading.current_thread().name
    try:
        _embed_loop(tile_queue, store_queue, stats, stats_lock, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _embed_loop(tile_queue, store_queue, stats, stats_lock, name):
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

        # Embed with circuit-breaker retry - survives 2-min container restarts
        texts = [tw.tile.text for tw in batch]
        embeddings = None
        attempt = 0
        while not (_FATAL or _SHUTDOWN):
            attempt += 1
            _circuit.wait_if_open()

            session = get_session()
            result, good_batch = _embed_batch_with_retry(
                session, texts, batch, name, stats, stats_lock)

            if isinstance(result, list):
                # Success (possibly with some oversized tiles filtered out)
                embeddings = result
                batch = good_batch
                break
            elif result == "retry":
                _circuit.report_down()
                wait = min(30, 2 ** min(attempt, 5))
                if attempt % 8 == 1:
                    print(f"  [{name}] Embed unreachable (attempt {attempt}), "
                          f"waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            elif result == "fatal":
                set_fatal(f"[{name}] Embed unexpected error")
                break
            elif result == "error":
                set_fatal(f"[{name}] Embed connection error")
                break

        if _FATAL:
            break
        if not batch or embeddings is None:
            continue

        if len(embeddings) != len(batch):
            set_fatal(f"[{name}] Mismatch: sent {len(batch)}, got {len(embeddings)}")
            break

        for tw, emb in zip(batch, embeddings):
            try:
                store_queue.put(EmbeddedTile(
                    tile=tw.tile, embedding=emb,
                    content_hash=tw.content_hash, filename=tw.filename,
                    stage=tw.stage, priority=tw.priority,
                ), timeout=30)
            except queue.Full:
                if _FATAL or _SHUTDOWN:
                    break
                print(f"  [{name}] Store queue full, waiting...", flush=True)
                store_queue.put(EmbeddedTile(
                    tile=tw.tile, embedding=emb,
                    content_hash=tw.content_hash, filename=tw.filename,
                    stage=tw.stage, priority=tw.priority,
                ))

        with stats_lock:
            stats["embedded"] += len(batch)

# =============================================================================
# STORE WORKERS
# =============================================================================


def store_worker(store_queue: queue.Queue,
                 stats: Dict, stats_lock: threading.Lock):
    name = threading.current_thread().name
    try:
        _store_loop(store_queue, stats, stats_lock, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _store_loop(store_queue, stats, stats_lock, name):
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

        now = datetime.now().isoformat()
        objects = []
        for et in batch:
            uid = tile_uuid(et.content_hash, et.tile.scale, et.tile.index)
            objects.append({
                "class": "ISMA_Quantum",
                "id": uid,
                "properties": {
                    "content": et.tile.text,
                    "source_type": "document",
                    "source_file": et.filename,
                    "layer": LAYER_INT.get(et.stage, 2),
                    "platform": "corpus",
                    "scale": et.tile.scale,
                    "tile_index": et.tile.index,
                    "start_char": et.tile.start_char,
                    "end_char": et.tile.end_char,
                    "token_count": et.tile.estimated_tokens,
                    "content_hash": et.content_hash,
                    "document_id": str(uuid.uuid5(DOC_UUID_NAMESPACE, et.content_hash)),
                    "loaded_at": now,
                    "timestamp": now,
                    "actor": "process_corpus",
                },
                "vector": et.embedding,
            })

        try:
            session = get_session()
            r = session.post(f"{WEAVIATE_URL}/v1/batch/objects",
                             json={"objects": objects}, timeout=60)
        except Exception as e:
            set_fatal(f"[{name}] Weaviate error: {type(e).__name__}: {e}")
            break

        if r.status_code not in [200, 201]:
            set_fatal(f"[{name}] Weaviate {r.status_code}: {r.text[:300]}")
            break

        stored = 0
        for obj_r in r.json():
            errs = obj_r.get("result", {}).get("errors")
            if errs:
                print(f"  WARN [{name}]: {errs}", flush=True)
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
        print(f"  Embedding LB: OK ({model})", flush=True)
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
        driver = get_neo4j()
        with driver.session() as s:
            r = s.run("MATCH (d:Document) RETURN count(d) AS c")
            doc_count = r.single()["c"]
        print(f"  Neo4j: OK ({doc_count} Document nodes)", flush=True)
    except Exception as e:
        print(f"  Neo4j: FAIL - {e}", flush=True)
        ok = False

    # Check corpus directories
    total_files = 0
    for stage_name, path, priority, pattern in STAGES:
        files = get_files(path, pattern)
        total_files += len(files)
        print(f"  {stage_name}: {len(files)} files ({path})", flush=True)

    if total_files == 0:
        print("  Corpus: FAIL - no files found", flush=True)
        ok = False

    if ok:
        print(f"All infrastructure OK. {total_files} corpus files.", flush=True)
    return ok

# =============================================================================
# MAIN
# =============================================================================


def run(stage_filter: str = None, dry_run: bool = False):
    if not check_infra():
        sys.exit(1)

    # Load dedup paths for enriching Document nodes
    dedup_paths = load_dedup_paths()

    stats = {"files": 0, "tiles": 0, "tokens": 0,
             "embedded": 0, "stored": 0, "neo4j_docs": 0, "skipped": 0}
    stats_lock = threading.Lock()
    start_time = time.time()

    tile_queue_obj = queue.Queue(maxsize=TILE_QUEUE_SIZE)
    store_queue_obj = queue.Queue(maxsize=STORE_QUEUE_SIZE)

    threads = []

    if not dry_run:
        # Start embed workers
        for i in range(EMBED_WORKERS):
            t = threading.Thread(target=embed_worker, name=f"embed-{i}",
                                 args=(tile_queue_obj, store_queue_obj,
                                       stats, stats_lock))
            t.start()
            threads.append(('embed', t))

        # Start store workers
        for i in range(STORE_WORKERS):
            t = threading.Thread(target=store_worker, name=f"store-{i}",
                                 args=(store_queue_obj, stats, stats_lock))
            t.start()
            threads.append(('store', t))

        print(f"\nPipeline: reader → tile_queue({TILE_QUEUE_SIZE}) → "
              f"{EMBED_WORKERS} embed (batch={EMBED_BATCH}) → "
              f"store_queue({STORE_QUEUE_SIZE}) → "
              f"{STORE_WORKERS} store", flush=True)

    seen_hashes: Set[str] = set()

    for stage_name, path, priority, pattern in STAGES:
        if _SHUTDOWN or _FATAL:
            break
        if stage_filter and stage_name != stage_filter:
            continue

        files = get_files(path, pattern)
        print(f"\n{'='*60}", flush=True)
        print(f"Stage: {stage_name} (priority={priority}, {len(files)} files)", flush=True)
        print(f"{'='*60}", flush=True)

        for i, filepath in enumerate(files):
            if _SHUTDOWN or _FATAL:
                break

            try:
                content = filepath.read_text(encoding='utf-8')
            except Exception as e:
                print(f"  WARN: Skip {filepath.name}: {e}", flush=True)
                continue

            if not content.strip():
                continue

            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            if content_hash in seen_hashes:
                with stats_lock:
                    stats["skipped"] += 1
                continue
            seen_hashes.add(content_hash)

            # Gather all known paths for this content
            all_paths = dedup_paths.get(content_hash, [])
            if str(filepath) not in all_paths:
                all_paths.append(str(filepath))

            file_size = filepath.stat().st_size
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()

            # Multi-scale tile
            try:
                tiles = multi_scale_tile(content,
                                         source_file=filepath.name,
                                         layer=stage_name)
            except Exception as e:
                print(f"  WARN: Tiling failed for {filepath.name}: {e}", flush=True)
                continue

            tile_count = len(tiles)
            token_count = sum(t.estimated_tokens for t in tiles)

            # No file size limit - all files get tiled and embedded regardless of size

            # Create Neo4j Document node (even in dry run for preview)
            if not dry_run:
                try:
                    doc_uuid = create_document_node(
                        filename=filepath.name,
                        content_hash=content_hash,
                        stage=stage_name,
                        priority=priority,
                        file_size=file_size,
                        all_paths=all_paths,
                        mtime=mtime,
                        tile_count=tile_count,
                    )
                    with stats_lock:
                        stats["neo4j_docs"] += 1
                except Exception as e:
                    print(f"  WARN: Neo4j failed for {filepath.name}: {e}", flush=True)

            print(f"  [{i+1}/{len(files)}] {filepath.name}: "
                  f"{tile_count} tiles, ~{token_count} tok, "
                  f"{len(all_paths)} paths", flush=True)

            with stats_lock:
                stats["files"] += 1
                stats["tiles"] += tile_count
                stats["tokens"] += token_count

            if dry_run:
                continue

            # Feed tiles into queue with timeout to avoid deadlock
            for tile in tiles:
                if _SHUTDOWN or _FATAL:
                    break
                work = TileWork(
                    tile=tile,
                    content_hash=content_hash,
                    filename=filepath.name,
                    stage=stage_name,
                    priority=priority,
                )
                while not (_SHUTDOWN or _FATAL):
                    try:
                        tile_queue_obj.put(work, timeout=2.0)
                        break
                    except queue.Full:
                        continue

    # Drain pipeline
    if not dry_run:
        print("\nDraining pipeline...", flush=True)

        # Wait for tile queue to drain, reporting progress
        while tile_queue_obj.qsize() > 0 or store_queue_obj.qsize() > 0:
            if _FATAL:
                break
            with stats_lock:
                snap = dict(stats)
            tq = tile_queue_obj.qsize()
            sq = store_queue_obj.qsize()
            elapsed = time.time() - start_time
            tok_rate = snap["tokens"] / elapsed if elapsed > 0 else 0
            print(f"  draining: emb={snap['embedded']} stored={snap['stored']} "
                  f"tq={tq} sq={sq} ~{tok_rate:.0f} tok/s", flush=True)
            time.sleep(5)

        # Poison embed workers
        for _ in range(EMBED_WORKERS):
            tile_queue_obj.put(_POISON)
        for role, t in threads:
            if role == 'embed':
                t.join(timeout=120)

        # Poison store workers
        for _ in range(STORE_WORKERS):
            store_queue_obj.put(_POISON)
        for role, t in threads:
            if role == 'store':
                t.join(timeout=60)

    elapsed = time.time() - start_time
    print(f"\n{'DRY RUN ' if dry_run else ''}Complete:", flush=True)
    print(f"  Files: {stats['files']}", flush=True)
    print(f"  Tiles: {stats['tiles']}", flush=True)
    print(f"  Tokens: {stats['tokens']:,}", flush=True)
    if not dry_run:
        print(f"  Embedded: {stats['embedded']}", flush=True)
        print(f"  Stored: {stats['stored']}", flush=True)
        print(f"  Neo4j Documents: {stats['neo4j_docs']}", flush=True)
    print(f"  Skipped (dedup): {stats['skipped']}", flush=True)
    print(f"  Unique hashes: {len(seen_hashes)}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    if elapsed > 0:
        print(f"  Throughput: {stats['tokens']/elapsed:.0f} tok/s", flush=True)

    if _FATAL:
        print(f"\nFATAL: {_FATAL}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISMA Corpus Processor")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--all", action="store_true", help="Process all stages")
    parser.add_argument("--stage", type=str, help="Process single stage")
    args = parser.parse_args()

    try:
        if args.check:
            check_infra()
        elif args.all or args.stage:
            run(stage_filter=args.stage, dry_run=args.dry_run)
        else:
            parser.print_help()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
