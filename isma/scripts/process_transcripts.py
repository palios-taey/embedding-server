#!/usr/bin/env python3
"""
ISMA File-Based Transcript Processor v2

Reads enriched parsed transcript files from disk, normalizes schemas,
tiles exchanges, embeds tiles independently, stores to Weaviate.

Architecture:
  File readers → tile_queue → Embed workers → store_queue → Store workers

  Tiles are the atomic unit. Each tile flows through queues independently.
  No session batching. Nothing held on GPUs. Once processed, it leaves.

Input schemas handled:
  Schema A (ChatGPT, Gemini, Grok, Claude Chat):
    - platform, conversation_id, exchanges[].user_prompt, exchanges[].responses[]
    - responses[] contains: text, artifacts[], tools[], metadata{}
  Schema B (Perplexity):
    - source, conversation_id, exchanges[].prompt, exchanges[].response
  Schema C (Claude Code):
    - platform, conversation_id, exchanges[].role, exchanges[].content
    - Flat message list - pair consecutive user/assistant into exchanges

Usage:
    python3 process_transcripts.py --check
    python3 process_transcripts.py --dry-run --limit 10
    python3 process_transcripts.py --all
    python3 process_transcripts.py --all --resume
    python3 process_transcripts.py --dir /path/to/transcripts --limit 50
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
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"

# Default transcript directory (updated to new enriched format)
DEFAULT_TRANSCRIPT_DIR = "/home/spark/data/transcripts/parsed"

# Pipeline sizing - maximize GPU saturation across 3 Sparks
READER_THREADS = 4       # File readers feeding tiles into queue
EMBED_WORKERS = 8        # Threads pulling from tile_queue, posting to 3 GPUs via LB
EMBED_BATCH = 16         # Tiles per GPU request (vLLM v1 engine bugs out at >20)
STORE_WORKERS = 2        # Threads pulling from store_queue, writing Weaviate
WEAVIATE_BATCH = 100     # Objects per Weaviate batch insert
MAX_EMBED_CHARS = None       # No truncation - vLLM max_model_len handles limits

# Queue sizes (backpressure: readers block when queues are full)
TILE_QUEUE_SIZE = 512    # Tiles waiting to be embedded
STORE_QUEUE_SIZE = 256   # Embedded tiles waiting to be stored

# Deterministic UUIDs
TILE_UUID_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')

# Progress
PROGRESS_DIR = Path("/var/spark/isma")
PROGRESS_FILE = PROGRESS_DIR / "transcript_progress_v2.json"
HASHES_FILE = PROGRESS_DIR / "transcript_hashes_v2.json"

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
    print(f"\nSignal {signum}, draining queues...", flush=True)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# Sentinel
_POISON = None


# =============================================================================
# CIRCUIT BREAKER - shared across all embed workers
# =============================================================================

class _CircuitBreaker:
    """When one worker detects backend down, all workers wait together."""
    def __init__(self, recovery_time=150.0):
        self._lock = threading.Lock()
        self._down_since = None
        self._recovery_time = recovery_time  # > 120s container restart

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
        """If circuit open, sleep until near end of recovery window."""
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
class EnrichedExchange:
    """Normalized exchange with all searchable content."""
    user_text: str
    assistant_text: str
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    thinking: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    has_artifacts: bool = False
    artifact_count: int = 0
    has_thinking: bool = False


@dataclass
class TileWork:
    """A tile ready to be embedded."""
    tile: MultiScaleTile
    conversation_id: str
    platform: str
    content_hash: str
    source_file: str
    model: str
    has_artifacts: bool
    artifact_count: int
    has_thinking: bool


@dataclass
class EmbeddedTile:
    """A tile with its embedding, ready to be stored."""
    tile: MultiScaleTile
    embedding: List[float]
    conversation_id: str
    platform: str
    content_hash: str
    source_file: str
    model: str
    has_artifacts: bool
    artifact_count: int
    has_thinking: bool

# =============================================================================
# HTTP (thread-local sessions - requests.Session is NOT thread-safe)
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
# UUID HELPERS
# =============================================================================


def tile_uuid(content_hash: str, scale: str, index: int) -> str:
    """Deterministic UUID for a tile. Same content → same UUID (idempotent)."""
    return str(uuid.uuid5(TILE_UUID_NAMESPACE, f"{content_hash}:{scale}:{index}"))


def make_uuid(text: str) -> str:
    """Convert non-UUID string to deterministic UUID using uuid5."""
    try:
        uuid.UUID(text)
        return text
    except (ValueError, AttributeError):
        return str(uuid.uuid5(TILE_UUID_NAMESPACE, text or "unknown"))

# =============================================================================
# SCHEMA NORMALIZATION
# =============================================================================


def detect_platform(filepath: str, data: dict) -> str:
    """Detect platform from directory path or data."""
    # Check data first
    platform = data.get('platform') or data.get('source')
    if platform:
        return platform.lower()

    # Fall back to directory name
    parts = Path(filepath).parts
    known = {'chatgpt', 'claude_code', 'claude_chat', 'gemini', 'grok', 'perplexity'}
    for part in parts:
        if part in known:
            return part
    return 'unknown'


def normalize_exchanges(data: dict, filepath: str) -> List[EnrichedExchange]:
    """Normalize JSON transcript into enriched exchange objects.

    Returns list of EnrichedExchange objects with all searchable content.
    Handles Schema A (ChatGPT/Gemini/Grok/Claude Chat), Schema B (Perplexity),
    and Schema C (Claude Code) transparently.
    """
    exchanges = data.get('exchanges', [])
    if not exchanges:
        return []

    platform = detect_platform(filepath, data)
    top_level_model = data.get('model', '')

    results = []

    # Schema C: Claude Code (flat role/content messages)
    if exchanges and 'role' in exchanges[0]:
        # Pair consecutive user+assistant messages
        i = 0
        while i < len(exchanges):
            ex = exchanges[i]
            if ex.get('role') == 'user':
                user_text = ex.get('content', '').strip()
                assistant_text = ""
                timestamp = ex.get('timestamp', '')

                # Look for following assistant message
                if i + 1 < len(exchanges) and exchanges[i + 1].get('role') == 'assistant':
                    assistant_text = exchanges[i + 1].get('content', '').strip()
                    i += 2
                else:
                    i += 1

                results.append(EnrichedExchange(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    timestamp=timestamp,
                    model='claude_code',
                ))
            else:
                i += 1
        return results

    # Schema B: Perplexity (prompt/response)
    if exchanges and 'prompt' in exchanges[0]:
        for ex in exchanges:
            user = ex.get('prompt', '') or ''
            resp = ex.get('response', '') or ''
            timestamp = ex.get('timestamp', '')

            if isinstance(resp, dict):
                resp = resp.get('text', '') or ''

            results.append(EnrichedExchange(
                user_text=user.strip(),
                assistant_text=resp.strip(),
                timestamp=timestamp,
                model='perplexity',
            ))
        return results

    # Schema A: ChatGPT/Gemini/Grok/Claude Chat (user_prompt/responses[])
    for ex in exchanges:
        user = ex.get('user_prompt', '') or ''
        timestamp = ex.get('timestamp', '')

        # Responses is a list of dicts with text, artifacts, tools, metadata
        responses = ex.get('responses', [])
        resp_parts = []
        all_artifacts = []
        all_tools = []
        thinking = ""
        model = ""

        for r in responses:
            if isinstance(r, str):
                resp_parts.append(r)
                continue

            if not isinstance(r, dict):
                continue

            # Main response text
            text = r.get('text', '') or ''
            if text:
                resp_parts.append(text)

            # Artifacts
            artifacts = r.get('artifacts', [])
            if artifacts:
                all_artifacts.extend(artifacts)

            # Tools
            tools = r.get('tools', [])
            if tools:
                all_tools.extend(tools)

            # Metadata (thinking trace, model, etc.)
            metadata = r.get('metadata', {})
            if metadata:
                if not thinking and 'thinking_trace' in metadata:
                    thinking = metadata['thinking_trace']
                if not model and 'model' in metadata:
                    model = metadata['model']

            # Model at response level (ChatGPT format)
            if not model and 'model' in r:
                model = r['model']

        # Use top-level model if not found in responses
        if not model:
            model = top_level_model

        assistant = '\n'.join(resp_parts).strip()

        results.append(EnrichedExchange(
            user_text=user.strip(),
            assistant_text=assistant,
            artifacts=all_artifacts,
            thinking=thinking,
            tools=all_tools,
            model=model,
            timestamp=timestamp,
            has_artifacts=len(all_artifacts) > 0,
            artifact_count=len(all_artifacts),
            has_thinking=bool(thinking),
        ))

    return results


def get_conversation_id(data: dict) -> str:
    """Extract conversation ID from any schema."""
    return (data.get('conversation_id')
            or data.get('sessionId')
            or data.get('session_id')
            or '')


def format_exchange(exchange: EnrichedExchange, conversation_id: str,
                    platform: str) -> str:
    """Format an enriched exchange for embedding.

    Includes all searchable content: user prompt, assistant response,
    artifacts (full content), thinking traces (truncated), tool results.
    """
    parts = [
        f"[Platform: {platform}] [Session: {conversation_id[:12]}]"
    ]

    if exchange.model:
        parts.append(f"[Model: {exchange.model}]")

    if exchange.user_text:
        parts.append(f"\n[User]: {exchange.user_text}")

    if exchange.assistant_text:
        parts.append(f"\n[Assistant]: {exchange.assistant_text}")

    # Include thinking trace (first 2000 chars - can be huge)
    if exchange.thinking:
        thinking_trunc = exchange.thinking[:2000]
        if len(exchange.thinking) > 2000:
            thinking_trunc += "... (truncated)"
        parts.append(f"\n[Thinking]: {thinking_trunc}")

    # Include full artifact content - multi_scale_tile will chunk appropriately
    for artifact in exchange.artifacts:
        atype = artifact.get('type', 'unknown')
        lang = artifact.get('language', '')
        content = artifact.get('content', '')

        if content:
            header = f"[Artifact {atype}"
            if lang:
                header += f"/{lang}"
            header += "]:"
            parts.append(f"\n{header}\n{content}")

    # Include substantial tool results
    for tool in exchange.tools:
        if isinstance(tool, dict):
            tool_output = tool.get('output', '') or tool.get('result', '') or ''
            if len(tool_output) > 50:
                tool_type = tool.get('type', 'unknown')
                parts.append(f"\n[Tool {tool_type}]: {tool_output}")

    return "\n".join(parts)

# =============================================================================
# FILE DISCOVERY
# =============================================================================


def discover_files(base_dir: str) -> List[Tuple[str, str]]:
    """Discover all JSON transcript files.

    Returns list of (filepath, platform) tuples sorted by path.
    Detects platform from directory name (fast - no file I/O).
    """
    files = []
    base = Path(base_dir)

    if not base.exists():
        print(f"ERROR: Directory not found: {base_dir}", flush=True)
        return files

    known = {'chatgpt', 'claude_code', 'claude_chat', 'gemini', 'grok', 'perplexity'}

    for json_file in sorted(base.rglob("*.json")):
        # Detect platform from directory structure (fast)
        platform = 'unknown'
        for part in json_file.parts:
            if part in known:
                platform = part
                break
        files.append((str(json_file), platform))

    return files

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
# STAGE 1: FILE READERS - Read files, normalize, tile, feed tile_queue
# =============================================================================


def reader_worker(file_list: List[Tuple[int, str, str]],
                  tile_queue: queue.Queue,
                  seen_hashes: set, hash_lock: threading.Lock,
                  stats: Dict, stats_lock: threading.Lock,
                  dry_run: bool):
    """Read transcript files, normalize exchanges, tile, feed queue."""
    name = threading.current_thread().name
    try:
        _reader_loop(file_list, tile_queue, seen_hashes, hash_lock,
                     stats, stats_lock, dry_run, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _reader_loop(file_list, tile_queue, seen_hashes, hash_lock,
                 stats, stats_lock, dry_run, name):
    for idx, filepath, platform in file_list:
        if _SHUTDOWN or _FATAL:
            break

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARN [{name}]: Skip {filepath}: {e}", flush=True)
            with stats_lock:
                stats["errors"] += 1
            continue

        conversation_id = get_conversation_id(data) or Path(filepath).stem
        exchanges = normalize_exchanges(data, filepath)

        file_tiles = 0
        file_tokens = 0
        file_skipped = 0

        for exchange in exchanges:
            if _SHUTDOWN or _FATAL:
                break

            # Skip empty exchanges
            if len(exchange.user_text) + len(exchange.assistant_text) < 20:
                file_skipped += 1
                continue

            exchange_text = format_exchange(exchange, conversation_id, platform)
            ch = hashlib.sha256(exchange_text.encode()).hexdigest()[:16]

            with hash_lock:
                if ch in seen_hashes:
                    file_skipped += 1
                    continue

            tiles = multi_scale_tile(exchange_text,
                                     source_file=Path(filepath).name,
                                     layer="transcript")

            file_tiles += len(tiles)
            file_tokens += sum(t.estimated_tokens for t in tiles)

            if dry_run:
                with hash_lock:
                    seen_hashes.add(ch)
                continue

            # Feed tiles one at a time into queue with timeout to avoid deadlock
            for tile in tiles:
                if _SHUTDOWN or _FATAL:
                    break
                work = TileWork(
                    tile=tile,
                    conversation_id=conversation_id,
                    platform=platform,
                    content_hash=ch,
                    source_file=Path(filepath).name,
                    model=exchange.model,
                    has_artifacts=exchange.has_artifacts,
                    artifact_count=exchange.artifact_count,
                    has_thinking=exchange.has_thinking,
                )
                # Timeout on put() prevents deadlock if embed workers die
                while not (_SHUTDOWN or _FATAL):
                    try:
                        tile_queue.put(work, timeout=2.0)
                        break
                    except queue.Full:
                        continue

            with hash_lock:
                seen_hashes.add(ch)

        with stats_lock:
            stats["files"] += 1
            stats["exchanges"] += len(exchanges)
            stats["tiles"] += file_tiles
            stats["tokens"] += file_tokens
            stats["skipped"] += file_skipped

# =============================================================================
# STAGE 2: EMBED WORKERS - Pull tiles, embed via GPU, push to store_queue
# =============================================================================


def embed_worker(tile_queue: queue.Queue,
                 store_queue: queue.Queue,
                 stats: Dict, stats_lock: threading.Lock):
    """Pull tiles in batches, embed via GPU, push to store queue."""
    name = threading.current_thread().name
    try:
        _embed_loop(tile_queue, store_queue, stats, stats_lock, name)
    except Exception as e:
        set_fatal(f"[{name}] crashed: {type(e).__name__}: {e}")
        traceback.print_exc()


def _embed_loop(tile_queue, store_queue, stats, stats_lock, name):
    while not _FATAL:
        batch = []

        # Block for first item
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
                    tile=tw.tile,
                    embedding=emb,
                    conversation_id=tw.conversation_id,
                    platform=tw.platform,
                    content_hash=tw.content_hash,
                    source_file=tw.source_file,
                    model=tw.model,
                    has_artifacts=tw.has_artifacts,
                    artifact_count=tw.artifact_count,
                    has_thinking=tw.has_thinking,
                ), timeout=30)
            except queue.Full:
                if _FATAL or _SHUTDOWN:
                    break
                print(f"  [{name}] Store queue full, waiting...", flush=True)
                store_queue.put(EmbeddedTile(
                    tile=tw.tile,
                    embedding=emb,
                    conversation_id=tw.conversation_id,
                    platform=tw.platform,
                    content_hash=tw.content_hash,
                    source_file=tw.source_file,
                    model=tw.model,
                    has_artifacts=tw.has_artifacts,
                    artifact_count=tw.artifact_count,
                    has_thinking=tw.has_thinking,
                ))

        with stats_lock:
            stats["embedded"] += len(batch)

# =============================================================================
# STAGE 3: STORE WORKERS - Batch-write embedded tiles to Weaviate
# =============================================================================


def store_worker(store_queue: queue.Queue,
                 stats: Dict, stats_lock: threading.Lock):
    """Pull embedded tiles, batch-write to Weaviate."""
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

            # Convert conversation_id to valid UUID if needed
            session_uuid = make_uuid(et.conversation_id)

            objects.append({
                "class": "ISMA_Quantum",
                "id": uid,
                "properties": {
                    "content": et.tile.text,
                    "source_type": "transcript",
                    "source_file": et.source_file,
                    "layer": 2,
                    "platform": et.platform,
                    "session_id": session_uuid,
                    "scale": et.tile.scale,
                    "parent_tile_id": "",
                    "tile_index": et.tile.index,
                    "start_char": et.tile.start_char,
                    "end_char": et.tile.end_char,
                    "token_count": et.tile.estimated_tokens,
                    "content_hash": et.content_hash,
                    "loaded_at": now,
                    "timestamp": now,
                    "actor": "process_transcripts_v2",
                    "model": et.model,
                    "has_artifacts": et.has_artifacts,
                    "artifact_count": et.artifact_count,
                    "has_thinking": et.has_thinking,
                    "conversation_id": et.conversation_id,
                },
                "vector": et.embedding,
            })

        try:
            session = get_session()
            r = session.post(f"{WEAVIATE_URL}/v1/batch/objects",
                             json={"objects": objects}, timeout=60)
        except Exception as e:
            set_fatal(f"[{name}] Weaviate batch error: {type(e).__name__}: {e}")
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


def check_infra(transcript_dir: str) -> bool:
    print("Checking infrastructure...", flush=True)
    ok = True

    try:
        r = requests.get(EMBEDDING_URL.replace('/v1/embeddings', '/v1/models'),
                         timeout=5)
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

    # Check transcript directory
    files = discover_files(transcript_dir)
    if files:
        # Count per platform
        platforms = {}
        for _, p in files:
            platforms[p] = platforms.get(p, 0) + 1
        plat_str = ', '.join(f"{p}={c}" for p, c in sorted(platforms.items()))
        print(f"  Transcripts: OK ({len(files)} files: {plat_str})", flush=True)
    else:
        print(f"  Transcripts: FAIL - no JSON files in {transcript_dir}", flush=True)
        ok = False

    if ok:
        print("All infrastructure OK.", flush=True)
    return ok

# =============================================================================
# MAIN
# =============================================================================


def run(transcript_dir: str, limit: int = 0, dry_run: bool = False,
        resume: bool = False):
    if not check_infra(transcript_dir):
        sys.exit(1)

    print("\nDiscovering transcript files...", flush=True)
    all_files = discover_files(transcript_dir)
    print(f"Total: {len(all_files)} files", flush=True)

    if limit > 0:
        all_files = all_files[:limit]
        print(f"Limited to {len(all_files)} files", flush=True)

    # Load state
    seen = load_hashes() if resume else set()
    hash_lock = threading.Lock()

    if resume:
        progress = load_progress()
        print(f"Resuming with {len(seen)} known hashes", flush=True)

    stats = {"files": 0, "exchanges": 0, "tiles": 0, "tokens": 0,
             "skipped": 0, "embedded": 0, "stored": 0, "errors": 0}
    stats_lock = threading.Lock()
    start_time = time.time()
    total_files = len(all_files)

    if dry_run:
        print("\nDRY RUN - no data will be written\n", flush=True)
    else:
        print(f"\nPipeline: {READER_THREADS} readers → "
              f"tile_queue({TILE_QUEUE_SIZE}) → "
              f"{EMBED_WORKERS} embed workers (batch={EMBED_BATCH}) → "
              f"store_queue({STORE_QUEUE_SIZE}) → "
              f"{STORE_WORKERS} store workers", flush=True)

    tile_queue_obj = queue.Queue(maxsize=TILE_QUEUE_SIZE)
    store_queue_obj = queue.Queue(maxsize=STORE_QUEUE_SIZE)

    # Split files across reader threads
    chunks = [[] for _ in range(READER_THREADS)]
    for i, (filepath, platform) in enumerate(all_files):
        chunks[i % READER_THREADS].append((i, filepath, platform))

    threads = []

    # Start readers
    for i in range(READER_THREADS):
        t = threading.Thread(target=reader_worker, name=f"reader-{i}",
                             args=(chunks[i], tile_queue_obj, seen, hash_lock,
                                   stats, stats_lock, dry_run))
        t.start()
        threads.append(('reader', t))

    if not dry_run:
        # Embed workers
        for i in range(EMBED_WORKERS):
            t = threading.Thread(target=embed_worker, name=f"embed-{i}",
                                 args=(tile_queue_obj, store_queue_obj,
                                       stats, stats_lock))
            t.start()
            threads.append(('embed', t))

        # Store workers
        for i in range(STORE_WORKERS):
            t = threading.Thread(target=store_worker, name=f"store-{i}",
                                 args=(store_queue_obj, stats, stats_lock))
            t.start()
            threads.append(('store', t))

    # Progress reporting - every 10s regardless of file completion
    last_emb = 0
    last_time = time.time()
    while True:
        time.sleep(10)

        with stats_lock:
            done = stats["files"]
            snap = dict(stats)

        elapsed = time.time() - start_time
        interval = time.time() - last_time
        rate = done / elapsed if elapsed > 0 else 0
        tok_rate = snap["tokens"] / elapsed if elapsed > 0 else 0
        tq = tile_queue_obj.qsize() if not dry_run else 0
        sq = store_queue_obj.qsize() if not dry_run else 0

        # Show embed rate in this interval
        emb_delta = snap["embedded"] - last_emb
        emb_rate = emb_delta / interval if interval > 0 else 0

        # Count alive threads by role
        alive = {}
        for role, t in threads:
            alive.setdefault(role, 0)
            if t.is_alive():
                alive[role] += 1

        alive_str = ' '.join(f"{r}={c}" for r, c in sorted(alive.items()))

        print(f"  [{done}/{total_files}] "
              f"tiles={snap['tiles']} emb={snap['embedded']}(+{emb_delta}) "
              f"stored={snap['stored']} skip={snap['skipped']} "
              f"tq={tq} sq={sq} "
              f"~{tok_rate:.0f} tok/s emb/s={emb_rate:.0f} "
              f"threads=[{alive_str}]", flush=True)

        last_emb = snap["embedded"]
        last_time = time.time()

        # Checkpoint every 200 files
        if not dry_run and done > 0 and done % 200 == 0:
            with hash_lock:
                h_copy = set(seen)
            save_progress({
                "files_done": done,
                "tiles_stored": snap["stored"],
                "tokens_embedded": snap["tokens"],
                "checkpoint_at": datetime.now().isoformat(),
            })
            save_hashes(h_copy)

        readers_alive = any(t.is_alive() for role, t in threads
                            if role == 'reader')
        if not readers_alive:
            break

        if _FATAL:
            break

    # Wait for readers
    for role, t in threads:
        if role == 'reader':
            t.join(timeout=30)

    if not dry_run:
        # Poison tile queue → embed workers drain
        for _ in range(EMBED_WORKERS):
            tile_queue_obj.put(_POISON)
        for role, t in threads:
            if role == 'embed':
                t.join(timeout=120)

        # Poison store queue → store workers drain
        for _ in range(STORE_WORKERS):
            store_queue_obj.put(_POISON)
        for role, t in threads:
            if role == 'store':
                t.join(timeout=60)

    # Final save
    if not dry_run:
        save_progress({
            "files_done": stats["files"],
            "tiles_stored": stats["stored"],
            "tokens_embedded": stats["tokens"],
            "completed_at": datetime.now().isoformat(),
        })
        save_hashes(seen)

    elapsed = time.time() - start_time
    print(f"\n{'DRY RUN ' if dry_run else ''}Complete:", flush=True)
    print(f"  Files: {stats['files']}", flush=True)
    print(f"  Exchanges: {stats['exchanges']}", flush=True)
    print(f"  Tiles: {stats['tiles']}", flush=True)
    if not dry_run:
        print(f"  Embedded: {stats['embedded']}", flush=True)
        print(f"  Stored: {stats['stored']}", flush=True)
    print(f"  Skipped: {stats['skipped']}", flush=True)
    print(f"  Errors: {stats['errors']}", flush=True)
    print(f"  Tokens: {stats['tokens']:,}", flush=True)
    print(f"  Hashes: {len(seen)}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    if elapsed > 0 and stats["files"] > 0:
        print(f"  Rate: {stats['files']/elapsed:.1f} files/s", flush=True)
        print(f"  Throughput: {stats['tokens']/elapsed:.0f} tok/s", flush=True)

    if _FATAL:
        print(f"\nExiting due to fatal error: {_FATAL}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISMA Transcript Processor v2 (Enriched Format)")
    parser.add_argument("--check", action="store_true",
                        help="Check infrastructure only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview: read/tile but don't embed or store")
    parser.add_argument("--all", action="store_true",
                        help="Process all files")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip known hashes)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files")
    parser.add_argument("--dir", type=str, default=DEFAULT_TRANSCRIPT_DIR,
                        help="Transcript directory")
    args = parser.parse_args()

    try:
        if args.check:
            check_infra(args.dir)
        elif args.all or args.limit > 0:
            run(args.dir, limit=args.limit, dry_run=args.dry_run,
                resume=args.resume)
        else:
            parser.print_help()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
