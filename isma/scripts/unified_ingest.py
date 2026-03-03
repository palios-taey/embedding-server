#!/usr/bin/env python3
"""
ISMA Unified Ingest Pipeline

ONE pipeline that reads ALL data sources (corpus + transcripts),
deduplicates via pre-built manifest, phi-tiles at 3 scales,
embeds via vLLM, and writes to BOTH Weaviate AND Neo4j.

Replaces: process_corpus.py, process_transcripts.py, reprocess_neo4j.py

Architecture:
  load_manifest → discover files → skip dupes →
    phi_tile(3 scales) + link_parent_tile_ids →
    embed_batch(vLLM) →
    write_weaviate(tiles) + write_neo4j(session/exchange OR document) →
    checkpoint

Pipeline workers:
  4 readers → tile_queue(512) → 8 embed workers (batch=16) →
    store_queue(256) → 2 store workers (dual-write)

Usage:
    python3 unified_ingest.py --check              # Verify infrastructure
    python3 unified_ingest.py --all                 # Full rebuild from manifest
    python3 unified_ingest.py --stage kernel        # Single corpus stage
    python3 unified_ingest.py --stage transcripts   # All transcripts
    python3 unified_ingest.py --stage transcripts --platform claude_code
    python3 unified_ingest.py --incremental         # New files since last run
    python3 unified_ingest.py --dry-run --limit 10
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
from typing import List, Dict, Any, Optional, Tuple, Set
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

DATA_BASE = Path("/home/spark/data")
CORPUS_BASE = DATA_BASE / "corpus"
TRANSCRIPT_BASE = DATA_BASE / "transcripts" / "parsed"

MANIFEST_FILE = Path("/var/spark/isma/dedup_manifest.json")
PROGRESS_DIR = Path("/var/spark/isma")
PROGRESS_FILE = PROGRESS_DIR / "unified_progress.json"

# Corpus stages (same order as process_corpus.py)
STAGES = [
    ("kernel",        CORPUS_BASE / "kernel",                    1.0,  None),
    ("v0",            CORPUS_BASE / "layer_0",                   0.95, "v0*"),
    ("layer_0",       CORPUS_BASE / "layer_0",                   0.9,  "!v0*"),
    ("chewy",         DATA_BASE / "chewy",                       0.85, None),
    ("chewy-gallery", DATA_BASE / "chewy-consciousness-gallery", 0.8,  None),
    ("layer_1",       CORPUS_BASE / "layer_1",                   0.75, None),
    ("layer_2",       CORPUS_BASE / "layer_2",                   0.6,  None),
    ("github-repos",  DATA_BASE / "github-repos",                0.5,  None),
    ("expansion_md",  DATA_BASE / "expansion_md",                0.4,  None),
    ("mira_md",       DATA_BASE / "mira_md_files",               0.3,  None),
    ("mac_all_md",    CORPUS_BASE / "mac_all_md",                0.2,  None),
    ("spark_loose",   CORPUS_BASE / "spark_loose",               0.2,  None),
    ("ccm_new_mds",   DATA_BASE / "CCM_NEW_MDS",                 0.5,  None),
    # Transcripts last
    ("transcripts",   TRANSCRIPT_BASE,                           0.0,  None),
]

LAYER_INT = {
    "kernel": -1, "v0": 0, "layer_0": 0,
    "chewy": 0, "chewy-gallery": 0,
    "layer_1": 1, "layer_2": 2,
    "github-repos": 2, "expansion_md": 2, "mira_md": 2,
    "mac_all_md": 2, "spark_loose": 2,
    "ccm_new_mds": 1,
}

# Pipeline sizing
EMBED_WORKERS = 8
EMBED_BATCH = 16
STORE_WORKERS = 2
WEAVIATE_BATCH = 100
TILE_QUEUE_SIZE = 512
STORE_QUEUE_SIZE = 256

# UUIDs (same namespaces as existing scripts)
TILE_NS = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')
DOC_NS = uuid.UUID('d0c1b2a3-e4f5-6789-0abc-def123456789')
SESSION_NS = uuid.UUID('b2c3d4e5-f6a7-8901-bcde-f12345678901')

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

_POISON = object()


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class _CircuitBreaker:
    def __init__(self, recovery_time=150.0):
        self._lock = threading.Lock()
        self._down_since = None
        self._recovery_time = recovery_time

    def report_down(self):
        with self._lock:
            if self._down_since is None:
                self._down_since = time.monotonic()
                print("  [circuit] OPEN - backend down", flush=True)

    def report_up(self):
        with self._lock:
            if self._down_since is not None:
                elapsed = time.monotonic() - self._down_since
                self._down_since = None
                print(f"  [circuit] CLOSED after {elapsed:.0f}s", flush=True)

    def wait_if_open(self, probe_interval=15.0):
        with self._lock:
            if self._down_since is None:
                return
            elapsed = time.monotonic() - self._down_since
            remaining = self._recovery_time - elapsed
        if remaining > probe_interval:
            time.sleep(remaining - probe_interval)


_circuit = _CircuitBreaker(recovery_time=150.0)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class TileWork:
    """A tile ready to be embedded."""
    tile: MultiScaleTile
    content_hash: str
    source_type: str           # "document" or "transcript"
    source_file: str           # Relative path
    # Corpus fields
    stage: str = ""
    priority: float = 0.0
    document_id: str = ""
    timestamp: str = ""        # File mtime for corpus, exchange timestamp for transcripts
    # Transcript fields
    conversation_id: str = ""
    session_id: str = ""
    platform: str = ""
    model: str = ""
    has_artifacts: bool = False
    artifact_count: int = 0
    has_thinking: bool = False
    exchange_index: int = 0
    parent_tile_id: str = ""   # UUID of parent context_2048 tile


@dataclass
class EmbeddedTile:
    """A tile with its embedding, ready for dual-store write."""
    tile: MultiScaleTile
    embedding: List[float]
    content_hash: str
    source_type: str
    source_file: str
    stage: str = ""
    priority: float = 0.0
    document_id: str = ""
    timestamp: str = ""
    conversation_id: str = ""
    session_id: str = ""
    platform: str = ""
    model: str = ""
    has_artifacts: bool = False
    artifact_count: int = 0
    has_thinking: bool = False
    exchange_index: int = 0
    parent_tile_id: str = ""


# =============================================================================
# HTTP (thread-local sessions)
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
_NEO4J_LOCK = threading.Lock()


def get_neo4j():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is None:
        from neo4j import GraphDatabase
        _NEO4J_DRIVER = GraphDatabase.driver(NEO4J_URI, auth=None)
        _NEO4J_DRIVER.verify_connectivity()
    return _NEO4J_DRIVER


def write_neo4j_document(content_hash: str, filename: str, all_paths: List[str],
                         stage: str, priority: float, file_size: int,
                         mtime: str, tile_count: int):
    """Create/update Document node in Neo4j."""
    doc_uuid = str(uuid.uuid5(DOC_NS, content_hash))
    driver = get_neo4j()
    with driver.session() as s:
        s.run("""
            MERGE (d:Document {content_hash: $hash})
            SET d.document_id = $uuid,
                d.filename = $filename,
                d.all_paths = $paths,
                d.layer = $stage,
                d.priority = $priority,
                d.file_size = $file_size,
                d.last_modified = $mtime,
                d.tile_count = $tile_count,
                d.loaded_at = $now,
                d.actor = 'unified_ingest_v1'
        """, hash=content_hash, uuid=doc_uuid, filename=filename,
             paths=all_paths, stage=stage, priority=priority,
             file_size=file_size, mtime=mtime,
             tile_count=tile_count, now=datetime.now().isoformat())
    return doc_uuid


def write_neo4j_session(session_id: str, platform: str, title: str,
                        source_file: str, exchange_count: int,
                        created_at: str, model: str):
    """Create/update ISMASession node in Neo4j."""
    driver = get_neo4j()
    with driver.session() as s:
        s.run("""
            MERGE (s:ISMASession {session_id: $session_id})
            SET s.platform = $platform,
                s.title = $title,
                s.source_file = $source_file,
                s.exchange_count = $count,
                s.created_at = $created_at,
                s.model = $model,
                s.loaded_at = $now,
                s.actor = 'unified_ingest_v1'
        """, session_id=session_id, platform=platform, title=title,
             source_file=source_file, count=exchange_count,
             created_at=created_at, model=model,
             now=datetime.now().isoformat())


def write_neo4j_exchange(content_hash: str, session_id: str, index: int,
                         user_prompt: str, response: str,
                         timestamp: str, model: str):
    """Create/update ISMAExchange node and link to session."""
    driver = get_neo4j()
    with driver.session() as s:
        s.run("""
            MERGE (e:ISMAExchange {content_hash: $hash})
            SET e.session_id = $session_id,
                e.exchange_index = $index,
                e.user_prompt = $user_prompt,
                e.response = $response,
                e.timestamp = $timestamp,
                e.model = $model,
                e.loaded_at = $now,
                e.actor = 'unified_ingest_v1'
            WITH e
            MATCH (s:ISMASession {session_id: $session_id})
            MERGE (s)-[:CONTAINS]->(e)
        """, hash=content_hash, session_id=session_id, index=index,
             user_prompt=user_prompt[:10000], response=response[:50000],
             timestamp=timestamp, model=model,
             now=datetime.now().isoformat())


# =============================================================================
# UUID HELPERS
# =============================================================================

def tile_uuid(content_hash: str, scale: str, index: int) -> str:
    return str(uuid.uuid5(TILE_NS, f"{content_hash}:{scale}:{index}"))


def make_session_uuid(conversation_id: str) -> str:
    return str(uuid.uuid5(SESSION_NS, conversation_id or "unknown"))


# =============================================================================
# EMBEDDING
# =============================================================================

def _embed_batch_with_retry(session, texts, batch, name):
    try:
        r = session.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL, "input": texts,
        }, timeout=120)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return "retry", batch
    except Exception:
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
# TRANSCRIPT SCHEMA NORMALIZATION (from process_transcripts.py)
# =============================================================================

@dataclass
class EnrichedExchange:
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


def normalize_exchanges(data: dict, filepath: str) -> List[EnrichedExchange]:
    """Normalize JSON transcript into enriched exchange objects."""
    exchanges = data.get('exchanges', [])
    if not exchanges:
        return []

    top_level_model = data.get('model', '')
    results = []

    # Schema C: Claude Code (flat role/content messages)
    if exchanges and 'role' in exchanges[0]:
        i = 0
        while i < len(exchanges):
            ex = exchanges[i]
            if ex.get('role') == 'user':
                user_text = ex.get('content', '').strip()
                assistant_text = ""
                timestamp = ex.get('timestamp', '')
                if i + 1 < len(exchanges) and exchanges[i + 1].get('role') == 'assistant':
                    assistant_text = exchanges[i + 1].get('content', '').strip()
                    i += 2
                else:
                    i += 1
                results.append(EnrichedExchange(
                    user_text=user_text, assistant_text=assistant_text,
                    timestamp=timestamp, model='claude_code',
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
                user_text=user.strip(), assistant_text=resp.strip(),
                timestamp=timestamp, model='perplexity',
            ))
        return results

    # Schema A: ChatGPT/Gemini/Grok/Claude Chat
    for ex in exchanges:
        user = ex.get('user_prompt', '') or ''
        timestamp = ex.get('timestamp', '')
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
            text = r.get('text', '') or ''
            if text:
                resp_parts.append(text)
            artifacts = r.get('artifacts', [])
            if artifacts:
                all_artifacts.extend(artifacts)
            tools = r.get('tools', [])
            if tools:
                all_tools.extend(tools)
            metadata = r.get('metadata', {})
            if metadata:
                if not thinking and 'thinking_trace' in metadata:
                    thinking = metadata['thinking_trace']
                if not model and 'model' in metadata:
                    model = metadata['model']
            if not model and 'model' in r:
                model = r['model']

        if not model:
            model = top_level_model
        assistant = '\n'.join(resp_parts).strip()

        results.append(EnrichedExchange(
            user_text=user.strip(), assistant_text=assistant,
            artifacts=all_artifacts, thinking=thinking,
            tools=all_tools, model=model, timestamp=timestamp,
            has_artifacts=len(all_artifacts) > 0,
            artifact_count=len(all_artifacts),
            has_thinking=bool(thinking),
        ))

    return results


def format_exchange(exchange: EnrichedExchange, conversation_id: str,
                    platform: str) -> str:
    """Format enriched exchange for embedding (all searchable content)."""
    parts = [f"[Platform: {platform}] [Session: {conversation_id[:12]}]"]

    if exchange.model:
        parts.append(f"[Model: {exchange.model}]")
    if exchange.user_text:
        parts.append(f"\n[User]: {exchange.user_text}")
    if exchange.assistant_text:
        parts.append(f"\n[Assistant]: {exchange.assistant_text}")
    if exchange.thinking:
        thinking_trunc = exchange.thinking[:2000]
        if len(exchange.thinking) > 2000:
            thinking_trunc += "... (truncated)"
        parts.append(f"\n[Thinking]: {thinking_trunc}")
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
    for tool in exchange.tools:
        if isinstance(tool, dict):
            tool_output = tool.get('output', '') or tool.get('result', '') or ''
            if len(tool_output) > 50:
                tool_type = tool.get('type', 'unknown')
                parts.append(f"\n[Tool {tool_type}]: {tool_output}")

    return "\n".join(parts)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def hash_exchange(user_text: str, assistant_text: str) -> str:
    combined = normalize_text(user_text) + "\n" + normalize_text(assistant_text)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# =============================================================================
# EMBED WORKERS
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

        texts = [tw.tile.text for tw in batch]
        embeddings = None
        attempt = 0
        while not (_FATAL or _SHUTDOWN):
            attempt += 1
            _circuit.wait_if_open()
            session = get_session()
            result, good_batch = _embed_batch_with_retry(session, texts, batch, name)

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
            elif result in ("fatal", "error"):
                set_fatal(f"[{name}] Embed error: {result}")
                break

        if _FATAL or not batch or embeddings is None:
            break

        if len(embeddings) != len(batch):
            set_fatal(f"[{name}] Mismatch: sent {len(batch)}, got {len(embeddings)}")
            break

        for tw, emb in zip(batch, embeddings):
            et = EmbeddedTile(
                tile=tw.tile, embedding=emb, content_hash=tw.content_hash,
                source_type=tw.source_type, source_file=tw.source_file,
                stage=tw.stage, priority=tw.priority,
                document_id=tw.document_id, timestamp=tw.timestamp,
                conversation_id=tw.conversation_id, session_id=tw.session_id,
                platform=tw.platform, model=tw.model,
                has_artifacts=tw.has_artifacts, artifact_count=tw.artifact_count,
                has_thinking=tw.has_thinking, exchange_index=tw.exchange_index,
                parent_tile_id=tw.parent_tile_id,
            )
            while not (_SHUTDOWN or _FATAL):
                try:
                    store_queue.put(et, timeout=2.0)
                    break
                except queue.Full:
                    continue

        with stats_lock:
            stats["embedded"] += len(batch)


# =============================================================================
# STORE WORKERS (dual-write: Weaviate + Neo4j)
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
            props = {
                "content": et.tile.text,
                "source_type": et.source_type,
                "source_file": et.source_file,
                "scale": et.tile.scale,
                "tile_index": et.tile.index,
                "start_char": et.tile.start_char,
                "end_char": et.tile.end_char,
                "token_count": et.tile.estimated_tokens,
                "content_hash": et.content_hash,
                "loaded_at": now,
                "actor": "unified_ingest_v1",
            }

            if et.source_type == "document":
                props["layer"] = LAYER_INT.get(et.stage, 2)
                props["platform"] = "corpus"
                props["document_id"] = et.document_id
                props["timestamp"] = et.timestamp or now
                props["priority"] = et.priority
            else:
                props["layer"] = 2  # transcripts are layer 2
                props["platform"] = et.platform
                props["session_id"] = et.session_id
                props["conversation_id"] = et.conversation_id
                props["model"] = et.model or ""
                props["has_artifacts"] = et.has_artifacts
                props["artifact_count"] = et.artifact_count
                props["has_thinking"] = et.has_thinking
                props["exchange_index"] = et.exchange_index
                props["timestamp"] = et.timestamp or now

            if et.parent_tile_id:
                props["parent_tile_id"] = et.parent_tile_id

            objects.append({
                "class": "ISMA_Quantum",
                "id": uid,
                "properties": props,
                "vector": et.embedding,
            })

        # Write to Weaviate (with retry)
        r = None
        for attempt in range(5):
            try:
                session = get_session()
                r = session.post(f"{WEAVIATE_URL}/v1/batch/objects",
                                 json={"objects": objects}, timeout=60)
                if r.status_code in [200, 201]:
                    break
                print(f"  WARN [{name}] Weaviate {r.status_code} (attempt {attempt+1}/5): {r.text[:200]}", flush=True)
            except Exception as e:
                print(f"  WARN [{name}] Weaviate {type(e).__name__} (attempt {attempt+1}/5): {e}", flush=True)
                r = None
            if _SHUTDOWN or _FATAL:
                break
            time.sleep(min(5 * (attempt + 1), 30))

        if r is None or r.status_code not in [200, 201]:
            set_fatal(f"[{name}] Weaviate failed after 5 attempts")
            break

        stored = 0
        for obj_r in r.json():
            errs = obj_r.get("result", {}).get("errors")
            if errs:
                msgs = [e.get("message", "") for e in errs.get("error", [])]
                # Only log non-duplicate errors
                for m in msgs:
                    if "already exists" not in m:
                        print(f"  WARN [{name}]: {m[:200]}", flush=True)
            else:
                stored += 1

        with stats_lock:
            stats["stored"] += stored


# =============================================================================
# MANIFEST LOADING
# =============================================================================

def load_manifest() -> dict:
    """Load the dedup manifest built by build_dedup_manifest.py."""
    if not MANIFEST_FILE.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_FILE}", flush=True)
        print("Run build_dedup_manifest.py first.", flush=True)
        sys.exit(1)

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    t = manifest.get("transcript_exchanges", {})
    c = manifest.get("corpus_documents", {})
    print(f"Manifest loaded (built {manifest.get('built_at', '?')}):", flush=True)
    print(f"  Transcript files: {len(t.get('kept_files', {}))}", flush=True)
    print(f"  Unique exchanges: {t.get('unique_exchanges', 0)}", flush=True)
    print(f"  Unique corpus docs: {c.get('unique', 0)}", flush=True)
    return manifest


# =============================================================================
# PROGRESS
# =============================================================================

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_files": [], "last_stage": "", "last_run": ""}


def save_progress(progress: dict):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


# =============================================================================
# WEAVIATE SCHEMA
# =============================================================================

WEAVIATE_SCHEMA = {
    "class": "ISMA_Quantum",
    "vectorizer": "none",
    "properties": [
        {"name": "content",         "dataType": ["text"],    "tokenization": "word"},
        {"name": "source_type",     "dataType": ["text"],    "tokenization": "word"},
        {"name": "source_file",     "dataType": ["text"],    "tokenization": "word"},
        {"name": "layer",           "dataType": ["int"]},
        {"name": "platform",        "dataType": ["text"],    "tokenization": "word"},
        {"name": "session_id",      "dataType": ["text"],    "tokenization": "field"},
        {"name": "document_id",     "dataType": ["text"],    "tokenization": "field"},
        {"name": "scale",           "dataType": ["text"],    "tokenization": "word"},
        {"name": "parent_tile_id",  "dataType": ["text"],    "tokenization": "field"},
        {"name": "tile_index",      "dataType": ["int"]},
        {"name": "start_char",      "dataType": ["int"]},
        {"name": "end_char",        "dataType": ["int"]},
        {"name": "token_count",     "dataType": ["int"]},
        {"name": "content_hash",    "dataType": ["text"],    "tokenization": "field"},
        {"name": "timestamp",       "dataType": ["text"],    "tokenization": "field"},
        {"name": "loaded_at",       "dataType": ["text"],    "tokenization": "field"},
        {"name": "actor",           "dataType": ["text"],    "tokenization": "word"},
        {"name": "model",           "dataType": ["text"],    "tokenization": "word"},
        {"name": "has_artifacts",   "dataType": ["boolean"]},
        {"name": "artifact_count",  "dataType": ["int"]},
        {"name": "has_thinking",    "dataType": ["boolean"]},
        {"name": "conversation_id", "dataType": ["text"],    "tokenization": "field"},
        {"name": "priority",        "dataType": ["number"]},
        {"name": "exchange_index",  "dataType": ["int"]},
    ],
}


def wipe_and_recreate_weaviate():
    """Delete ISMA_Quantum class and recreate with clean schema.

    Weaviate's raft persistence can auto-restore deleted classes from commit log,
    so we delete and immediately recreate in quick succession.
    """
    print("Wiping Weaviate ISMA_Quantum...", flush=True)

    # Delete class
    try:
        r = requests.delete(f"{WEAVIATE_URL}/v1/schema/ISMA_Quantum", timeout=30)
        if r.status_code in (200, 404):
            print(f"  Deleted ISMA_Quantum (status {r.status_code})", flush=True)
        else:
            print(f"  Delete returned {r.status_code}: {r.text[:200]}", flush=True)
    except Exception as e:
        print(f"  Delete error (may be OK): {e}", flush=True)

    # Quick sleep then immediately recreate to beat raft restore
    time.sleep(1)

    # Recreate with retries (raft may auto-restore old schema)
    for attempt in range(5):
        try:
            r = requests.post(f"{WEAVIATE_URL}/v1/schema",
                              json=WEAVIATE_SCHEMA, timeout=30)
            if r.status_code == 200:
                # Verify property count
                v = requests.get(
                    f"{WEAVIATE_URL}/v1/schema/ISMA_Quantum", timeout=5)
                props = v.json().get("properties", [])
                print(f"  Created ISMA_Quantum with {len(props)} properties",
                      flush=True)
                return
            elif r.status_code == 422 and "already exists" in r.text:
                # Delete and retry
                requests.delete(
                    f"{WEAVIATE_URL}/v1/schema/ISMA_Quantum", timeout=30)
                time.sleep(1)
                continue
            else:
                print(f"  Create attempt {attempt+1}: {r.status_code}: "
                      f"{r.text[:200]}", flush=True)
        except Exception as e:
            print(f"  Create error attempt {attempt+1}: {e}", flush=True)
        time.sleep(2)

    print("  WARN: Could not recreate schema cleanly, proceeding anyway",
          flush=True)


def wipe_neo4j_isma():
    """Delete all ISMA-related nodes (preserve non-ISMA data)."""
    print("Wiping Neo4j ISMA nodes...", flush=True)
    driver = get_neo4j()
    with driver.session() as s:
        # Delete in batches to avoid memory pressure
        labels_to_delete = [
            "ISMAMessage", "ISMASession", "ISMAExchange",
            "ISMAEntity", "ISMAEvent", "ISMAMemory",
            "ISMAPlatform", "ISMA_Entity", "Document",
        ]
        for label in labels_to_delete:
            result = s.run(f"MATCH (n:{label}) DETACH DELETE n RETURN count(n) AS c")
            count = result.single()["c"]
            if count > 0:
                print(f"  Deleted {count} {label} nodes", flush=True)

    # Create indexes
    print("Creating Neo4j indexes...", flush=True)
    with driver.session() as s:
        indexes = [
            "CREATE INDEX isma_session_id IF NOT EXISTS FOR (s:ISMASession) ON (s.session_id)",
            "CREATE INDEX isma_exchange_hash IF NOT EXISTS FOR (e:ISMAExchange) ON (e.content_hash)",
            "CREATE INDEX isma_exchange_session IF NOT EXISTS FOR (e:ISMAExchange) ON (e.session_id)",
            "CREATE INDEX document_hash IF NOT EXISTS FOR (d:Document) ON (d.content_hash)",
        ]
        for idx in indexes:
            s.run(idx)
        print("  Indexes created", flush=True)


def clear_old_state():
    """Remove old progress/hash files."""
    old_files = [
        "transcript_hashes_v2.json", "reprocess_hashes.json",
        "reprocess_progress.json", "transcript_progress_v2.json",
        "corpus_progress.json", "nightly_state.json",
        "unified_progress.json",
    ]
    for fn in old_files:
        fp = PROGRESS_DIR / fn
        if fp.exists():
            fp.unlink()
            print(f"  Removed {fp}", flush=True)


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
        print(f"  Weaviate: FAIL or empty - {e}", flush=True)
        # Not fatal - might be after wipe

    try:
        driver = get_neo4j()
        with driver.session() as s:
            r = s.run("MATCH (n) RETURN count(n) AS c")
            count = r.single()["c"]
        print(f"  Neo4j: OK ({count} total nodes)", flush=True)
    except Exception as e:
        print(f"  Neo4j: FAIL - {e}", flush=True)
        ok = False

    # Check manifest
    if MANIFEST_FILE.exists():
        size_mb = MANIFEST_FILE.stat().st_size / (1024 * 1024)
        print(f"  Manifest: OK ({size_mb:.1f} MB)", flush=True)
    else:
        print(f"  Manifest: NOT FOUND - run build_dedup_manifest.py", flush=True)
        ok = False

    if ok:
        print("Infrastructure OK.", flush=True)
    return ok


# =============================================================================
# CORPUS PROCESSING
# =============================================================================

def process_corpus_stage(stage_name: str, stage_dir: Path, priority: float,
                         pattern: Optional[str], manifest: dict,
                         tile_queue_obj: queue.Queue,
                         stats: Dict, stats_lock: threading.Lock,
                         dry_run: bool = False, limit: int = 0):
    """Process a single corpus stage."""
    corpus_hashes = manifest.get("corpus_documents", {}).get("hashes", {})
    all_paths_map = manifest.get("corpus_documents", {}).get("all_paths", {})

    if not stage_dir.exists():
        print(f"  [{stage_name}] directory not found: {stage_dir}", flush=True)
        return

    # Discover files and filter by manifest
    count = 0
    processed = 0

    # Binary extensions to skip
    BINARY_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
                  '.mp4', '.mp3', '.wav', '.zip', '.tar', '.gz',
                  '.pdf', '.ico', '.svg', '.woff', '.ttf', '.eot',
                  '.tfam', '.tped', '.bed', '.bim', '.fam',
                  '.min.js', '.min.css'}

    # Filename patterns to skip (junk content)
    SKIP_PREFIXES = ('heartbeat_', 'CCM_heartbeat_')

    for fp in sorted(stage_dir.rglob("*")):
        if _SHUTDOWN or _FATAL:
            break
        if not fp.is_file():
            continue
        if limit and processed >= limit:
            break
        if fp.suffix.lower() in BINARY_EXT:
            continue
        if fp.name.startswith(SKIP_PREFIXES):
            continue
        try:
            if fp.stat().st_size > 512 * 1024 or fp.stat().st_size == 0:
                continue
        except OSError:
            continue

        try:
            content = fp.read_text(errors='replace')
        except OSError:
            continue

        if not content.strip():
            continue

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        count += 1

        # Check if this hash is in manifest (deduped)
        if content_hash not in corpus_hashes:
            continue

        # Check if this is the canonical path for this hash
        canonical = corpus_hashes[content_hash]
        if canonical["stage"] != stage_name:
            continue  # This hash belongs to a different (higher priority) stage

        file_size = len(content)
        try:
            mtime = datetime.fromtimestamp(fp.stat().st_mtime).isoformat()
        except OSError:
            mtime = datetime.now().isoformat()

        all_paths = all_paths_map.get(content_hash, [str(fp)])
        doc_uuid = str(uuid.uuid5(DOC_NS, content_hash))

        # Multi-scale tile
        tiles = multi_scale_tile(content, source_file=fp.name, layer=stage_name)
        tile_count = len(tiles)
        token_count = sum(t.estimated_tokens for t in tiles)

        # Compute parent_tile_id mapping (search_512 -> context_2048)
        parent_map = {}  # tile index -> parent tile UUID
        context_tiles = [t for t in tiles if t.scale == "context_2048"]
        for t in tiles:
            if t.scale == "search_512" and t.parent_index >= 0:
                parent_map[t.index] = tile_uuid(
                    content_hash, "context_2048", t.parent_index)

        # Neo4j document node
        if not dry_run:
            try:
                write_neo4j_document(
                    content_hash=content_hash, filename=fp.name,
                    all_paths=all_paths, stage=stage_name,
                    priority=priority, file_size=file_size,
                    mtime=mtime, tile_count=tile_count)
            except Exception as e:
                print(f"  WARN: Neo4j doc failed {fp.name}: {e}", flush=True)

        with stats_lock:
            stats["files"] += 1
            stats["tiles"] += tile_count
            stats["tokens"] += token_count
            stats["neo4j_docs"] += 1

        processed += 1
        if processed <= 3 or processed % 100 == 0:
            print(f"  [{stage_name}] {processed}: {fp.name} "
                  f"({tile_count} tiles, ~{token_count} tok)", flush=True)

        if dry_run:
            continue

        # Feed tiles to queue
        rel_path = str(fp)
        for t in tiles:
            if _SHUTDOWN or _FATAL:
                break
            parent_tid = ""
            if t.scale == "search_512":
                parent_tid = parent_map.get(t.index, "")

            work = TileWork(
                tile=t, content_hash=content_hash,
                source_type="document", source_file=rel_path,
                stage=stage_name, priority=priority,
                document_id=doc_uuid, timestamp=mtime,
                parent_tile_id=parent_tid,
            )
            while not (_SHUTDOWN or _FATAL):
                try:
                    tile_queue_obj.put(work, timeout=2.0)
                    break
                except queue.Full:
                    continue

    print(f"  [{stage_name}] scanned {count}, processed {processed}", flush=True)


# =============================================================================
# TRANSCRIPT PROCESSING
# =============================================================================

def process_transcripts(manifest: dict, tile_queue_obj: queue.Queue,
                        stats: Dict, stats_lock: threading.Lock,
                        platform_filter: str = None,
                        dry_run: bool = False, limit: int = 0):
    """Process all transcript files using manifest for dedup."""
    kept_files = manifest.get("transcript_exchanges", {}).get("kept_files", {})
    exchange_hashes_manifest = manifest.get("transcript_exchanges", {}).get("exchange_hashes", {})

    print(f"\n{'='*60}", flush=True)
    print(f"Stage: transcripts ({len(kept_files)} files from manifest)", flush=True)
    print(f"{'='*60}", flush=True)

    # Build set of canonical exchange hashes (the ones we should embed)
    canonical_hashes = set()
    for h, info in exchange_hashes_manifest.items():
        canonical_hashes.add(h)

    processed_files = 0
    processed_exchanges = 0
    sessions_created = 0

    for filepath, file_info in sorted(kept_files.items()):
        if _SHUTDOWN or _FATAL:
            break
        if limit and processed_files >= limit:
            break

        platform = file_info["platform"]
        conv_id = file_info["conv_id"]

        if platform_filter and platform != platform_filter:
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            continue

        session_id = make_session_uuid(conv_id)
        title = data.get('title', '') or ''
        created_at = data.get('created_at', '') or data.get('timestamp', '') or ''
        top_model = data.get('model', '') or ''

        # Normalize exchanges
        enriched = normalize_exchanges(data, filepath)
        if not enriched:
            continue

        # Create Neo4j session
        if not dry_run:
            try:
                write_neo4j_session(
                    session_id=session_id, platform=platform,
                    title=title, source_file=filepath,
                    exchange_count=len(enriched),
                    created_at=created_at, model=top_model)
                sessions_created += 1
            except Exception as e:
                print(f"  WARN: Neo4j session failed {filepath}: {e}", flush=True)

        processed_files += 1
        if processed_files <= 3 or processed_files % 200 == 0:
            print(f"  [transcripts/{platform}] {processed_files}: "
                  f"{Path(filepath).name} ({len(enriched)} exchanges)", flush=True)

        # Process each exchange
        for idx, ex in enumerate(enriched):
            if _SHUTDOWN or _FATAL:
                break

            ex_hash = hash_exchange(ex.user_text, ex.assistant_text)

            # Check if this exchange is in the canonical set
            if ex_hash not in canonical_hashes:
                continue

            # Check if we are the canonical file for this hash
            canonical_info = exchange_hashes_manifest.get(ex_hash)
            if canonical_info and canonical_info.get("file") != filepath:
                # Another file owns this exchange - skip to avoid double-embed
                continue

            # Create Neo4j exchange
            if not dry_run:
                try:
                    write_neo4j_exchange(
                        content_hash=ex_hash, session_id=session_id,
                        index=idx, user_prompt=ex.user_text,
                        response=ex.assistant_text,
                        timestamp=ex.timestamp, model=ex.model)
                except Exception as e:
                    if processed_exchanges < 5:
                        print(f"  WARN: Neo4j exchange failed: {e}", flush=True)

            # Format and tile
            text = format_exchange(ex, conv_id, platform)
            tiles = multi_scale_tile(text, source_file=Path(filepath).name,
                                     layer="transcript")
            tile_count = len(tiles)
            token_count = sum(t.estimated_tokens for t in tiles)

            # Compute parent_tile_id mapping
            parent_map = {}
            for t in tiles:
                if t.scale == "search_512" and t.parent_index >= 0:
                    parent_map[t.index] = tile_uuid(
                        ex_hash, "context_2048", t.parent_index)

            with stats_lock:
                stats["exchanges"] += 1
                stats["tiles"] += tile_count
                stats["tokens"] += token_count

            processed_exchanges += 1

            if dry_run:
                continue

            # Feed tiles to queue
            for t in tiles:
                if _SHUTDOWN or _FATAL:
                    break
                parent_tid = ""
                if t.scale == "search_512":
                    parent_tid = parent_map.get(t.index, "")

                work = TileWork(
                    tile=t, content_hash=ex_hash,
                    source_type="transcript", source_file=filepath,
                    conversation_id=conv_id, session_id=session_id,
                    platform=platform, model=ex.model or top_model,
                    has_artifacts=ex.has_artifacts,
                    artifact_count=ex.artifact_count,
                    has_thinking=ex.has_thinking,
                    exchange_index=idx,
                    timestamp=ex.timestamp,
                    parent_tile_id=parent_tid,
                )
                while not (_SHUTDOWN or _FATAL):
                    try:
                        tile_queue_obj.put(work, timeout=2.0)
                        break
                    except queue.Full:
                        continue

    with stats_lock:
        stats["files"] += processed_files
        stats["neo4j_sessions"] = sessions_created

    print(f"  [transcripts] files={processed_files}, "
          f"exchanges={processed_exchanges}, sessions={sessions_created}", flush=True)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(stage_filter: str = None, platform_filter: str = None,
                 dry_run: bool = False, limit: int = 0,
                 wipe: bool = False):
    """Run the full unified ingest pipeline."""
    manifest = load_manifest()

    if wipe:
        wipe_and_recreate_weaviate()
        wipe_neo4j_isma()
        clear_old_state()
        print("Stores wiped and recreated.\n", flush=True)

    if not check_infra():
        if not wipe:
            print("Infrastructure check failed. Use --wipe for fresh start.", flush=True)
            sys.exit(1)

    stats = {
        "files": 0, "tiles": 0, "tokens": 0, "exchanges": 0,
        "embedded": 0, "stored": 0, "neo4j_docs": 0, "neo4j_sessions": 0,
    }
    stats_lock = threading.Lock()
    start_time = time.time()

    tile_queue_obj = queue.Queue(maxsize=TILE_QUEUE_SIZE)
    store_queue_obj = queue.Queue(maxsize=STORE_QUEUE_SIZE)
    threads = []

    if not dry_run:
        for i in range(EMBED_WORKERS):
            t = threading.Thread(target=embed_worker, name=f"embed-{i}",
                                 args=(tile_queue_obj, store_queue_obj,
                                       stats, stats_lock), daemon=True)
            t.start()
            threads.append(('embed', t))

        for i in range(STORE_WORKERS):
            t = threading.Thread(target=store_worker, name=f"store-{i}",
                                 args=(store_queue_obj, stats, stats_lock),
                                 daemon=True)
            t.start()
            threads.append(('store', t))

        print(f"\nPipeline: reader → tile_queue({TILE_QUEUE_SIZE}) → "
              f"{EMBED_WORKERS} embed (batch={EMBED_BATCH}) → "
              f"store_queue({STORE_QUEUE_SIZE}) → "
              f"{STORE_WORKERS} store (dual-write)\n", flush=True)

    # Start progress reporter
    progress_stop = threading.Event()

    def progress_reporter():
        while not progress_stop.is_set():
            progress_stop.wait(10.0)
            if progress_stop.is_set():
                break
            with stats_lock:
                snap = dict(stats)
            elapsed = time.time() - start_time
            tok_rate = snap["tokens"] / elapsed if elapsed > 0 else 0
            tq = tile_queue_obj.qsize()
            sq = store_queue_obj.qsize()
            print(f"  [{elapsed/60:.1f}m] files={snap['files']} "
                  f"tiles={snap['tiles']} emb={snap['embedded']} "
                  f"stored={snap['stored']} tq={tq} sq={sq} "
                  f"~{tok_rate:.0f} tok/s", flush=True)

    if not dry_run:
        reporter = threading.Thread(target=progress_reporter, daemon=True)
        reporter.start()

    # Process corpus stages
    for stage_name, stage_dir, priority, pattern in STAGES:
        if _SHUTDOWN or _FATAL:
            break

        if stage_name == "transcripts":
            # Handle transcripts separately
            if stage_filter and stage_filter != "transcripts":
                continue
            process_transcripts(manifest, tile_queue_obj, stats, stats_lock,
                               platform_filter=platform_filter,
                               dry_run=dry_run, limit=limit)
        else:
            if stage_filter and stage_filter != stage_name:
                continue
            print(f"\n{'='*60}", flush=True)
            print(f"Stage: {stage_name} (priority={priority})", flush=True)
            print(f"{'='*60}", flush=True)
            process_corpus_stage(stage_name, stage_dir, priority, pattern,
                                manifest, tile_queue_obj, stats, stats_lock,
                                dry_run=dry_run, limit=limit)

    # Drain pipeline
    if not dry_run:
        print("\nDraining pipeline...", flush=True)

        # Send poison to embed workers (they'll finish current batch first)
        for _ in range(EMBED_WORKERS + 1):
            tile_queue_obj.put(_POISON)

        # Wait for embed workers to finish
        for role, t in threads:
            if role == 'embed':
                t.join(timeout=300)
                if t.is_alive():
                    print(f"  WARN: {t.name} still alive after 300s", flush=True)

        # Send poison to store workers
        for _ in range(STORE_WORKERS + 1):
            store_queue_obj.put(_POISON)

        # Wait for store workers to finish
        for role, t in threads:
            if role == 'store':
                t.join(timeout=120)
                if t.is_alive():
                    print(f"  WARN: {t.name} still alive after 120s", flush=True)

        progress_stop.set()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}", flush=True)
    print(f"{'DRY RUN ' if dry_run else ''}COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Files: {stats['files']}", flush=True)
    print(f"  Exchanges: {stats['exchanges']}", flush=True)
    print(f"  Tiles: {stats['tiles']}", flush=True)
    print(f"  Tokens: {stats['tokens']:,}", flush=True)
    if not dry_run:
        print(f"  Embedded: {stats['embedded']}", flush=True)
        print(f"  Stored (Weaviate): {stats['stored']}", flush=True)
    print(f"  Neo4j Documents: {stats['neo4j_docs']}", flush=True)
    print(f"  Neo4j Sessions: {stats.get('neo4j_sessions', 0)}", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    if elapsed > 0 and stats['tokens'] > 0:
        print(f"  Throughput: {stats['tokens']/elapsed:.0f} tok/s", flush=True)

    if _FATAL:
        print(f"\nFATAL: {_FATAL}", flush=True)
        sys.exit(1)


# =============================================================================
# INCREMENTAL MODE
# =============================================================================

def run_incremental():
    """Process only files not yet in stores. Requires manifest."""
    manifest = load_manifest()

    # Check what's already in Weaviate
    print("Checking existing data...", flush=True)
    try:
        r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={
            "query": "{ Aggregate { ISMA_Quantum { meta { count } } } }"
        }, timeout=5)
        wv_count = r.json()["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]
        print(f"  Weaviate: {wv_count:,} objects", flush=True)
    except Exception:
        wv_count = 0

    if wv_count == 0:
        print("  Weaviate empty - running full ingest instead.", flush=True)
        run_pipeline(wipe=False)
        return

    # For incremental, just run pipeline without wipe
    # Weaviate MERGE (same UUID) is idempotent
    # Neo4j MERGE is idempotent
    print("Running incremental (idempotent upserts)...", flush=True)
    run_pipeline(wipe=False)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISMA Unified Ingest Pipeline")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    parser.add_argument("--all", action="store_true", help="Full rebuild (wipes stores!)")
    parser.add_argument("--stage", type=str, help="Process single stage")
    parser.add_argument("--platform", type=str,
                        help="Filter transcripts by platform")
    parser.add_argument("--incremental", action="store_true",
                        help="Process new files only")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit files per stage")
    parser.add_argument("--wipe", action="store_true",
                        help="Wipe stores before ingest (implied by --all)")
    args = parser.parse_args()

    try:
        if args.check:
            check_infra()
        elif args.all:
            run_pipeline(wipe=True, dry_run=args.dry_run, limit=args.limit)
        elif args.stage:
            run_pipeline(stage_filter=args.stage,
                         platform_filter=args.platform,
                         dry_run=args.dry_run, limit=args.limit,
                         wipe=args.wipe)
        elif args.incremental:
            run_incremental()
        elif args.wipe:
            wipe_and_recreate_weaviate()
            wipe_neo4j_isma()
            clear_old_state()
            print("Stores wiped. Run --all to rebuild.", flush=True)
        else:
            parser.print_help()
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
