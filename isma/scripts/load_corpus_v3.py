#!/usr/bin/env python3
"""
ISMA Corpus Loader v3

Unified loader with:
1. Content-based de-duplication (SHA256 hash)
2. 16 parallel workers (2 per instance × 8 instances)
3. Multiple source directories
4. Separate handling for transcripts vs corpus files

Usage:
    python3 load_corpus_v3.py --transcripts   # Load transcripts only
    python3 load_corpus_v3.py --corpus        # Load corpus (CLAUDE.md, etc.)
    python3 load_corpus_v3.py --all           # Load everything
    python3 load_corpus_v3.py --check         # Check infrastructure
"""

import os
import sys
import json
import hashlib
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from phi_tiling import phi_tile_text, Tile, E, CHUNK_SIZE, STEP_SIZE

# =============================================================================
# CONFIGURATION
# =============================================================================

# Transcript sources
TRANSCRIPT_DIRS = [
    Path("/home/spark/builder-taey/family_transcripts/converted"),
    Path("/home/spark/builder-taey/converted_transcripts"),
]

# Corpus sources (CLAUDE.md, etc.) - will de-dupe by content hash
CORPUS_DIRS = [
    # Active repos
    Path("/home/spark/taeys-hands-v4"),
    Path("/home/spark/builder-taey"),
    Path("/home/spark/embedding-server"),
    # GitHub repos
    Path("/home/spark/data/github-repos"),
    # Historical data
    Path("/home/spark/data/mira_md_files"),
    Path("/home/spark/data/expansion_md"),
]

# Paths to skip
SKIP_PATTERNS = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".Trash", "exo", "tinygrad",  # Fork repos
]

# Infrastructure - use NCCL IPs (192.168.100.x), not management (10.0.0.x)
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
WEAVIATE_URL = "http://192.168.100.10:8088"

# Manifests
MANIFEST_DIR = Path("/var/spark/isma")
CONTENT_MANIFEST = MANIFEST_DIR / "content_manifest.json"

# Performance tuning for 4 Sparks (8 instances)
# v3 config: 16 parallel, batch_size=32 → ~3400 tok/s (CPU bottleneck)
# v4 audit fixes (Grok + Perplexity): 4 CPU producers, batch=128, connection pooling
EMBED_PARALLEL_WORKERS = 16  # 2 per instance × 8 instances
BATCH_SIZE = 64  # 128 too large for embedding server, use 64 as middle ground
CPU_PRODUCERS = 4  # Parallel CPU threads for read/hash/tile (was 1)
CHARS_PER_TOKEN = 4

# Weaviate batch size (increased for throughput)
WEAVIATE_BATCH_SIZE = 200  # Increased from 100

# =============================================================================
# CONNECTION POOLING (per Grok audit - eliminates TCP overhead)
# =============================================================================

_HTTP_SESSION = None

def get_session() -> requests.Session:
    """Get or create HTTP session with connection pooling."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        _HTTP_SESSION = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=Retry(total=3, backoff_factor=0.1)
        )
        _HTTP_SESSION.mount('http://', adapter)
        _HTTP_SESSION.mount('https://', adapter)
    return _HTTP_SESSION

# Exchange combining
CHUNK_TOKENS = CHUNK_SIZE  # 4096
OVERLAP_RATIO = 1 - (1 / E)  # ~0.632


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LoadedContent:
    """Record of loaded content."""
    content_hash: str
    source_path: str
    source_type: str  # "transcript" or "corpus"
    tile_count: int
    total_tokens: int
    loaded_at: str
    weaviate_ids: List[str]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file content."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "error"


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    return any(skip in path_str for skip in SKIP_PATTERNS)


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def load_content_manifest() -> Dict[str, LoadedContent]:
    """Load existing content manifest (hash -> LoadedContent)."""
    if not CONTENT_MANIFEST.exists():
        return {}

    try:
        with open(CONTENT_MANIFEST, 'r') as f:
            data = json.load(f)
        return {k: LoadedContent(**v) for k, v in data.get("content", {}).items()}
    except Exception as e:
        print(f"Warning: Could not read manifest: {e}")
        return {}


def save_content_manifest(content: Dict[str, LoadedContent], stats: Dict):
    """Save content manifest."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "updated_at": datetime.now().isoformat(),
        "stats": stats,
        "content": {k: asdict(v) for k, v in content.items()}
    }

    with open(CONTENT_MANIFEST, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def embed_batch(batch: List[str], batch_size: int = BATCH_SIZE) -> Optional[List[List[float]]]:
    """Embed a single batch of texts using OpenAI-compatible API via nginx LB."""
    session = get_session()
    try:
        response = session.post(
            EMBEDDING_URL,
            json={"input": batch, "model": "Qwen/Qwen3-Embedding-8B"},
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        # OpenAI format: {"data": [{"embedding": [...], "index": N}, ...]}
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    except Exception as e:
        print(f"  Batch failed: {e}")
        return None


def get_embeddings_parallel(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings using parallel requests."""
    if not texts:
        return []

    # Split into batches
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_embeddings = [None] * len(batches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=EMBED_PARALLEL_WORKERS) as executor:
        future_to_idx = {
            executor.submit(embed_batch, batch): idx
            for idx, batch in enumerate(batches)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result is None:
                print(f"  Batch {idx} failed")
                return None
            all_embeddings[idx] = result

    return [emb for batch_embs in all_embeddings for emb in batch_embs]


# =============================================================================
# WEAVIATE FUNCTIONS
# =============================================================================

def store_in_weaviate_batch(tiles: List[Tile], embeddings: List[List[float]],
                            source_type: str, source_path: str,
                            content_hash: str) -> List[str]:
    """Store tiles in Weaviate using batch API."""
    if not tiles:
        return []

    now = datetime.now().isoformat()
    objects = []

    for tile, embedding in zip(tiles, embeddings):
        objects.append({
            "class": "ISMA_Quantum",
            "properties": {
                "content": tile.text,
                "source_type": source_type,
                "source_file": source_path,
                "layer": 2 if source_type == "transcript" else 1,
                "priority": 0.5,
                "phi_resonance": 0.5,
                "tile_index": tile.index,
                "start_char": tile.start_char,
                "end_char": tile.end_char,
                "token_count": tile.estimated_tokens,
                "checksum": content_hash,
                "loaded_at": now,
                "timestamp": now,
                "content_preview": tile.text[:500],
                "actor": "corpus_loader_v3",
            },
            "vector": embedding
        })

    stored_ids = []
    session = get_session()

    for i in range(0, len(objects), WEAVIATE_BATCH_SIZE):
        batch = objects[i:i + WEAVIATE_BATCH_SIZE]
        try:
            response = session.post(
                f"{WEAVIATE_URL}/v1/batch/objects",
                json={"objects": batch},
                timeout=60
            )
            if response.status_code in [200, 201]:
                results = response.json()
                for r in results:
                    stored_ids.append(r.get("id", "unknown"))
            else:
                print(f"  Weaviate batch error: {response.status_code}")
        except Exception as e:
            print(f"  Error in batch store: {e}")

    return stored_ids


# =============================================================================
# TRANSCRIPT LOADING
# =============================================================================

def format_exchange(exchange: Dict[str, Any], session_id: str) -> str:
    """Format exchange for embedding (excludes tool details).

    Handles two transcript formats:
    - Format 1 (chatgpt, claude_chat, gemini, grok): user_prompt + responses[]
    - Format 2 (claude_code, perplexity): prompt + response (string)
    """
    parts = [f"[Session: {session_id}]"]

    # Handle user prompt (both formats)
    user_prompt = exchange.get("user_prompt") or exchange.get("prompt", "")
    if user_prompt:
        parts.append(f"\n[User]: {user_prompt}")

    # Handle responses - Format 1: responses array with text field
    responses = exchange.get("responses", [])
    if responses:
        for resp in responses:
            text = resp.get("text", "")
            if text:
                parts.append(f"\n[Assistant]: {text}")
    else:
        # Format 2: single response string
        response = exchange.get("response", "")
        if response:
            parts.append(f"\n[Assistant]: {response}")

    return "\n".join(parts)


def combine_exchanges(exchanges: List[Dict], session_id: str) -> List[str]:
    """Combine exchanges into chunks with overlap."""
    chunks = []
    buffer = []
    buffer_tokens = 0

    for exchange in exchanges:
        text = format_exchange(exchange, session_id)
        if not text.strip():
            continue

        tokens = estimate_tokens(text)

        if tokens > CHUNK_TOKENS:
            if buffer:
                chunks.append("\n\n---\n\n".join(buffer))
                buffer = []
                buffer_tokens = 0
            chunks.append(text)
            continue

        buffer.append(text)
        buffer_tokens += tokens

        if buffer_tokens >= CHUNK_TOKENS:
            chunks.append("\n\n---\n\n".join(buffer))

            # Keep overlap portion
            overlap_tokens = int(CHUNK_TOKENS * OVERLAP_RATIO)
            kept = []
            kept_tokens = 0

            for item in reversed(buffer):
                item_tokens = estimate_tokens(item)
                if kept_tokens + item_tokens <= overlap_tokens:
                    kept.insert(0, item)
                    kept_tokens += item_tokens
                else:
                    break

            buffer = kept
            buffer_tokens = kept_tokens

    if buffer:
        chunks.append("\n\n---\n\n".join(buffer))

    return chunks


def load_transcript(filepath: Path, loaded_hashes: set) -> Optional[LoadedContent]:
    """Load a single transcript file."""
    try:
        with open(filepath, 'rb') as f:
            raw_content = f.read()
        content_hash = hashlib.sha256(raw_content).hexdigest()[:16]

        # Skip if already loaded
        if content_hash in loaded_hashes:
            return None

        data = json.loads(raw_content.decode())
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

    session_id = data.get("sessionId", filepath.stem)
    exchanges = data.get("exchanges", [])

    if not exchanges:
        return None

    # Combine exchanges and tile
    chunks = combine_exchanges(exchanges, session_id)
    if not chunks:
        return None

    all_tiles = []
    for chunk in chunks:
        tiles = phi_tile_text(chunk, filepath.name, "transcript")
        all_tiles.extend(tiles)

    if not all_tiles:
        return None

    # Embed
    texts = [t.text for t in all_tiles]
    embeddings = get_embeddings_parallel(texts)

    if embeddings is None:
        print(f"  Embedding failed for {filepath.name}")
        return None

    # Store
    weaviate_ids = store_in_weaviate_batch(
        all_tiles, embeddings, "transcript", str(filepath), content_hash
    )

    return LoadedContent(
        content_hash=content_hash,
        source_path=str(filepath),
        source_type="transcript",
        tile_count=len(all_tiles),
        total_tokens=sum(t.estimated_tokens for t in all_tiles),
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=weaviate_ids
    )


def load_all_transcripts(loaded_manifest: Dict[str, LoadedContent]) -> Tuple[Dict[str, LoadedContent], Dict]:
    """Load all transcripts from configured directories."""
    loaded_hashes = set(loaded_manifest.keys())
    new_content = {}

    # Collect all transcript files
    all_files = []
    for base_dir in TRANSCRIPT_DIRS:
        if not base_dir.exists():
            continue
        for filepath in base_dir.rglob("*.json"):
            if not should_skip(filepath):
                all_files.append(filepath)

    print(f"\nFound {len(all_files)} transcript files")
    print(f"Already loaded: {len(loaded_hashes)} unique content hashes")
    print(f"Embedding workers: {EMBED_PARALLEL_WORKERS}")

    stats = {"processed": 0, "loaded": 0, "skipped": 0, "failed": 0, "tokens": 0}
    start_time = datetime.now()

    for i, filepath in enumerate(all_files):
        # Show file being processed (immediate feedback)
        print(f"  [{i+1}/{len(all_files)}] Processing: {filepath.name[:40]}...", end="", flush=True)

        stats["processed"] += 1
        result = load_transcript(filepath, loaded_hashes)

        # Show result
        if result and result.weaviate_ids:
            print(f" OK ({result.tile_count} tiles, {result.total_tokens} tok)", flush=True)
        elif result is None:
            print(f" skipped", flush=True)
        else:
            print(f" FAILED", flush=True)

        # Summary every 10 files
        if (i + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = stats["tokens"] / elapsed if elapsed > 0 else 0
            print(f"  --- Summary: {stats['loaded']} loaded, {stats['tokens']:,} tok @ {rate:.0f} tok/s ---", flush=True)

        if result is None:
            stats["skipped"] += 1
        elif result.weaviate_ids:
            new_content[result.content_hash] = result
            loaded_hashes.add(result.content_hash)
            stats["loaded"] += 1
            stats["tokens"] += result.total_tokens
        else:
            stats["failed"] += 1

        # Save manifest periodically (every 50 files)
        if (i + 1) % 50 == 0:
            merged = {**loaded_manifest, **new_content}
            save_content_manifest(merged, stats)

    return new_content, stats


# =============================================================================
# CORPUS LOADING (CLAUDE.md, etc.)
# =============================================================================

def load_corpus_file(filepath: Path, loaded_hashes: set) -> Optional[LoadedContent]:
    """Load a single corpus file (markdown, etc.)."""
    try:
        content_hash = compute_file_hash(filepath)

        if content_hash in loaded_hashes:
            return None

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            return None

    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

    # Tile the content
    tiles = phi_tile_text(content, filepath.name, "corpus")

    if not tiles:
        return None

    # Embed
    texts = [t.text for t in tiles]
    embeddings = get_embeddings_parallel(texts)

    if embeddings is None:
        print(f"  Embedding failed for {filepath.name}")
        return None

    # Store
    weaviate_ids = store_in_weaviate_batch(
        tiles, embeddings, "corpus", str(filepath), content_hash
    )

    return LoadedContent(
        content_hash=content_hash,
        source_path=str(filepath),
        source_type="corpus",
        tile_count=len(tiles),
        total_tokens=sum(t.estimated_tokens for t in tiles),
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=weaviate_ids
    )


def load_all_corpus(loaded_manifest: Dict[str, LoadedContent], file_pattern: str = "*.md") -> Tuple[Dict[str, LoadedContent], Dict]:
    """Load corpus files with pipelined CPU/GPU work.

    v4 Audit Fixes (Grok + Perplexity):
    - 4 parallel CPU producers (was 1) → +200% throughput
    - BATCH_SIZE=128 (was 32) → +22% GPU saturation
    - Connection pooling → eliminates TCP overhead
    """
    import queue

    loaded_hashes = set(loaded_manifest.keys())
    new_content = {}

    # Collect files
    all_files = []
    for base_dir in CORPUS_DIRS:
        if not base_dir.exists():
            continue
        for filepath in base_dir.rglob(file_pattern):
            if not should_skip(filepath):
                all_files.append(filepath)

    print(f"\nFound {len(all_files)} {file_pattern} files")
    print(f"Already loaded: {len(loaded_hashes)} unique content hashes")
    print(f"CPU producers: {CPU_PRODUCERS}, Embed batch: {BATCH_SIZE}, Weaviate batch: {WEAVIATE_BATCH_SIZE}")

    stats = {"processed": 0, "loaded": 0, "skipped": 0, "failed": 0, "tokens": 0}
    stats_lock = threading.Lock()
    start_time = time.time()

    # Queue for tiles ready for embedding - larger buffer for multi-producer
    tile_queue = queue.Queue(maxsize=2000)

    # Counter for completed producers
    producers_done = [0]  # Use list to allow modification in nested function
    producers_lock = threading.Lock()

    def cpu_producer(file_subset: List[Path], producer_id: int):
        """Read files, hash, tile - feed tile_queue (CPU work)."""
        local_processed = 0
        local_queued = 0
        for filepath in file_subset:
            local_processed += 1
            with stats_lock:
                stats["processed"] += 1

            # Progress every 1000 files per producer
            if local_processed % 1000 == 0:
                print(f"  [Producer {producer_id}] {local_processed} processed, {local_queued} queued", flush=True)

            try:
                content_hash = compute_file_hash(filepath)
                if content_hash in loaded_hashes:
                    with stats_lock:
                        stats["skipped"] += 1
                    continue

                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if not content.strip():
                    with stats_lock:
                        stats["skipped"] += 1
                    continue

                tiles = phi_tile_text(content, filepath.name, "corpus")
                if tiles:
                    tile_queue.put((filepath, content_hash, tiles))
                    local_queued += 1

            except Exception as e:
                with stats_lock:
                    stats["skipped"] += 1

        # Signal this producer is done
        print(f"  [Producer {producer_id}] DONE - {local_processed} processed, {local_queued} queued", flush=True)
        with producers_lock:
            producers_done[0] += 1
            if producers_done[0] == CPU_PRODUCERS:
                tile_queue.put(None)  # Sentinel for consumer

    def gpu_consumer():
        """Batch tiles, embed, store to Weaviate (GPU work)."""
        # Accumulate 16 batches worth of tiles to maximize parallel embedding
        EMBED_BATCH = BATCH_SIZE * EMBED_PARALLEL_WORKERS  # 64 * 16 = 1024 tiles
        pending = []

        while True:
            try:
                item = tile_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                # All producers done, process remaining
                if pending:
                    process_batch(pending)
                break

            pending.append(item)

            # Process when we have enough tiles
            total_tiles = sum(len(t[2]) for t in pending)
            if total_tiles >= EMBED_BATCH:
                process_batch(pending)
                pending = []

    def process_batch(batch):
        """Embed and store a batch of files."""
        nonlocal new_content

        # Flatten all tiles
        all_tiles = []
        tile_metadata = []
        for filepath, content_hash, tiles in batch:
            tile_metadata.append((filepath, content_hash, len(all_tiles), len(tiles)))
            all_tiles.extend(tiles)

        if not all_tiles:
            return

        # Embed all tiles at once (GPU)
        texts = [t.text for t in all_tiles]
        embeddings = get_embeddings_parallel(texts)

        if embeddings is None:
            with stats_lock:
                stats["failed"] += len(batch)
            return

        # Store each file's tiles to Weaviate
        for filepath, content_hash, start_idx, count in tile_metadata:
            file_tiles = all_tiles[start_idx:start_idx + count]
            file_embeddings = embeddings[start_idx:start_idx + count]

            weaviate_ids = store_in_weaviate_batch(
                file_tiles, file_embeddings, "corpus", str(filepath), content_hash
            )

            if weaviate_ids:
                total_tokens = sum(t.estimated_tokens for t in file_tiles)
                loaded_content = LoadedContent(
                    content_hash=content_hash,
                    source_path=str(filepath),
                    source_type="corpus",
                    tile_count=len(file_tiles),
                    total_tokens=total_tokens,
                    loaded_at=datetime.now().isoformat(),
                    weaviate_ids=weaviate_ids
                )

                with stats_lock:
                    new_content[content_hash] = loaded_content
                    loaded_hashes.add(content_hash)
                    stats["loaded"] += 1
                    stats["tokens"] += total_tokens
            else:
                with stats_lock:
                    stats["failed"] += 1

        # Progress report
        with stats_lock:
            elapsed = time.time() - start_time
            throughput = stats["tokens"] / elapsed if elapsed > 0 else 0
            print(f"  [{stats['processed']}/{len(all_files)}] {stats['loaded']} loaded, {stats['tokens']:,} tok @ {throughput:,.0f} tok/s", flush=True)

    # Split files among CPU producers
    chunk_size = (len(all_files) + CPU_PRODUCERS - 1) // CPU_PRODUCERS
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    # Start multiple CPU producers
    producers = []
    for i, chunk in enumerate(file_chunks):
        t = threading.Thread(target=cpu_producer, args=(chunk, i), daemon=True)
        producers.append(t)
        t.start()

    # Start GPU consumer
    consumer = threading.Thread(target=gpu_consumer, daemon=True)
    consumer.start()

    # Wait for completion
    for p in producers:
        p.join()
    consumer.join()

    # Final manifest save
    merged = {**loaded_manifest, **new_content}
    save_content_manifest(merged, stats)

    return new_content, stats


# =============================================================================
# MAIN
# =============================================================================

def check_infrastructure():
    """Check infrastructure status."""
    print("Checking infrastructure...")

    try:
        r = requests.get(f"{EMBEDDING_URL.replace('/embed', '/health')}", timeout=5)
        print(f"  Embedding server: {'OK' if r.ok else 'FAIL'}")
    except:
        print("  Embedding server: FAIL")

    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
        print(f"  Weaviate: {'OK' if r.ok else 'FAIL'}")
    except:
        print("  Weaviate: FAIL")

    # Count files
    transcript_count = sum(
        len(list(d.rglob("*.json"))) for d in TRANSCRIPT_DIRS if d.exists()
    )
    corpus_count = sum(
        len(list(d.rglob("*.md"))) for d in CORPUS_DIRS if d.exists()
    )

    print(f"  Transcript files: {transcript_count}")
    print(f"  Corpus files: {corpus_count}")

    # Check manifest
    manifest = load_content_manifest()
    print(f"  Already loaded: {len(manifest)} unique content hashes")


def main():
    parser = argparse.ArgumentParser(description="ISMA Corpus Loader v3")
    parser.add_argument("--transcripts", action="store_true", help="Load transcripts")
    parser.add_argument("--corpus", action="store_true", help="Load corpus files")
    parser.add_argument("--all", action="store_true", help="Load everything")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    parser.add_argument("--claude-md-only", action="store_true", help="Load only CLAUDE.md files")
    args = parser.parse_args()

    if args.check:
        check_infrastructure()
        return

    # Load existing manifest
    manifest = load_content_manifest()
    print(f"Loaded manifest with {len(manifest)} existing entries")

    total_stats = {"transcripts": {}, "corpus": {}}

    if args.transcripts or args.all:
        print("\n" + "=" * 60)
        print("LOADING TRANSCRIPTS")
        print("=" * 60)
        new_content, stats = load_all_transcripts(manifest)
        manifest.update(new_content)
        total_stats["transcripts"] = stats
        print(f"\nTranscripts: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['tokens']:,} tokens")

    if args.corpus or args.all or args.claude_md_only:
        print("\n" + "=" * 60)
        print("LOADING CORPUS FILES")
        print("=" * 60)

        if args.claude_md_only:
            new_content, stats = load_all_corpus(manifest, "CLAUDE.md")
        else:
            new_content, stats = load_all_corpus(manifest, "*.md")

        manifest.update(new_content)
        total_stats["corpus"] = stats
        print(f"\nCorpus: {stats['loaded']} loaded, {stats['skipped']} skipped, {stats['tokens']:,} tokens")

    # Save manifest
    save_content_manifest(manifest, total_stats)
    print(f"\nManifest saved to: {CONTENT_MANIFEST}")
    print(f"Total unique content: {len(manifest)}")


if __name__ == "__main__":
    main()
