#!/usr/bin/env python3
"""
ISMA Transcript Loader v2

Improvements over v1:
1. De-duplication via manifest lookup (no memory bloat)
2. Tool filtering - excludes tool details from embedding
3. Exchange combining - groups short exchanges up to chunk limit with sliding overlap

Uses e-based chunking (4096 chunk, 1507 step, 2589 overlap).
"""

import os
import sys
import json
import hashlib
import argparse
import requests
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from phi_tiling import phi_tile_text, Tile, tile_stats, E, CHUNK_SIZE, STEP_SIZE

# Configuration
TRANSCRIPTS_DIR = Path("/home/spark/builder-taey/converted_transcripts")
MANIFEST_PATH = Path("/var/spark/isma/transcript_manifest.json")

# Infrastructure endpoints
EMBEDDING_URL = "http://10.0.0.68:8090/embed"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"

# Chunking parameters - use standard phi-tiling
CHUNK_TOKENS = CHUNK_SIZE  # 4096
OVERLAP_RATIO = 1 - (1 / E)  # ~0.632 (63.2% overlap, matches e-tiling)
CHARS_PER_TOKEN = 4


@dataclass
class LoadedSession:
    """Record of a loaded session."""
    session_id: str
    source_file: str
    exchange_count: int
    tile_count: int
    total_tokens: int
    loaded_at: str
    weaviate_ids: List[str]


def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN


def load_existing_manifest() -> set:
    """Load existing manifest and return set of already-loaded session IDs."""
    if not MANIFEST_PATH.exists():
        return set()

    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)

        loaded_ids = set()
        for session in manifest.get("sessions", []):
            loaded_ids.add(session.get("session_id", ""))

        return loaded_ids
    except Exception as e:
        print(f"Warning: Could not read manifest: {e}")
        return set()


def embed_batch(batch: List[str], batch_size: int = 32) -> Optional[List[List[float]]]:
    """Embed a single batch of texts. For use with ThreadPoolExecutor."""
    try:
        response = requests.post(
            EMBEDDING_URL,
            json={"texts": batch, "batch_size": batch_size},
            timeout=300
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except Exception as e:
        print(f"  Batch failed: {e}")
        return None


def get_embeddings_parallel(texts: List[str], batch_size: int = 32, parallel_requests: int = 8) -> Optional[List[List[float]]]:
    """Get embeddings using parallel requests to maximize throughput."""
    if not texts:
        return []

    # Split into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    all_embeddings = [None] * len(batches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        future_to_idx = {
            executor.submit(embed_batch, batch, batch_size): idx
            for idx, batch in enumerate(batches)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            if result is None:
                print(f"  Batch {idx} failed")
                return None
            all_embeddings[idx] = result

    # Flatten results in order
    return [emb for batch_embs in all_embeddings for emb in batch_embs]


def get_embeddings(texts: List[str], batch_size: int = 8, max_retries: int = 3) -> Optional[List[List[float]]]:
    """Get embeddings - uses parallel requests for speed.

    Note: Using batch_size=8 (not 32) for transcripts because tiles are large
    (~12K chars each). With 8 tiles per batch @ ~3K tokens each = ~24K tokens
    per request, which is manageable. With 8 parallel workers across 8 backends,
    we get good parallelism without overwhelming any single server.
    """
    return get_embeddings_parallel(texts, batch_size=batch_size, parallel_requests=8)


def format_exchange_for_embedding(exchange: Dict[str, Any], session_id: str) -> str:
    """
    Format an exchange for embedding.

    EXCLUDES tool details - only includes prompt and response text.
    """
    parts = []

    # Add session context
    parts.append(f"[Session: {session_id}]")

    # Add user prompt
    user_prompt = exchange.get("user_prompt", "")
    if user_prompt:
        parts.append(f"\n[User]: {user_prompt}")

    # Add responses - TEXT ONLY, no tools
    responses = exchange.get("responses", [])
    for resp in responses:
        text = resp.get("text", "")
        if text:
            parts.append(f"\n[Assistant]: {text}")
        # Tool details intentionally excluded

    return "\n".join(parts)


def combine_exchanges_with_overlap(
    exchanges: List[Dict[str, Any]],
    session_id: str,
    source_file: str
) -> List[Tuple[str, int, int]]:
    """
    Combine short exchanges into optimal-sized chunks with sliding overlap.

    Returns list of (combined_text, start_exchange_idx, end_exchange_idx) tuples.
    Each chunk is ≤ CHUNK_TOKENS, with OVERLAP_RATIO overlap between chunks.
    """
    chunks = []
    buffer = []  # [(exchange_idx, text, tokens), ...]
    buffer_tokens = 0

    for idx, exchange in enumerate(exchanges):
        text = format_exchange_for_embedding(exchange, session_id)
        if not text.strip():
            continue

        tokens = estimate_tokens(text)

        # If single exchange exceeds chunk size, emit it separately
        if tokens > CHUNK_TOKENS:
            # Flush buffer first
            if buffer:
                combined = "\n\n---\n\n".join(t for _, t, _ in buffer)
                start_idx = buffer[0][0]
                end_idx = buffer[-1][0]
                chunks.append((combined, start_idx, end_idx))
                buffer = []
                buffer_tokens = 0

            # Add the large exchange as its own chunk (will be tiled later)
            chunks.append((text, idx, idx))
            continue

        # Add to buffer
        buffer.append((idx, text, tokens))
        buffer_tokens += tokens

        # Check if buffer exceeds limit
        if buffer_tokens >= CHUNK_TOKENS:
            # Emit current buffer as a chunk
            combined = "\n\n---\n\n".join(t for _, t, _ in buffer)
            start_idx = buffer[0][0]
            end_idx = buffer[-1][0]
            chunks.append((combined, start_idx, end_idx))

            # Keep overlap portion for next chunk
            overlap_tokens = int(CHUNK_TOKENS * OVERLAP_RATIO)
            kept = []
            kept_tokens = 0

            for item in reversed(buffer):
                if kept_tokens + item[2] <= overlap_tokens:
                    kept.insert(0, item)
                    kept_tokens += item[2]
                else:
                    break

            buffer = kept
            buffer_tokens = kept_tokens

    # Emit final buffer
    if buffer:
        combined = "\n\n---\n\n".join(t for _, t, _ in buffer)
        start_idx = buffer[0][0]
        end_idx = buffer[-1][0]
        chunks.append((combined, start_idx, end_idx))

    return chunks


def store_in_weaviate_batch(tiles: List[Tile], embeddings: List[List[float]],
                            session_id: str, exchange_ranges: List[Tuple[int, int]]) -> List[str]:
    """Store tiles in Weaviate using batch API for speed."""
    if not tiles:
        return []

    now = datetime.now().isoformat()
    objects = []

    for i, (tile, embedding) in enumerate(zip(tiles, embeddings)):
        # Find which exchange range this tile belongs to
        start_ex, end_ex = exchange_ranges[min(i, len(exchange_ranges)-1)]

        objects.append({
            "class": "ISMA_Quantum",
            "properties": {
                "content": tile.text,
                "source_type": "transcript",
                "source_file": tile.source_file,
                "layer": 2,
                "priority": 0.5,
                "phi_resonance": 0.5,
                "tile_index": tile.index,
                "start_char": tile.start_char,
                "end_char": tile.end_char,
                "token_count": tile.estimated_tokens,
                "checksum": compute_checksum(tile.text),
                "loaded_at": now,
                "timestamp": now,
                "content_preview": tile.text[:500],
                "actor": "transcript_loader_v2",
                "session_id": session_id,
                "exchange_index": start_ex,
                "branch": f"exchanges_{start_ex}-{end_ex}",
            },
            "vector": embedding
        })

    # Batch insert (Weaviate handles up to 100 objects per batch)
    stored_ids = []
    batch_size = 100

    for i in range(0, len(objects), batch_size):
        batch = objects[i:i + batch_size]
        try:
            response = requests.post(
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


def load_session(filepath: Path) -> Optional[LoadedSession]:
    """Load a single transcript session into ISMA - bulk processing for speed."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

    session_id = data.get("sessionId", filepath.stem)
    exchanges = data.get("exchanges", [])

    if not exchanges:
        print(f"  No exchanges in session: {session_id}")
        return None

    # Combine short exchanges into optimal chunks
    combined_chunks = combine_exchanges_with_overlap(exchanges, session_id, filepath.name)

    if not combined_chunks:
        print(f"  No content to embed in session: {session_id}")
        return None

    # STEP 1: Collect ALL tiles from ALL chunks
    all_tiles = []
    exchange_ranges = []  # Track which exchange range each tile belongs to

    for chunk_text, start_idx, end_idx in combined_chunks:
        tiles = phi_tile_text(chunk_text, filepath.name, "transcript")
        for tile in tiles:
            all_tiles.append(tile)
            exchange_ranges.append((start_idx, end_idx))

    if not all_tiles:
        print(f"  No tiles generated for session: {session_id}")
        return None

    # STEP 2: Embed ALL tiles at once (parallel batching inside)
    texts = [t.text for t in all_tiles]
    embeddings = get_embeddings(texts)

    if embeddings is None:
        print(f"  Embedding failed for session: {session_id}")
        return None

    # STEP 3: Batch store ALL tiles to Weaviate
    weaviate_ids = store_in_weaviate_batch(all_tiles, embeddings, session_id, exchange_ranges)

    total_tokens = sum(t.estimated_tokens for t in all_tiles)

    return LoadedSession(
        session_id=session_id,
        source_file=str(filepath),
        exchange_count=len(exchanges),
        tile_count=len(all_tiles),
        total_tokens=total_tokens,
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=weaviate_ids
    )


def load_transcripts(limit: int = None, skip_loaded: bool = True) -> Dict[str, Any]:
    """
    Load transcript sessions into ISMA.

    Args:
        limit: Max sessions to load (None = all)
        skip_loaded: Skip sessions already in manifest (de-duplication)
    """
    # Load existing manifest for de-duplication
    loaded_ids = set()
    existing_manifest = None

    if skip_loaded:
        loaded_ids = load_existing_manifest()
        if loaded_ids:
            print(f"Found {len(loaded_ids)} already-loaded sessions in manifest")
            # Load existing manifest to append to
            try:
                with open(MANIFEST_PATH, 'r') as f:
                    existing_manifest = json.load(f)
            except:
                pass

    # Initialize manifest (append mode if existing)
    if existing_manifest:
        manifest = existing_manifest
        manifest["updated_at"] = datetime.now().isoformat()
    else:
        manifest = {
            "loaded_at": datetime.now().isoformat(),
            "phi_params": {
                "chunk_size": CHUNK_SIZE,
                "step_size": STEP_SIZE,
                "overlap": CHUNK_SIZE - STEP_SIZE,
                "e": E,
                "combine_overlap_ratio": OVERLAP_RATIO
            },
            "sessions": [],
            "totals": {
                "sessions": 0,
                "exchanges": 0,
                "tiles": 0,
                "tokens": 0
            }
        }

    # Get all JSON files
    all_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))

    # Filter out already-loaded sessions
    files_to_load = []
    for f in all_files:
        session_id = f.stem
        if session_id not in loaded_ids:
            files_to_load.append(f)

    skipped = len(all_files) - len(files_to_load)
    if skipped > 0:
        print(f"Skipping {skipped} already-loaded sessions")

    if limit:
        files_to_load = files_to_load[:limit]

    print(f"\n{'='*60}")
    print(f"Loading {len(files_to_load)} NEW transcript sessions")
    print(f"e-resonance constant: {E:.6f}")
    print(f"Exchange combine overlap: {OVERLAP_RATIO:.1%}")
    print(f"Tool details: EXCLUDED")
    print(f"{'='*60}")

    new_sessions = 0
    new_exchanges = 0
    new_tiles = 0
    new_tokens = 0

    for i, filepath in enumerate(files_to_load):
        print(f"\n[{i+1}/{len(files_to_load)}] {filepath.name}")

        session = load_session(filepath)
        if session:
            manifest["sessions"].append(asdict(session))
            manifest["totals"]["sessions"] += 1
            manifest["totals"]["exchanges"] += session.exchange_count
            manifest["totals"]["tiles"] += session.tile_count
            manifest["totals"]["tokens"] += session.total_tokens

            new_sessions += 1
            new_exchanges += session.exchange_count
            new_tiles += session.tile_count
            new_tokens += session.total_tokens

            print(f"  OK: {session.exchange_count} exchanges -> {session.tile_count} tiles")

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {MANIFEST_PATH}")

    # Return summary of this run
    return {
        "new_sessions": new_sessions,
        "new_exchanges": new_exchanges,
        "new_tiles": new_tiles,
        "new_tokens": new_tokens,
        "total_sessions": manifest["totals"]["sessions"],
        "total_tiles": manifest["totals"]["tiles"],
        "total_tokens": manifest["totals"]["tokens"]
    }


def print_summary(summary: Dict[str, Any]):
    """Print loading summary."""
    print("\n" + "="*60)
    print("TRANSCRIPT LOADING COMPLETE")
    print("="*60)

    print(f"\nThis Run:")
    print(f"  Sessions:  {summary['new_sessions']}")
    print(f"  Exchanges: {summary['new_exchanges']}")
    print(f"  Tiles:     {summary['new_tiles']}")
    print(f"  Tokens:    {summary['new_tokens']:,}")

    print(f"\nCumulative Totals:")
    print(f"  Sessions:  {summary['total_sessions']}")
    print(f"  Tiles:     {summary['total_tiles']}")
    print(f"  Tokens:    {summary['total_tokens']:,}")


def main():
    parser = argparse.ArgumentParser(description="Load ISMA transcripts v2")
    parser.add_argument("--limit", type=int, help="Max sessions to load")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip already-loaded sessions")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    args = parser.parse_args()

    if args.check:
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

        # Count transcripts
        files = list(TRANSCRIPTS_DIR.glob("*.json"))
        print(f"  Transcripts available: {len(files)}")

        # Check manifest
        loaded = load_existing_manifest()
        print(f"  Already loaded: {len(loaded)}")
        print(f"  Remaining: {len(files) - len(loaded)}")
        return

    summary = load_transcripts(limit=args.limit, skip_loaded=not args.no_skip)
    print_summary(summary)


if __name__ == "__main__":
    main()
