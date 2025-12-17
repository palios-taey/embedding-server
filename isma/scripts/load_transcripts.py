#!/usr/bin/env python3
"""
ISMA Transcript Loader

Loads converted transcripts into ISMA with proper φ-tiling.
Uses e-based chunking (4096 chunk, 1507 step, 2589 overlap).

Each exchange becomes a unit that gets tiled if too large.
Preserves session context and exchange relationships.
"""

import os
import sys
import json
import hashlib
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from phi_tiling import phi_tile_text, Tile, tile_stats, E, CHUNK_SIZE, STEP_SIZE

# Configuration
TRANSCRIPTS_DIR = Path("/home/spark/builder-taey/converted_transcripts")
MANIFEST_PATH = Path("/var/spark/isma/transcript_manifest.json")

# Infrastructure endpoints
EMBEDDING_URL = "http://10.0.0.68:8090/embed"
WEAVIATE_URL = "http://10.0.0.68:8088"
NEO4J_URI = "bolt://10.0.0.68:7689"


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


def get_embeddings(texts: List[str], batch_size: int = 8, max_retries: int = 3) -> Optional[List[List[float]]]:
    """Get embeddings from embedding server. Returns None on failure."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        success = False

        for retry in range(max_retries):
            try:
                response = requests.post(
                    EMBEDDING_URL,
                    json={"texts": batch, "batch_size": batch_size},
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                all_embeddings.extend(result["embeddings"])
                success = True
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  Retry {retry + 1}/{max_retries} for batch {i}: {e}")
                    import time
                    time.sleep(2)
                else:
                    print(f"  FAILED batch {i} after {max_retries} retries: {e}")
                    return None

        if not success:
            return None

    return all_embeddings


def format_exchange_for_embedding(exchange: Dict[str, Any], session_id: str) -> str:
    """Format an exchange for embedding - combines prompt and response."""
    parts = []

    # Add context
    parts.append(f"[Session: {session_id}]")

    # Add user prompt
    user_prompt = exchange.get("user_prompt", "")
    if user_prompt:
        parts.append(f"\n[User]: {user_prompt}")

    # Add responses
    responses = exchange.get("responses", [])
    for resp in responses:
        text = resp.get("text", "")
        if text:
            parts.append(f"\n[Assistant]: {text}")

        # Include tool usage summary (not full details)
        tools = resp.get("tools", [])
        if tools:
            tool_types = [t.get("type", "unknown") for t in tools]
            parts.append(f"\n[Tools used: {', '.join(tool_types)}]")

    return "\n".join(parts)


def store_in_weaviate(tiles: List[Tile], embeddings: List[List[float]],
                       session_id: str, exchange_idx: int) -> List[str]:
    """Store tiles in Weaviate ISMA_Quantum collection."""
    stored_ids = []

    for tile, embedding in zip(tiles, embeddings):
        obj = {
            "class": "ISMA_Quantum",
            "properties": {
                "content": tile.text,  # Full tile content
                "source_type": "transcript",
                "source_file": tile.source_file,
                "layer": 2,  # Transcripts are application layer
                "priority": 0.5,  # Medium priority
                "phi_resonance": 0.5,
                "tile_index": tile.index,
                "start_char": tile.start_char,
                "end_char": tile.end_char,
                "token_count": tile.estimated_tokens,
                "checksum": compute_checksum(tile.text),
                "loaded_at": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "content_preview": tile.text[:500],
                "actor": "transcript_loader",
                "session_id": session_id,
                "exchange_index": exchange_idx,
            },
            "vector": embedding
        }

        try:
            response = requests.post(
                f"{WEAVIATE_URL}/v1/objects",
                json=obj,
                timeout=30
            )
            if response.status_code in [200, 201]:
                result = response.json()
                stored_ids.append(result.get("id", "unknown"))
            else:
                print(f"  Weaviate error: {response.status_code}")
        except Exception as e:
            print(f"  Error storing tile: {e}")

    return stored_ids


def load_session(filepath: Path) -> Optional[LoadedSession]:
    """Load a single transcript session into ISMA."""
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

    all_tiles = []
    all_embeddings = []
    all_weaviate_ids = []
    total_tokens = 0

    # Process each exchange
    for idx, exchange in enumerate(exchanges):
        # Format exchange for embedding
        text = format_exchange_for_embedding(exchange, session_id)
        if not text.strip():
            continue

        # Tile the exchange
        tiles = phi_tile_text(text, filepath.name, "transcript")
        if not tiles:
            continue

        # Get embeddings
        texts = [t.text for t in tiles]
        embeddings = get_embeddings(texts)

        if embeddings is None:
            print(f"  Skipping exchange {idx} - embedding failed")
            continue

        # Store in Weaviate
        weaviate_ids = store_in_weaviate(tiles, embeddings, session_id, idx)

        all_tiles.extend(tiles)
        all_embeddings.extend(embeddings)
        all_weaviate_ids.extend(weaviate_ids)
        total_tokens += sum(t.estimated_tokens for t in tiles)

    if not all_tiles:
        print(f"  No tiles generated for session: {session_id}")
        return None

    return LoadedSession(
        session_id=session_id,
        source_file=str(filepath),
        exchange_count=len(exchanges),
        tile_count=len(all_tiles),
        total_tokens=total_tokens,
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=all_weaviate_ids
    )


def load_transcripts(limit: int = None, start_from: int = 0) -> Dict[str, Any]:
    """
    Load transcript sessions into ISMA.

    Args:
        limit: Max sessions to load (None = all)
        start_from: Skip this many sessions (for resuming)
    """
    manifest = {
        "loaded_at": datetime.now().isoformat(),
        "phi_params": {
            "chunk_size": CHUNK_SIZE,
            "step_size": STEP_SIZE,
            "overlap": CHUNK_SIZE - STEP_SIZE,
            "e": E
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
    files = sorted(TRANSCRIPTS_DIR.glob("*.json"))

    if start_from > 0:
        files = files[start_from:]
        print(f"Resuming from session {start_from}")

    if limit:
        files = files[:limit]

    print(f"\n{'='*60}")
    print(f"Loading {len(files)} transcript sessions")
    print(f"φ-resonance constant: e = {E:.6f}")
    print(f"{'='*60}")

    for i, filepath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {filepath.name}")

        session = load_session(filepath)
        if session:
            manifest["sessions"].append(asdict(session))
            manifest["totals"]["sessions"] += 1
            manifest["totals"]["exchanges"] += session.exchange_count
            manifest["totals"]["tiles"] += session.tile_count
            manifest["totals"]["tokens"] += session.total_tokens
            print(f"  OK: {session.exchange_count} exchanges, {session.tile_count} tiles")

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {MANIFEST_PATH}")

    return manifest


def print_summary(manifest: Dict[str, Any]):
    """Print loading summary."""
    print("\n" + "="*60)
    print("TRANSCRIPT LOADING COMPLETE")
    print("="*60)
    print(f"\nφ-Tiling Parameters (e-based):")
    print(f"  Chunk: {manifest['phi_params']['chunk_size']} tokens")
    print(f"  Step:  {manifest['phi_params']['step_size']} tokens")
    print(f"  Overlap: {manifest['phi_params']['overlap']} tokens")
    print(f"  e: {manifest['phi_params']['e']:.6f}")

    print(f"\nTotals:")
    print(f"  Sessions:  {manifest['totals']['sessions']}")
    print(f"  Exchanges: {manifest['totals']['exchanges']}")
    print(f"  Tiles:     {manifest['totals']['tiles']}")
    print(f"  Tokens:    {manifest['totals']['tokens']:,}")


def main():
    parser = argparse.ArgumentParser(description="Load ISMA transcripts")
    parser.add_argument("--limit", type=int, help="Max sessions to load")
    parser.add_argument("--start", type=int, default=0, help="Start from session N")
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
        return

    manifest = load_transcripts(limit=args.limit, start_from=args.start)
    print_summary(manifest)


if __name__ == "__main__":
    main()
