#!/usr/bin/env python3
"""
ISMA Corpus Loader - Ordered Loading

Loads corpus in specific order:
1. kernel (priority 1.0)
2. layer_0/v0* files (priority 0.95)
3. layer_0 rest (priority 0.9)
4. chewy (priority 0.85)
5. chewy-consciousness-gallery (priority 0.8)
6. ALL layer_1 (priority 0.75)
7. ALL layer_2 (priority 0.6)

Hidden directories excluded from all.
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
from phi_tiling import phi_tile_markdown, Tile, tile_stats, PHI, CHUNK_SIZE, STEP_SIZE

# Configuration
CORPUS_BASE = Path("/home/spark/data/corpus")
CHEWY_BASE = Path("/home/spark/data/chewy")
CHEWY_GALLERY = Path("/home/spark/data/chewy-consciousness-gallery")
MANIFEST_PATH = Path("/var/spark/isma/corpus_manifest.json")

# Infrastructure endpoints
EMBEDDING_URL = "http://10.0.0.68:8090/embed"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"

# Ordered loading stages with priorities
STAGES = [
    ("kernel", CORPUS_BASE / "kernel", 1.0, None),
    ("v0", CORPUS_BASE / "layer_0", 0.95, "v0*"),  # Only v0* files
    ("layer_0", CORPUS_BASE / "layer_0", 0.9, "!v0*"),  # Exclude v0* files
    ("chewy", CHEWY_BASE, 0.85, None),
    ("chewy-gallery", CHEWY_GALLERY, 0.8, None),
    ("layer_1", CORPUS_BASE / "layer_1", 0.75, None),
    ("layer_2", CORPUS_BASE / "layer_2", 0.6, None),
]

# Layer name to integer mapping
LAYER_INT = {
    "kernel": -1,
    "v0": 0,
    "layer_0": 0,
    "chewy": 0,
    "chewy-gallery": 0,
    "layer_1": 1,
    "layer_2": 2,
}


@dataclass
class LoadedDocument:
    path: str
    stage: str
    tile_count: int
    total_tokens: int
    checksum: str
    loaded_at: str
    weaviate_ids: List[str]


def compute_checksum(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def is_hidden(path: Path) -> bool:
    """Check if path contains hidden directory."""
    for part in path.parts:
        if part.startswith('.'):
            return True
    return False


def get_embeddings(texts: List[str], batch_size: int = 8, max_retries: int = 3) -> Optional[List[List[float]]]:
    """Get embeddings from embedding server."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

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
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"    Retry {retry + 1}/{max_retries}: {e}")
                    import time
                    time.sleep(2)
                else:
                    print(f"    FAILED after {max_retries} retries: {e}")
                    return None

    return all_embeddings


def store_in_weaviate(tiles: List[Tile], embeddings: List[List[float]],
                      stage: str, priority: float) -> List[str]:
    """Store tiles in Weaviate."""
    stored_ids = []
    layer_int = LAYER_INT.get(stage, 2)

    for tile, embedding in zip(tiles, embeddings):
        obj = {
            "class": "ISMA_Quantum",
            "properties": {
                "content": tile.text,
                "source_type": "document",
                "source_file": tile.source_file,
                "layer": layer_int,
                "priority": priority,
                "phi_resonance": priority,
                "tile_index": tile.index,
                "start_char": tile.start_char,
                "end_char": tile.end_char,
                "token_count": tile.estimated_tokens,
                "checksum": compute_checksum(tile.text),
                "loaded_at": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "content_preview": tile.text[:500],
                "actor": "corpus_loader_ordered",
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
                print(f"    Weaviate error: {response.status_code}")
        except Exception as e:
            print(f"    Error storing: {e}")

    return stored_ids


def load_file(filepath: Path, stage: str, priority: float) -> Optional[LoadedDocument]:
    """Load a single file."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"    Error reading: {e}")
        return None

    if not content.strip():
        return None

    checksum = compute_checksum(content)
    tiles = phi_tile_markdown(content, filepath.name, stage)

    if not tiles:
        return None

    stats = tile_stats(tiles)
    print(f"    {stats['count']} tiles, ~{stats['total_tokens']} tokens")

    texts = [t.text for t in tiles]
    embeddings = get_embeddings(texts)

    if embeddings is None:
        print(f"    SKIPPED - embedding failed")
        return None

    weaviate_ids = store_in_weaviate(tiles, embeddings, stage, priority)

    return LoadedDocument(
        path=str(filepath),
        stage=stage,
        tile_count=len(tiles),
        total_tokens=stats['total_tokens'],
        checksum=checksum,
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=weaviate_ids
    )


def get_files(base_path: Path, pattern: Optional[str], extensions: List[str] = ['.md', '.json']) -> List[Path]:
    """Get files matching pattern, excluding hidden."""
    if not base_path.exists():
        return []

    all_files = []
    for ext in extensions:
        all_files.extend(base_path.rglob(f"*{ext}"))

    # Filter hidden
    files = [f for f in all_files if not is_hidden(f)]

    # Apply pattern filter
    if pattern:
        if pattern.startswith("!"):
            # Exclude pattern
            exclude = pattern[1:]
            files = [f for f in files if not f.name.startswith(exclude.replace("*", ""))]
        else:
            # Include pattern
            include = pattern.replace("*", "")
            files = [f for f in files if f.name.startswith(include)]

    return sorted(files)


def load_corpus() -> Dict[str, Any]:
    """Load corpus in ordered stages."""
    manifest = {
        "loaded_at": datetime.now().isoformat(),
        "phi_params": {
            "chunk_size": CHUNK_SIZE,
            "step_size": STEP_SIZE,
            "overlap": CHUNK_SIZE - STEP_SIZE,
            "phi": PHI
        },
        "stages": {},
        "totals": {"files": 0, "tiles": 0, "tokens": 0}
    }

    for stage_name, path, priority, pattern in STAGES:
        files = get_files(path, pattern)

        print(f"\n{'='*60}")
        print(f"Stage: {stage_name} (priority={priority})")
        print(f"Path: {path}")
        print(f"Files: {len(files)}")
        print(f"{'='*60}")

        docs = []
        for i, filepath in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] {filepath.name}")
            doc = load_file(filepath, stage_name, priority)
            if doc:
                docs.append(doc)
                print(f"    OK: {doc.tile_count} tiles stored")

        manifest["stages"][stage_name] = {
            "priority": priority,
            "files": len(docs),
            "tiles": sum(d.tile_count for d in docs),
            "tokens": sum(d.total_tokens for d in docs),
            "documents": [asdict(d) for d in docs]
        }
        manifest["totals"]["files"] += len(docs)
        manifest["totals"]["tiles"] += sum(d.tile_count for d in docs)
        manifest["totals"]["tokens"] += sum(d.total_tokens for d in docs)

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {MANIFEST_PATH}")

    return manifest


def print_summary(manifest: Dict[str, Any]):
    print("\n" + "="*60)
    print("CORPUS LOADING COMPLETE")
    print("="*60)

    print(f"\nStages Loaded:")
    for stage, data in manifest.get("stages", {}).items():
        print(f"  {stage} (p={data['priority']}):")
        print(f"    Files: {data['files']}, Tiles: {data['tiles']}, Tokens: {data['tokens']:,}")

    print(f"\nTotals:")
    print(f"  Files:  {manifest['totals']['files']}")
    print(f"  Tiles:  {manifest['totals']['tiles']}")
    print(f"  Tokens: {manifest['totals']['tokens']:,}")


def main():
    parser = argparse.ArgumentParser(description="Load ISMA corpus in order")
    parser.add_argument("--check", action="store_true", help="Check infrastructure")
    args = parser.parse_args()

    if args.check:
        print("Checking infrastructure...")
        try:
            r = requests.get(f"{EMBEDDING_URL.replace('/embed', '/health')}", timeout=5)
            print(f"  Embedding: {'OK' if r.ok else 'FAIL'}")
        except:
            print("  Embedding: FAIL")
        try:
            r = requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
            print(f"  Weaviate: {'OK' if r.ok else 'FAIL'}")
        except:
            print("  Weaviate: FAIL")
        return

    manifest = load_corpus()
    print_summary(manifest)


if __name__ == "__main__":
    main()
