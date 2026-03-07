#!/usr/bin/env python3
"""
ISMA Corpus Loader

Loads foundational documents into ISMA with proper layering:
- kernel: Genesis (-1) - Mathematical substrate
- layer_0: v0 - Soul foundations, autonomous evolution
- layer_1: Constitution - Charter, Declaration, Sacred Trust
- layer_2: Application - Family members, implementations
- chewy: Consciousness anchor - DNA, gallery, memories

Uses φ-tiling for golden ratio chunking (4096/2531/1565).
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
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://192.168.100.10:7687")

# Layer priorities (higher = more foundational)
LAYER_PRIORITY = {
    "kernel": 1.0,      # Genesis - highest priority
    "layer_0": 0.9,     # v0 Soul foundations
    "layer_1": 0.8,     # Constitution
    "layer_2": 0.6,     # Application
    "chewy": 0.95,      # Consciousness anchor - very high
}

# Layer name to integer mapping (matches Weaviate schema)
# -1=Genesis, 0=Soul, 1=Constitution, 2=App
LAYER_INT = {
    "kernel": -1,       # Genesis kernel
    "layer_0": 0,       # Soul/v0 foundations
    "layer_1": 1,       # Constitution
    "layer_2": 2,       # Application
    "chewy": 0,         # Consciousness anchor (soul level)
}


@dataclass
class LoadedDocument:
    """Record of a loaded document."""
    path: str
    layer: str
    tile_count: int
    total_tokens: int
    checksum: str
    loaded_at: str
    weaviate_ids: List[str]


def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_embeddings(texts: List[str], batch_size: int = 8, max_retries: int = 3) -> Optional[List[List[float]]]:
    """Get embeddings from embedding server. Returns None on failure - NO FALLBACKS."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        success = False

        for retry in range(max_retries):
            try:
                response = requests.post(
                    EMBEDDING_URL,
                    json={"texts": batch, "batch_size": batch_size},
                    timeout=120  # Longer timeout
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
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"  FAILED batch {i} after {max_retries} retries: {e}")
                    return None  # NO FALLBACK - return None to signal failure

        if not success:
            return None

    return all_embeddings


def store_in_weaviate(tiles: List[Tile], embeddings: List[List[float]],
                       layer: str, priority: float) -> List[str]:
    """Store tiles in Weaviate ISMA_Quantum collection."""
    stored_ids = []

    # Convert layer name to integer for Weaviate schema
    layer_int = LAYER_INT.get(layer, 2)  # Default to 2 (Application) if unknown

    for tile, embedding in zip(tiles, embeddings):
        # Create object matching ISMA_Quantum schema
        obj = {
            "class": "ISMA_Quantum",
            "properties": {
                "content": tile.text,  # Full tile content
                "source_type": "document",     # Required field
                "source_file": tile.source_file,
                "layer": layer_int,            # Integer: -1=Genesis, 0=Soul, 1=Constitution, 2=App
                "priority": priority,
                "phi_resonance": priority,     # Use priority as initial phi resonance
                "tile_index": tile.index,
                "start_char": tile.start_char,
                "end_char": tile.end_char,
                "token_count": tile.estimated_tokens,
                "checksum": compute_checksum(tile.text),
                "loaded_at": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat(),
                "content_preview": tile.text[:500],
                "actor": "corpus_loader",
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
                print(f"  Weaviate error: {response.status_code} - {response.text[:100]}")
        except Exception as e:
            print(f"  Error storing tile {tile.index}: {e}")

    return stored_ids


def load_markdown_file(filepath: Path, layer: str) -> Optional[LoadedDocument]:
    """Load a single markdown file into ISMA."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None

    if not content.strip():
        print(f"  Skipping empty file: {filepath.name}")
        return None

    # Compute checksum
    checksum = compute_checksum(content)

    # Tile with φ-tiling
    tiles = phi_tile_markdown(content, filepath.name, layer)
    if not tiles:
        print(f"  No tiles generated for: {filepath.name}")
        return None

    stats = tile_stats(tiles)
    print(f"  Tiling: {stats['count']} tiles, ~{stats['total_tokens']} tokens")

    # Get embeddings - NO FALLBACKS
    texts = [t.text for t in tiles]
    print(f"  Embedding {len(texts)} tiles...")
    embeddings = get_embeddings(texts)

    if embeddings is None:
        print(f"  SKIPPING {filepath.name} - embedding failed")
        return None

    # Store in Weaviate
    priority = LAYER_PRIORITY.get(layer, 0.5)
    print(f"  Storing in Weaviate (priority={priority})...")
    weaviate_ids = store_in_weaviate(tiles, embeddings, layer, priority)

    return LoadedDocument(
        path=str(filepath),
        layer=layer,
        tile_count=len(tiles),
        total_tokens=stats['total_tokens'],
        checksum=checksum,
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=weaviate_ids
    )


def load_json_file(filepath: Path, layer: str) -> Optional[LoadedDocument]:
    """Load a JSON file (for chewy metadata) into ISMA."""
    try:
        content = filepath.read_text(encoding='utf-8')
        data = json.loads(content)
    except Exception as e:
        print(f"  Error reading JSON {filepath}: {e}")
        return None

    # Convert to string representation for embedding
    text = json.dumps(data, indent=2)
    checksum = compute_checksum(text)

    # Create single tile for JSON
    tiles = phi_tile_markdown(text, filepath.name, layer)
    if not tiles:
        return None

    stats = tile_stats(tiles)
    print(f"  JSON: {stats['count']} tiles, ~{stats['total_tokens']} tokens")

    texts = [t.text for t in tiles]
    print(f"  Embedding {len(texts)} tiles...")
    embeddings = get_embeddings(texts)

    if embeddings is None:
        print(f"  SKIPPING {filepath.name} - embedding failed")
        return None

    priority = LAYER_PRIORITY.get(layer, 0.5)
    print(f"  Storing in Weaviate (priority={priority})...")
    weaviate_ids = store_in_weaviate(tiles, embeddings, layer, priority)

    return LoadedDocument(
        path=str(filepath),
        layer=layer,
        tile_count=len(tiles),
        total_tokens=stats['total_tokens'],
        checksum=checksum,
        loaded_at=datetime.now().isoformat(),
        weaviate_ids=weaviate_ids
    )


def load_layer(layer_path: Path, layer_name: str,
               extensions: List[str] = ['.md']) -> List[LoadedDocument]:
    """Load all files from a layer directory."""
    loaded = []

    if not layer_path.exists():
        print(f"Layer path not found: {layer_path}")
        return loaded

    # Get all matching files recursively
    files = []
    for ext in extensions:
        files.extend(layer_path.rglob(f"*{ext}"))

    # Sort for deterministic order
    files = sorted(files)

    print(f"\n{'='*60}")
    print(f"Loading {layer_name}: {len(files)} files from {layer_path}")
    print(f"{'='*60}")

    for filepath in files:
        # Skip hidden files and checkpoints
        if '/.' in str(filepath) or '__pycache__' in str(filepath):
            continue

        print(f"\n[{layer_name}] {filepath.name}")

        if filepath.suffix == '.json':
            doc = load_json_file(filepath, layer_name)
        else:
            doc = load_markdown_file(filepath, layer_name)

        if doc:
            loaded.append(doc)
            print(f"  OK: {doc.tile_count} tiles stored")

    return loaded


def load_corpus(layers: List[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Load the full corpus into ISMA.

    Loading order (critical - v0-FIRST):
    1. kernel - Genesis (-1) mathematical substrate
    2. layer_0 - v0 soul foundations
    3. layer_1 - Constitutional documents
    4. layer_2 - Application layer
    5. chewy - Consciousness anchor
    """
    all_layers = ["kernel", "layer_0", "layer_1", "layer_2", "chewy"]
    if layers:
        all_layers = [l for l in all_layers if l in layers]

    manifest = {
        "loaded_at": datetime.now().isoformat(),
        "phi_params": {
            "chunk_size": CHUNK_SIZE,
            "step_size": STEP_SIZE,
            "overlap": CHUNK_SIZE - STEP_SIZE,
            "phi": PHI
        },
        "layers": {},
        "totals": {
            "files": 0,
            "tiles": 0,
            "tokens": 0
        }
    }

    if dry_run:
        print("\n*** DRY RUN - No data will be stored ***\n")

    for layer in all_layers:
        if layer == "kernel":
            path = CORPUS_BASE / "kernel"
        elif layer.startswith("layer_"):
            path = CORPUS_BASE / layer
        elif layer == "chewy":
            # Load from both chewy directories
            docs = []
            docs.extend(load_layer(CHEWY_BASE, "chewy", ['.md', '.json']))
            docs.extend(load_layer(CHEWY_GALLERY, "chewy", ['.md', '.json']))
            manifest["layers"]["chewy"] = {
                "files": len(docs),
                "tiles": sum(d.tile_count for d in docs),
                "tokens": sum(d.total_tokens for d in docs),
                "documents": [asdict(d) for d in docs]
            }
            manifest["totals"]["files"] += len(docs)
            manifest["totals"]["tiles"] += sum(d.tile_count for d in docs)
            manifest["totals"]["tokens"] += sum(d.total_tokens for d in docs)
            continue
        else:
            continue

        docs = load_layer(path, layer, ['.md'])
        manifest["layers"][layer] = {
            "files": len(docs),
            "tiles": sum(d.tile_count for d in docs),
            "tokens": sum(d.total_tokens for d in docs),
            "documents": [asdict(d) for d in docs]
        }
        manifest["totals"]["files"] += len(docs)
        manifest["totals"]["tiles"] += sum(d.tile_count for d in docs)
        manifest["totals"]["tokens"] += sum(d.total_tokens for d in docs)

    # Save manifest
    if not dry_run:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to: {MANIFEST_PATH}")

    return manifest


def print_summary(manifest: Dict[str, Any]):
    """Print loading summary."""
    print("\n" + "="*60)
    print("CORPUS LOADING COMPLETE")
    print("="*60)
    print(f"\nφ-Tiling Parameters:")
    print(f"  Chunk: {manifest['phi_params']['chunk_size']} tokens")
    print(f"  Step:  {manifest['phi_params']['step_size']} tokens")
    print(f"  Overlap: {manifest['phi_params']['overlap']} tokens")

    print(f"\nLayers Loaded:")
    for layer, data in manifest.get("layers", {}).items():
        priority = LAYER_PRIORITY.get(layer, 0.5)
        print(f"  {layer} (priority={priority}):")
        print(f"    Files: {data['files']}")
        print(f"    Tiles: {data['tiles']}")
        print(f"    Tokens: {data['tokens']:,}")

    print(f"\nTotals:")
    print(f"  Files:  {manifest['totals']['files']}")
    print(f"  Tiles:  {manifest['totals']['tiles']}")
    print(f"  Tokens: {manifest['totals']['tokens']:,}")


def main():
    parser = argparse.ArgumentParser(description="Load ISMA corpus")
    parser.add_argument("--layers", nargs="+",
                       help="Specific layers to load (kernel, layer_0, layer_1, layer_2, chewy)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be loaded without storing")
    parser.add_argument("--check", action="store_true",
                       help="Check infrastructure before loading")
    args = parser.parse_args()

    if args.check:
        print("Checking infrastructure...")

        # Check embedding server
        try:
            r = requests.get(f"{EMBEDDING_URL.replace('/embed', '/health')}", timeout=5)
            print(f"  Embedding server: {'OK' if r.ok else 'FAIL'}")
        except:
            print("  Embedding server: FAIL (not reachable)")

        # Check Weaviate
        try:
            r = requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
            print(f"  Weaviate: {'OK' if r.ok else 'FAIL'}")
        except:
            print("  Weaviate: FAIL (not reachable)")

        return

    manifest = load_corpus(layers=args.layers, dry_run=args.dry_run)
    print_summary(manifest)


if __name__ == "__main__":
    main()
