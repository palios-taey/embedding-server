#!/usr/bin/env python3
"""Create ISMA_Themes collection — dedicated routing layer for 24 canonical themes.

Migrates theme tiles from ISMA_Quantum (wrong: mixed with 1M peers) to a
dedicated 24-object ISMA_Themes collection enabling sub-millisecond nearVector
routing without forcing HNSW to scan 1M tiles to find 24 special objects.

Architecture after migration:
  Query → embed → nearVector(ISMA_Themes, 24 objects) → top theme →
  required_motifs → where-filter on ISMA_Quantum main search

  NOT: inject theme text into reranker instruction (was causing 3.8x latency + regression)

Usage:
    python3 create_isma_themes_collection.py [--dry-run] [--force-recreate]
    python3 create_isma_themes_collection.py --verify  # show current state
"""

import argparse
import hashlib
import json
import logging
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

WEAVIATE_URL = "http://192.168.100.10:8088"
EMBEDDING_URL = "http://192.168.100.10:8091"
THEME_INDEX_PATH = "/var/spark/isma/theme_search_index.json"
THEMES_CLASS = "ISMA_Themes"
QUANTUM_CLASS = "ISMA_Quantum"


# =============================================================================
# SCHEMA
# =============================================================================

SCHEMA = {
    "class": THEMES_CLASS,
    "description": (
        "24 canonical ISMA themes — routing layer for query-to-theme mapping. "
        "Query → nearVector(ISMA_Themes) → required_motifs → filter on ISMA_Quantum. "
        "Sub-millisecond lookup (24 objects). NOT mixed into main 1M-tile index."
    ),
    "vectorizer": "none",
    "invertedIndexConfig": {
        "bm25": {"b": 0.75, "k1": 1.2},
        "cleanupIntervalSeconds": 300,
        "stopwords": {"preset": "en"},
    },
    "properties": [
        {"name": "theme_id", "dataType": ["text"],
         "tokenization": "field", "indexFilterable": True, "indexSearchable": False,
         "description": "e.g. '001' through '024'"},
        {"name": "display_name", "dataType": ["text"],
         "tokenization": "word", "indexSearchable": True, "indexFilterable": False},
        {"name": "description", "dataType": ["text"],
         "tokenization": "word", "indexSearchable": True, "indexFilterable": False},
        {"name": "required_motifs", "dataType": ["text[]"],
         "indexFilterable": True, "indexSearchable": True,
         "description": "Motifs that define this theme — used as filter predicate on ISMA_Quantum"},
        {"name": "supporting_motifs", "dataType": ["text[]"],
         "indexFilterable": True, "indexSearchable": True},
        {"name": "content_hash", "dataType": ["text"],
         "tokenization": "field", "indexFilterable": True, "indexSearchable": False,
         "description": "SHA256 of theme content (not just ID) for provenance tracking"},
        {"name": "content", "dataType": ["text"],
         "tokenization": "word", "indexSearchable": True, "indexFilterable": False,
         "description": "Full theme tile text used for embedding"},
        {"name": "source_quantum_uuid", "dataType": ["text"],
         "tokenization": "field", "indexFilterable": True, "indexSearchable": False,
         "description": "UUID of the source tile in ISMA_Quantum (for cross-reference)"},
    ],
}


# =============================================================================
# WEAVIATE HELPERS
# =============================================================================

def class_exists(class_name: str) -> bool:
    r = requests.get(f"{WEAVIATE_URL}/v1/schema/{class_name}", timeout=10)
    return r.status_code == 200


def delete_class(class_name: str):
    r = requests.delete(f"{WEAVIATE_URL}/v1/schema/{class_name}", timeout=30)
    if r.status_code in (200, 204):
        log.info("Deleted class %s", class_name)
    else:
        log.error("Failed to delete %s: %s %s", class_name, r.status_code, r.text[:200])
        sys.exit(1)


def create_class(schema: dict):
    r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, timeout=30)
    if r.status_code in (200, 201):
        log.info("Created class %s", schema["class"])
    else:
        log.error("Failed to create class: %s %s", r.status_code, r.text[:400])
        sys.exit(1)


def get_embedding(text: str) -> list:
    """Get Qwen3 embedding from the LB endpoint."""
    payload = {"model": "Qwen/Qwen3-Embedding-8B", "input": text}
    r = requests.post(f"{EMBEDDING_URL}/v1/embeddings", json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Embedding failed: {r.status_code} {r.text[:200]}")
    return r.json()["data"][0]["embedding"]


def fetch_theme_tiles_from_quantum() -> dict:
    """Fetch existing theme tiles from ISMA_Quantum, keyed by content_hash."""
    gql = f"""{{
        Get {{
            {QUANTUM_CLASS}(
                where: {{
                    path: ["scale"]
                    operator: ContainsAny
                    valueText: ["theme"]
                }}
                limit: 50
            ) {{
                content_hash
                content
                rosetta_summary
                dominant_motifs
                source_file
                scale
                _additional {{ id }}
            }}
        }}
    }}"""
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=30)
    tiles = r.json().get("data", {}).get("Get", {}).get(QUANTUM_CLASS, []) or []
    log.info("Found %d theme tiles in ISMA_Quantum", len(tiles))
    return {t["content_hash"]: t for t in tiles}


def store_theme(theme_id: str, display_name: str, description: str,
                required_motifs: list, supporting_motifs: list,
                content: str, source_uuid: str, vector: list) -> bool:
    """Store a single theme object in ISMA_Themes."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    obj = {
        "class": THEMES_CLASS,
        "properties": {
            "theme_id": theme_id,
            "display_name": display_name,
            "description": description,
            "required_motifs": required_motifs,
            "supporting_motifs": supporting_motifs,
            "content_hash": content_hash,
            "content": content[:4000],
            "source_quantum_uuid": source_uuid,
        },
        "vector": vector,
    }
    r = requests.post(f"{WEAVIATE_URL}/v1/objects", json=obj, timeout=15)
    return r.status_code in (200, 201)


# =============================================================================
# VERIFY
# =============================================================================

def verify():
    gql = f"""{{
        Aggregate {{
            {THEMES_CLASS} {{ meta {{ count }} }}
        }}
    }}"""
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=15)
    count = (r.json().get("data", {}).get("Aggregate", {})
             .get(THEMES_CLASS, [{}])[0].get("meta", {}).get("count", "?"))

    # Also fetch all themes to show state
    gql2 = f"""{{
        Get {{
            {THEMES_CLASS}(limit: 30) {{
                theme_id display_name required_motifs content_hash
            }}
        }}
    }}"""
    r2 = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql2}, timeout=15)
    themes = r2.json().get("data", {}).get("Get", {}).get(THEMES_CLASS, []) or []

    print(f"\n{THEMES_CLASS}: {count} themes stored")
    for t in sorted(themes, key=lambda x: x.get("theme_id", "")):
        motifs = t.get("required_motifs", [])
        print(f"  [{t['theme_id']}] {t['display_name'][:40]:<40} "
              f"hash={t['content_hash']} motifs={motifs}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print without writing to Weaviate")
    parser.add_argument("--force-recreate", action="store_true",
                        help="Delete ISMA_Themes if it exists before creating")
    parser.add_argument("--verify", action="store_true",
                        help="Show current ISMA_Themes state and exit")
    args = parser.parse_args()

    if args.verify:
        if not class_exists(THEMES_CLASS):
            print(f"{THEMES_CLASS} does not exist yet")
            return
        verify()
        return

    # Load theme index
    with open(THEME_INDEX_PATH) as f:
        index = json.load(f)
    themes_list = list(index.get("themes", index).values()) \
        if isinstance(index.get("themes", index), dict) else index.get("themes", [])
    log.info("Loaded %d themes from index", len(themes_list))

    # Fetch existing tiles from ISMA_Quantum for content
    quantum_tiles = fetch_theme_tiles_from_quantum()
    log.info("Found %d matching tiles in ISMA_Quantum", len(quantum_tiles))

    # Handle class creation
    if class_exists(THEMES_CLASS):
        if args.force_recreate:
            delete_class(THEMES_CLASS)
        else:
            log.info("%s already exists. Use --force-recreate to rebuild.", THEMES_CLASS)
            verify()
            return

    if not args.dry_run:
        create_class(SCHEMA)

    # Map old content_hash (ID-based) to theme
    import hashlib as _hs
    old_hash_to_theme = {
        _hs.sha256(f"theme_tile_{t['theme_id']}".encode()).hexdigest()[:16]: t
        for t in themes_list
    }

    success = failed = 0
    for theme in themes_list:
        theme_id = theme["theme_id"]
        display_name = theme["display_name"]
        description = theme.get("description", "")
        required_motifs = theme.get("required_motifs", [])
        supporting_motifs = theme.get("supporting_motifs", [])

        # Find existing quantum tile for this theme
        old_hash = _hs.sha256(f"theme_tile_{theme_id}".encode()).hexdigest()[:16]
        quantum_tile = quantum_tiles.get(old_hash)

        if quantum_tile:
            content = quantum_tile.get("content") or quantum_tile.get("rosetta_summary") or ""
            source_uuid = quantum_tile["_additional"]["id"]
        else:
            # Theme tile not yet built — construct minimal content from index
            log.warning("No ISMA_Quantum tile for theme %s — building from index", theme_id)
            content = f"# Theme {theme_id}: {display_name}\n\n{description}\n\n"
            content += f"Key motifs: {', '.join(required_motifs)}\n"
            content += f"Supporting motifs: {', '.join(supporting_motifs)}"
            source_uuid = ""

        if not content.strip():
            log.warning("Empty content for theme %s, skipping", theme_id)
            failed += 1
            continue

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        log.info("[%s] %s — content_hash=%s (content-based, was ID-based)", theme_id, display_name, content_hash)

        if args.dry_run:
            print(f"  DRY RUN [{theme_id}] {display_name[:40]} | required_motifs={required_motifs}")
            success += 1
            continue

        # Get embedding from Qwen3
        try:
            vector = get_embedding(content[:2000])
        except Exception as e:
            log.error("Embedding failed for theme %s: %s", theme_id, e)
            failed += 1
            continue

        ok = store_theme(theme_id, display_name, description,
                         required_motifs, supporting_motifs,
                         content, source_uuid, vector)
        if ok:
            log.info("  ✓ Stored theme %s", theme_id)
            success += 1
        else:
            log.error("  ✗ Failed to store theme %s", theme_id)
            failed += 1

        time.sleep(0.1)  # brief pause between Weaviate writes

    log.info("Done — %d stored, %d failed", success, failed)
    if not args.dry_run and success > 0:
        verify()


if __name__ == "__main__":
    main()
