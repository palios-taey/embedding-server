#!/usr/bin/env python3
"""MuVera ColBERT encoding pipeline for ISMA.

Encodes tiles with answerai-colbert-small-v1 (33M params) and inserts
multi-vector embeddings into Weaviate with MuVera FDE compression.

Usage:
    # Create collection (run once)
    python3 muvera_encode.py create-collection

    # Encode a shard (run on each node)
    python3 muvera_encode.py encode --shard 0 --total-shards 3

    # Encode all (single node)
    python3 muvera_encode.py encode

    # PoC: encode 10K tiles
    python3 muvera_encode.py encode --limit 10000

    # Check progress
    python3 muvera_encode.py stats
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

# --- Config ---
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
SOURCE_CLASS = "ISMA_Quantum"
COLBERT_CLASS = "ISMA_ColBERT"
MODEL_NAME = "lightonai/answerai-colbert-small-v1"
BATCH_SIZE = 64  # tiles per encoding batch
INSERT_BATCH = 50  # objects per Weaviate batch insert
TOKEN_DIM = 96  # answerai-colbert-small-v1 token dimension

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _gql(query: str) -> dict:
    """Execute GraphQL query against Weaviate."""
    r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data


def create_collection():
    """Create ISMA_ColBERT collection with MuVera multi-vector support."""
    # Check if exists
    r = requests.get(f"{WEAVIATE_URL}/v1/schema/{COLBERT_CLASS}")
    if r.status_code == 200:
        log.info("Collection %s already exists", COLBERT_CLASS)
        count_r = _gql(f'{{ Aggregate {{ {COLBERT_CLASS} {{ meta {{ count }} }} }} }}')
        count = count_r["data"]["Aggregate"][COLBERT_CLASS][0]["meta"]["count"]
        log.info("  Current object count: %d", count)
        return

    schema = {
        "class": COLBERT_CLASS,
        "description": "MuVera ColBERT multi-vector encodings for ISMA tiles",
        "vectorConfig": {
            "colbert": {
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "efConstruction": 128,
                    "maxConnections": 32,
                    "ef": -1,
                    "multivector": {
                        "enabled": True,
                        "aggregation": "maxSim",
                        "muvera": {
                            "enabled": True,
                            "ksim": 4,
                            "dprojections": 16,
                            "repetitions": 20,
                        },
                    },
                },
                "vectorizer": {
                    "none": {},
                },
            },
        },
        "properties": [
            {
                "name": "content_hash",
                "dataType": ["text"],
                "tokenization": "field",
                "indexSearchable": True,
                "indexFilterable": True,
            },
            {
                "name": "platform",
                "dataType": ["text"],
                "tokenization": "word",
                "indexSearchable": True,
                "indexFilterable": True,
            },
            {
                "name": "content_preview",
                "dataType": ["text"],
                "tokenization": "word",
                "indexSearchable": True,
            },
            {
                "name": "rosetta_summary",
                "dataType": ["text"],
                "tokenization": "word",
                "indexSearchable": True,
            },
            {
                "name": "dominant_motifs",
                "dataType": ["text[]"],
                "tokenization": "field",
            },
            {
                "name": "scale",
                "dataType": ["text"],
                "tokenization": "word",
                "indexFilterable": True,
            },
            {
                "name": "encoded_at",
                "dataType": ["text"],
                "tokenization": "field",
            },
        ],
    }

    r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema)
    if r.status_code not in (200, 201):
        log.error("Failed to create collection: %s %s", r.status_code, r.text[:500])
        sys.exit(1)
    log.info("Created collection %s with MuVera enabled", COLBERT_CLASS)
    log.info("  MuVera config: ksim=4, dprojections=16, repetitions=20")
    log.info("  FDE dimensionality: 20 * 2^4 * 16 = 5,120")


def load_model():
    """Load ColBERT model via PyLate."""
    from pylate import models

    log.info("Loading %s...", MODEL_NAME)
    t0 = time.monotonic()
    model = models.ColBERT(MODEL_NAME)
    elapsed = time.monotonic() - t0
    log.info("Model loaded in %.1fs", elapsed)
    return model


def get_content_hashes(shard: int = 0, total_shards: int = 1,
                       limit: Optional[int] = None) -> List[str]:
    """Get all unique content_hashes from source collection, sharded."""
    log.info("Fetching content_hashes from %s (shard %d/%d)...",
             SOURCE_CLASS, shard + 1, total_shards)

    # Get already-encoded hashes to skip (cursor-based pagination)
    encoded = set()
    try:
        cursor = None
        while True:
            after_clause = f', after: "{cursor}"' if cursor else ""
            r = _gql(f'''{{ Get {{ {COLBERT_CLASS}(
                limit: 10000{after_clause}
            ) {{ content_hash _additional {{ id }} }} }} }}''')
            batch = r["data"]["Get"][COLBERT_CLASS]
            if not batch:
                break
            for obj in batch:
                encoded.add(obj["content_hash"])
            cursor = batch[-1]["_additional"]["id"]
            if len(batch) < 10000:
                break
    except Exception as e:
        log.warning("Could not fetch encoded hashes: %s", e)

    log.info("Already encoded: %d hashes", len(encoded))

    # Get all unique content_hashes from source (cursor-based pagination)
    # Note: Weaviate cursor API does not support `where` + `after` together,
    # so we fetch ALL objects and deduplicate client-side.
    seen = set()
    all_hashes = []
    cursor = None
    batch_size = 10000
    fetched = 0

    while True:
        after_clause = f', after: "{cursor}"' if cursor else ""
        r = _gql(f'''{{ Get {{ {SOURCE_CLASS}(
            limit: {batch_size}{after_clause}
        ) {{ content_hash _additional {{ id }} }} }} }}''')
        batch = r["data"]["Get"][SOURCE_CLASS]
        if not batch:
            break
        for obj in batch:
            ch = obj.get("content_hash")
            if ch and ch not in encoded and ch not in seen:
                seen.add(ch)
                all_hashes.append(ch)
        fetched += len(batch)
        cursor = batch[-1]["_additional"]["id"]
        if fetched % 100000 == 0:
            log.info("  Scanned %d objects, found %d new hashes...", fetched, len(all_hashes))
        if len(batch) < batch_size:
            break
    log.info("  Scanned %d total objects", fetched)

    # Deduplicate (search_512 may have duplicates per content_hash)
    unique = list(dict.fromkeys(all_hashes))
    log.info("Total unique content_hashes to encode: %d", len(unique))

    # Shard
    shard_hashes = [h for i, h in enumerate(unique) if i % total_shards == shard]
    log.info("This shard: %d hashes", len(shard_hashes))

    if limit:
        shard_hashes = shard_hashes[:limit]
        log.info("Limited to %d hashes", len(shard_hashes))

    return shard_hashes


def fetch_tile_content(content_hashes: List[str]) -> Dict[str, dict]:
    """Batch-fetch tile content from source collection.

    Fetches each hash individually to avoid Or-operand limits.
    Prefers search_512 scale, falls back to any scale with content.
    """
    result = {}
    for i, ch in enumerate(content_hashes):
        if ch in result:
            continue
        try:
            r = _gql(f'''{{ Get {{ {SOURCE_CLASS}(
                where: {{ path: ["content_hash"], operator: Equal, valueText: "{ch}" }}
                limit: 5
            ) {{ content_hash content platform scale content_preview
                 rosetta_summary dominant_motifs }} }} }}''')
            tiles = r["data"]["Get"][SOURCE_CLASS]
            if not tiles:
                continue
            # Prefer tile with most content
            best = max(tiles, key=lambda t: len(t.get("content") or ""))
            result[ch] = best
        except Exception as e:
            if i < 3:  # Only log first few
                log.warning("Failed to fetch hash %s: %s", ch[:12], e)

    return result


def encode_and_insert(model, content_hashes: List[str]):
    """Encode tiles with ColBERT and insert into Weaviate with multi-vectors."""
    from datetime import datetime, timezone

    total = len(content_hashes)
    encoded_count = 0
    insert_count = 0
    t_start = time.monotonic()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_hashes = content_hashes[batch_start:batch_start + BATCH_SIZE]
        t0 = time.monotonic()

        # Fetch content
        tiles = fetch_tile_content(batch_hashes)
        if not tiles:
            log.warning("No content fetched for batch at %d", batch_start)
            continue

        # Prepare texts for encoding (content + rosetta for richer representation)
        texts = []
        hash_order = []
        for ch in batch_hashes:
            if ch in tiles:
                tile = tiles[ch]
                text = (tile.get("content") or "") + " " + (tile.get("rosetta_summary") or "")
                text = text.strip()[:4096]  # Cap at 4K chars
                if text:
                    texts.append(text)
                    hash_order.append(ch)

        if not texts:
            continue

        # Encode with ColBERT (produces list of list of floats — multi-vector)
        try:
            embeddings = model.encode(texts, is_query=False)
            encoded_count += len(embeddings)
        except Exception as e:
            log.error("Encoding failed at batch %d: %s", batch_start, e)
            continue

        # Insert into Weaviate
        now = datetime.now(timezone.utc).isoformat()
        objects = []
        for idx, ch in enumerate(hash_order):
            if idx >= len(embeddings):
                break
            tile = tiles.get(ch, {})
            # ColBERT returns numpy array — convert to nested list
            vectors = embeddings[idx]
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()

            obj = {
                "class": COLBERT_CLASS,
                "properties": {
                    "content_hash": ch,
                    "platform": tile.get("platform") or "",
                    "content_preview": (tile.get("content_preview") or "")[:200],
                    "rosetta_summary": (tile.get("rosetta_summary") or "")[:500],
                    "dominant_motifs": tile.get("dominant_motifs") or [],
                    "scale": tile.get("scale") or "search_512",
                    "encoded_at": now,
                },
                "vectors": {"colbert": vectors},
            }
            objects.append(obj)

        # Batch insert
        for i in range(0, len(objects), INSERT_BATCH):
            ins_batch = objects[i:i + INSERT_BATCH]
            try:
                r = requests.post(
                    f"{WEAVIATE_URL}/v1/batch/objects",
                    json={"objects": ins_batch},
                    timeout=60,
                )
                if r.status_code == 200:
                    results = r.json()
                    for res in results:
                        if "result" in res and res["result"].get("errors"):
                            log.warning("Insert error: %s",
                                        res["result"]["errors"]["error"][:200])
                        else:
                            insert_count += 1
                else:
                    log.error("Batch insert failed: %s %s", r.status_code, r.text[:300])
            except Exception as e:
                log.error("Batch insert exception: %s", e)

        elapsed = time.monotonic() - t0
        total_elapsed = time.monotonic() - t_start
        rate = encoded_count / max(total_elapsed, 0.1)
        remaining = (total - batch_start - len(batch_hashes)) / max(rate, 0.1)
        log.info(
            "[%d/%d] encoded=%d inserted=%d  %.1f tiles/sec  ETA %.0fs",
            batch_start + len(batch_hashes), total,
            encoded_count, insert_count,
            rate, remaining,
        )

    total_elapsed = time.monotonic() - t_start
    log.info(
        "DONE: encoded=%d inserted=%d in %.1fs (%.1f tiles/sec)",
        encoded_count, insert_count, total_elapsed,
        encoded_count / max(total_elapsed, 0.1),
    )


def stats():
    """Show encoding progress stats."""
    try:
        r = _gql(f'{{ Aggregate {{ {COLBERT_CLASS} {{ meta {{ count }} }} }} }}')
        colbert_count = r["data"]["Aggregate"][COLBERT_CLASS][0]["meta"]["count"]
    except Exception:
        colbert_count = 0

    try:
        r = _gql(f'{{ Aggregate {{ {SOURCE_CLASS} {{ meta {{ count }} }} }} }}')
        source_total = r["data"]["Aggregate"][SOURCE_CLASS][0]["meta"]["count"]
    except Exception:
        source_total = 0

    try:
        r = _gql(f'''{{ Aggregate {{ {SOURCE_CLASS}(
            where: {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
        ) {{ meta {{ count }} }} }} }}''')
        source_512 = r["data"]["Aggregate"][SOURCE_CLASS][0]["meta"]["count"]
    except Exception:
        source_512 = 0

    # ColBERT encodes per content_hash (one per doc), not per tile
    # Unique content_hashes ≈ colbert_count (each hash = 1 ColBERT object)
    print(f"Source objects total:      {source_total:,}")
    print(f"Source search_512 tiles:   {source_512:,}")
    print(f"ColBERT objects:           {colbert_count:,}")
    print(f"  (1 ColBERT obj per unique content_hash — encoding is per-document, not per-tile)")


def main():
    parser = argparse.ArgumentParser(description="MuVera ColBERT encoding pipeline")
    parser.add_argument("command", choices=["create-collection", "encode", "stats"])
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--total-shards", type=int, default=1, help="Total shards")
    parser.add_argument("--limit", type=int, default=None, help="Limit tiles to encode")
    args = parser.parse_args()

    if args.command == "create-collection":
        create_collection()
    elif args.command == "stats":
        stats()
    elif args.command == "encode":
        create_collection()  # Ensure collection exists
        model = load_model()
        hashes = get_content_hashes(args.shard, args.total_shards, args.limit)
        if hashes:
            encode_and_insert(model, hashes)
        else:
            log.info("No hashes to encode")
        stats()


if __name__ == "__main__":
    main()
