#!/usr/bin/env python3
"""
patch_v2_content.py — Fix ISMA_Quantum_v2 content field truncation.

Problem: migrate_v1_to_v2.py stored only content[:2048] from the FIRST search_512 tile.
Fix: For each V2 object, fetch ALL its search_512 tiles from ISMA_Quantum,
     concatenate their full content (no truncation), re-embed with Qwen3,
     then PATCH the V2 object with new content + new raw named vector.

Usage:
    python3 patch_v2_content.py [--batch-size 50] [--dry-run] [--limit N]

Progress is checkpointed in Redis key: isma:v2_patch:done (set of patched content_hashes).
"""

import os
import argparse
import json
import logging
import time
import sys
from typing import Optional

import requests
import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/patch_v2_content.log")],
)
log = logging.getLogger(__name__)

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379
V1_CLASS = "ISMA_Quantum"
V2_CLASS = "ISMA_Quantum_v2"
CHECKPOINT_KEY = "isma:v2_patch:done"

session = requests.Session()
session.headers["Content-Type"] = "application/json"


def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def embed_text(text: str) -> Optional[list]:
    """Embed text via Qwen3 embedding LB (OpenAI-compatible). Returns vector or None.

    Qwen3-Embedding-8B has 8192 token context. For very long text, the vLLM server
    may return 400 if the input exceeds its configured max_model_len.
    We truncate at 32000 chars (~8000 tokens) as a safe upper bound.
    """
    # Truncate only if extremely long (32000 chars ≈ 8000 tokens, at model limit)
    if len(text) > 32000:
        text = text[:32000]
    try:
        r = session.post(EMBEDDING_URL, json={"input": text, "model": "Qwen/Qwen3-Embedding-8B"}, timeout=60)
        if not r.ok:
            log.warning("Embed HTTP %d: %s", r.status_code, r.text[:300])
            return None
        data = r.json()
        # OpenAI-compatible response: {"data": [{"embedding": [...]}]}
        embedding_data = data.get("data", [])
        if embedding_data:
            return embedding_data[0].get("embedding")
    except Exception as e:
        log.warning("Embed failed: %s", e)
    return None


def fetch_v2_page(limit: int, offset: int) -> list:
    """Fetch a page of V2 objects with content_hash and tile_ids_512."""
    query = """
    {
      Get {
        ISMA_Quantum_v2(limit: %d offset: %d) {
          _additional { id }
          content_hash
          tile_ids_512
        }
      }
    }
    """ % (limit, offset)
    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("data", {}).get("Get", {}).get("ISMA_Quantum_v2", []) or []
    except Exception as e:
        log.error("fetch_v2_page failed at offset=%d: %s", offset, e)
        return []


def fetch_v1_tile(uuid: str) -> Optional[str]:
    """Fetch a single V1 tile by UUID, return its content."""
    try:
        r = session.get(
            f"{WEAVIATE_URL}/v1/objects/ISMA_Quantum/{uuid}",
            params={"include": ""},
            timeout=15,
        )
        if r.status_code == 200:
            props = r.json().get("properties", {})
            return props.get("content", "") or ""
    except Exception as e:
        log.debug("fetch_v1_tile %s failed: %s", uuid[:8], e)
    return None


def batch_fetch_v1_tiles(tile_ids: list) -> list:
    """Fetch multiple V1 tiles by UUID. Returns list of (tile_index, content) sorted by index."""
    if not tile_ids:
        return []
    # Use GraphQL with id in filter — fetch tile_index + content
    # Weaviate supports filtering by internal id via _additional.id
    # For large sets, chunk and use object REST API per tile
    chunk_size = 50
    results = []
    for i in range(0, len(tile_ids), chunk_size):
        chunk = tile_ids[i:i + chunk_size]
        ids_list = ", ".join([f'"{uid}"' for uid in chunk])
        query = """
        {
          Get {
            ISMA_Quantum(
              where: {
                operator: Or
                operands: [%s]
              }
              limit: %d
            ) {
              _additional { id }
              content
              tile_index
            }
          }
        }
        """ % (
            ", ".join([
                '{ operator: Equal path: ["id"] valueText: "%s" }' % uid
                for uid in chunk
            ]),
            len(chunk),
        )
        try:
            r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=60)
            r.raise_for_status()
            tiles = r.json().get("data", {}).get("Get", {}).get("ISMA_Quantum", []) or []
            for t in tiles:
                idx = t.get("tile_index") or 0
                content = t.get("content", "") or ""
                if content:
                    results.append((idx, content))
        except Exception as e:
            log.warning("GraphQL chunk failed, falling back to REST: %s", e)
            # Fallback: fetch individually
            for uid in chunk:
                content = fetch_v1_tile(uid)
                if content:
                    results.append((0, content))

    # Sort by tile_index to preserve document order
    results.sort(key=lambda x: x[0])
    return [c for _, c in results]


def patch_v2_object_content_only(uuid: str, content: str) -> bool:
    """PATCH a V2 object with new full content (no re-embed — Option E uses V1 vectors)."""
    try:
        r = session.patch(
            f"{WEAVIATE_URL}/v1/objects/{V2_CLASS}/{uuid}",
            json={"properties": {"content": content}},
            timeout=30,
        )
        if r.status_code not in (200, 204):
            log.warning("PATCH failed for %s: HTTP %d %s", uuid[:8], r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        log.warning("PATCH error for %s: %s", uuid[:8], e)
        return False


def run(batch_size: int, dry_run: bool, limit: int):
    rdb = get_redis()
    done_hashes = rdb.smembers(CHECKPOINT_KEY)
    log.info("Already patched: %d objects", len(done_hashes))

    offset = 0
    page_size = 250
    total_patched = 0
    total_skipped = 0
    total_failed = 0
    t0 = time.monotonic()

    while True:
        if limit and total_patched >= limit:
            log.info("Reached --limit %d, stopping.", limit)
            break

        page = fetch_v2_page(page_size, offset)
        if not page:
            log.info("No more pages at offset=%d. Done.", offset)
            break

        offset += len(page)
        batch_to_patch = []
        for obj in page:
            ch = obj.get("content_hash", "")
            if ch in done_hashes:
                total_skipped += 1
                continue
            uuid = obj.get("_additional", {}).get("id", "")
            tile_ids = obj.get("tile_ids_512") or []
            batch_to_patch.append((uuid, ch, tile_ids))

        log.info("Page: %d objects, %d to patch, %d skipped",
                 len(page), len(batch_to_patch), len(page) - len(batch_to_patch))

        for uuid, ch, tile_ids in batch_to_patch:
            if limit and total_patched >= limit:
                break

            # Fetch all V1 tiles for this document
            tile_contents = batch_fetch_v1_tiles(tile_ids)
            if not tile_contents:
                log.warning("No tile contents for %s (uuid=%s), skipping", ch[:12], uuid[:8])
                total_failed += 1
                continue

            # Concatenate ALL tile content — no truncation
            full_content = "\n\n".join(tile_contents)

            if dry_run:
                log.info("DRY RUN: %s — %d tiles, %d chars total",
                         ch[:12], len(tile_contents), len(full_content))
                total_patched += 1
                continue

            # Patch V2 object — content only (no re-embed: Option E uses V1 vectors)
            ok = patch_v2_object_content_only(uuid, full_content)
            if ok:
                rdb.sadd(CHECKPOINT_KEY, ch)
                done_hashes.add(ch)
                total_patched += 1
            else:
                total_failed += 1

            # Progress
            if total_patched % 100 == 0 and total_patched > 0:
                elapsed = time.monotonic() - t0
                rate = total_patched / elapsed
                log.info("Patched %d | skipped %d | failed %d | %.1f/s",
                         total_patched, total_skipped, total_failed, rate)

    elapsed = time.monotonic() - t0
    log.info("DONE: patched=%d skipped=%d failed=%d in %.0fs",
             total_patched, total_skipped, total_failed, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch V2 content truncation")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N patches (0=all)")
    args = parser.parse_args()

    log.info("patch_v2_content.py starting (dry_run=%s, limit=%d)", args.dry_run, args.limit)
    run(batch_size=args.batch_size, dry_run=args.dry_run, limit=args.limit)
