#!/usr/bin/env python3
"""
F8 - Tier 1 Batch Classification of ALL Weaviate tiles.

Uses cosine similarity against anchor vectors (from F5) for pure vector math
classification. No LLM needed - just matrix multiplication.

Algorithm:
  1. Load 48 anchor vectors from Redis -> numpy matrix (48 x 4096)
  2. Cursor-iterate ALL search_512 tiles from Weaviate in pages of 500
  3. Batch cosine similarity: tiles @ anchors.T
  4. Motifs where similarity >= 0.4 become assignments, >= 0.6 become dominant
  5. PATCH each tile with enrichment properties
  6. Propagate to parent tiles (context_2048, full_4096)

Usage:
    tier1_batch_classify.py --check          # Verify anchors, show stats
    tier1_batch_classify.py --canary 1000    # Test on 1000 tiles
    tier1_batch_classify.py --run            # Full classification
    tier1_batch_classify.py --run --resume   # Resume from checkpoint
    tier1_batch_classify.py --propagate      # Propagate to parent tiles only
    tier1_batch_classify.py --stats          # Coverage report
"""

import os
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hmm.redis_store import HMMRedisStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tier1")

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
WEAVIATE_CLASS = "ISMA_Quantum"
CHECKPOINT_PATH = "/var/spark/isma/tier1_progress.json"
ENRICHMENT_VERSION = "tier1_1.0.0"

# Thresholds
SIMILARITY_THRESHOLD = 0.4   # include in motif_data_json
DOMINANT_THRESHOLD = 0.6     # include in dominant_motifs array
PAGE_SIZE = 500


# =============================================================================
# ANCHOR LOADING
# =============================================================================

def load_anchors(redis_store: HMMRedisStore) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load all anchor vectors from Redis, return normalized numpy matrix.

    Returns:
        (anchor_matrix, anchor_ids, anchor_types)
        anchor_matrix: (N, dim) normalized numpy array
        anchor_ids: list of motif/theme IDs
        anchor_types: list of 'motif' or 'theme' per anchor
    """
    motif_anchors = redis_store.anchor_get_all("motif")
    theme_anchors = redis_store.anchor_get_all("theme")

    if not motif_anchors:
        log.error("No motif anchors in Redis. Run generate_anchor_vectors.py --generate first.")
        return None, [], []

    ids = []
    types = []
    vectors = []

    for mid in sorted(motif_anchors.keys()):
        ids.append(mid)
        types.append("motif")
        vectors.append(motif_anchors[mid])

    for tid in sorted(theme_anchors.keys()):
        ids.append(tid)
        types.append("theme")
        vectors.append(theme_anchors[tid])

    matrix = np.array(vectors, dtype=np.float32)
    # Normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    log.info(f"Loaded {len(ids)} anchors ({matrix.shape[1]}-dim): "
             f"{len(motif_anchors)} motifs, {len(theme_anchors)} themes")
    return matrix, ids, types


# =============================================================================
# WEAVIATE CURSOR ITERATION
# =============================================================================

def fetch_tile_page(session: requests.Session, cursor: str = None,
                    scale: str = "search_512",
                    limit: int = PAGE_SIZE) -> Tuple[List[dict], str]:
    """Fetch a page of tiles with vectors from Weaviate using cursor.

    Weaviate cursor API only supports 'after' + 'limit' (no where/sort).
    We fetch all scales and filter client-side.

    Returns (objects, next_cursor). Empty list when done.
    """
    after_clause = f', after: "{cursor}"' if cursor else ""

    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                limit: {limit}
                {after_clause}
            ) {{
                content_hash
                scale
                dominant_motifs
                hmm_enrichment_version
                _additional {{ id vector }}
            }}
        }}
    }}"""

    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                         json={"query": gql}, timeout=60)
        if r.status_code != 200:
            log.error(f"Weaviate query failed: {r.status_code} {r.text[:200]}")
            return [], ""

        data = r.json()
        errors = data.get("errors")
        if errors:
            log.error(f"GraphQL errors: {errors}")
            return [], ""

        all_objects = data.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
        if not all_objects:
            return [], ""

        last_id = all_objects[-1]["_additional"]["id"]

        # Filter client-side by scale
        if scale:
            objects = [o for o in all_objects if o.get("scale") == scale]
        else:
            objects = all_objects

        return objects, last_id

    except Exception as e:
        log.error(f"Weaviate fetch error: {e}")
        return [], ""


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_batch(
    tile_vectors: np.ndarray,
    anchor_matrix: np.ndarray,
    anchor_ids: List[str],
    anchor_types: List[str],
) -> List[dict]:
    """Classify a batch of tile vectors against anchor vectors.

    Args:
        tile_vectors: (batch, dim) normalized numpy array
        anchor_matrix: (N_anchors, dim) normalized numpy array
        anchor_ids: list of anchor IDs
        anchor_types: list of 'motif' or 'theme'

    Returns:
        List of classification dicts, one per tile.
    """
    # Batch cosine similarity: (batch, dim) @ (dim, N_anchors) = (batch, N_anchors)
    similarities = tile_vectors @ anchor_matrix.T

    results = []
    for i in range(len(tile_vectors)):
        sims = similarities[i]
        dominant = []
        motif_data = []

        for j, (aid, atype, sim) in enumerate(zip(anchor_ids, anchor_types, sims)):
            # Only classify against motif anchors (not theme anchors)
            if atype != "motif":
                continue

            sim_val = float(sim)
            if sim_val >= SIMILARITY_THRESHOLD:
                motif_data.append({
                    "motif_id": aid,
                    "similarity": round(sim_val, 4),
                    "tier": "tier1_vector",
                })
                if sim_val >= DOMINANT_THRESHOLD:
                    dominant.append(aid)

        # Sort by similarity descending
        motif_data.sort(key=lambda m: m["similarity"], reverse=True)

        results.append({
            "dominant_motifs": dominant,
            "motif_data_json": json.dumps(motif_data),
        })

    return results


def build_enrichment_patch(classification: dict) -> dict:
    """Build Weaviate PATCH payload from classification result."""
    return {
        "dominant_motifs": classification["dominant_motifs"],
        "motif_data_json": classification["motif_data_json"],
        "hmm_enriched": True,
        "hmm_enrichment_version": ENRICHMENT_VERSION,
        "hmm_enriched_at": datetime.now(timezone.utc).isoformat(),
        "hmm_consensus": False,
        "hmm_gate_flags": ["TIER1_VECTOR", "AUTOMATED"],
    }


def should_classify(obj: dict) -> bool:
    """Determine if a tile should be classified.

    Skip tiles with tier2 enrichment (preserve Family results).
    Re-classify tiles with tier1 version (allow re-runs).
    Always classify unenriched tiles.
    """
    version = obj.get("hmm_enrichment_version") or ""
    if version.startswith("tier2"):
        return False
    return True


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(cursor: str, tiles_processed: int, tiles_enriched: int,
                    pages_done: int):
    """Save progress checkpoint."""
    data = {
        "cursor": cursor,
        "tiles_processed": tiles_processed,
        "tiles_enriched": tiles_enriched,
        "pages_done": pages_done,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint() -> Optional[dict]:
    """Load progress checkpoint if it exists."""
    try:
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# =============================================================================
# MAIN OPERATIONS
# =============================================================================

def check(redis_store: HMMRedisStore, session: requests.Session):
    """Verify anchors and show current stats."""
    log.info("=== CHECK ===")

    # Load anchors
    matrix, ids, types = load_anchors(redis_store)
    if matrix is None:
        return

    motif_count = sum(1 for t in types if t == "motif")
    theme_count = sum(1 for t in types if t == "theme")
    log.info(f"Anchors: {motif_count} motifs + {theme_count} themes = {len(ids)} total")
    log.info(f"Matrix shape: {matrix.shape}")

    # Weaviate tile counts
    for scale in ["search_512", "context_2048", "full_4096"]:
        gql = f"""{{ Aggregate {{ {WEAVIATE_CLASS}(
            where: {{ path: ["scale"], operator: Equal, valueText: "{scale}" }}
        ) {{ meta {{ count }} }} }} }}"""
        try:
            r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                             json={"query": gql}, timeout=10)
            count = r.json()["data"]["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
            log.info(f"  {scale}: {count:,} tiles")
        except Exception:
            log.warning(f"  {scale}: count failed")

    # Enriched counts
    for version_prefix in ["tier1", "tier2"]:
        gql = f"""{{ Aggregate {{ {WEAVIATE_CLASS}(
            where: {{ operator: And, operands: [
                {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }},
                {{ path: ["hmm_enrichment_version"], operator: Like, valueText: "{version_prefix}*" }}
            ] }}
        ) {{ meta {{ count }} }} }} }}"""
        try:
            r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                             json={"query": gql}, timeout=10)
            count = r.json()["data"]["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
            log.info(f"  {version_prefix} enriched: {count:,} tiles")
        except Exception:
            pass

    # Checkpoint
    cp = load_checkpoint()
    if cp:
        log.info(f"Checkpoint: {cp['tiles_processed']:,} processed, "
                 f"{cp['tiles_enriched']:,} enriched, "
                 f"{cp['pages_done']} pages, "
                 f"updated {cp['updated_at']}")


def run_classification(redis_store: HMMRedisStore, session: requests.Session,
                       limit: int = 0, resume: bool = False):
    """Run full Tier 1 classification."""
    log.info("=== RUN CLASSIFICATION ===")

    # Load anchors
    anchor_matrix, anchor_ids, anchor_types = load_anchors(redis_store)
    if anchor_matrix is None:
        return

    # Resume from checkpoint?
    cursor = None
    total_processed = 0
    total_enriched = 0
    total_skipped = 0
    pages_done = 0

    if resume:
        cp = load_checkpoint()
        if cp:
            cursor = cp["cursor"]
            total_processed = cp["tiles_processed"]
            total_enriched = cp["tiles_enriched"]
            pages_done = cp["pages_done"]
            log.info(f"Resuming from checkpoint: page {pages_done}, "
                     f"{total_processed:,} processed")

    t_start = time.time()
    remaining_limit = limit if limit > 0 else float('inf')
    empty_pages = 0  # Track consecutive pages with 0 matching tiles

    while remaining_limit > 0:
        # Always fetch full pages (cursor iterates ALL scales)
        objects, next_cursor = fetch_tile_page(session, cursor=cursor,
                                               scale="search_512",
                                               limit=PAGE_SIZE)

        # next_cursor empty = no more objects in collection
        if not next_cursor and not objects:
            log.info("No more tiles in collection")
            break

        if not objects:
            # Page had objects but none matched our scale - keep going
            empty_pages += 1
            if empty_pages > 100:
                log.info("100 consecutive pages with no matching tiles, stopping")
                break
            cursor = next_cursor
            pages_done += 1
            continue

        empty_pages = 0

        # Filter: extract vectors, check should_classify
        tile_uuids = []
        tile_vectors_list = []

        for obj in objects:
            additional = obj.get("_additional", {})
            uuid = additional.get("id", "")
            vector = additional.get("vector")

            if not vector or not uuid:
                total_skipped += 1
                continue

            if not should_classify(obj):
                total_skipped += 1
                continue

            tile_uuids.append(uuid)
            tile_vectors_list.append(vector)

        if tile_vectors_list:
            # Build numpy matrix and normalize
            tile_matrix = np.array(tile_vectors_list, dtype=np.float32)
            norms = np.linalg.norm(tile_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            tile_matrix = tile_matrix / norms

            # Classify
            classifications = classify_batch(
                tile_matrix, anchor_matrix, anchor_ids, anchor_types
            )

            # PATCH each tile
            page_enriched = 0
            for uuid, classification in zip(tile_uuids, classifications):
                payload = build_enrichment_patch(classification)
                try:
                    r = session.patch(
                        f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{uuid}",
                        json={"properties": payload},
                        timeout=10,
                    )
                    if r.status_code in (200, 204):
                        page_enriched += 1
                    else:
                        log.warning(f"PATCH failed {uuid}: {r.status_code}")
                except Exception as e:
                    log.warning(f"PATCH error {uuid}: {e}")

            total_enriched += page_enriched

        total_processed += len(objects)
        pages_done += 1
        cursor = next_cursor

        if limit > 0:
            remaining_limit -= len(objects)

        # Progress every 10 pages
        if pages_done % 10 == 0:
            elapsed = time.time() - t_start
            rate = total_processed / elapsed if elapsed > 0 else 0
            log.info(f"Page {pages_done}: {total_processed:,} processed, "
                     f"{total_enriched:,} enriched, "
                     f"{total_skipped:,} skipped, "
                     f"{rate:.0f} tiles/s")

        # Checkpoint every 20 pages
        if pages_done % 20 == 0:
            save_checkpoint(cursor, total_processed, total_enriched, pages_done)

    # Final checkpoint
    save_checkpoint(cursor or "", total_processed, total_enriched, pages_done)

    elapsed = time.time() - t_start
    log.info(
        f"Classification complete: {total_processed:,} processed, "
        f"{total_enriched:,} enriched, {total_skipped:,} skipped "
        f"in {elapsed:.1f}s ({pages_done} pages)"
    )


def propagate_to_parents(session: requests.Session):
    """Propagate enrichment from search_512 to context_2048 and full_4096 parents.

    For each parent tile: union of child dominant_motifs, max amplitudes.
    """
    log.info("=== PROPAGATE TO PARENTS ===")

    for parent_scale, child_scale in [("context_2048", "search_512"),
                                       ("full_4096", "context_2048")]:
        log.info(f"Propagating {child_scale} -> {parent_scale}")

        cursor = None
        page_num = 0
        enriched = 0
        empty_runs = 0

        while True:
            # Fetch parent tiles (cursor iterates all scales, filtered client-side)
            objects, next_cursor = fetch_tile_page(
                session, cursor=cursor, scale=parent_scale, limit=PAGE_SIZE
            )
            if not next_cursor and not objects:
                break
            if not objects:
                empty_runs += 1
                if empty_runs > 100:
                    break
                cursor = next_cursor
                page_num += 1
                continue
            empty_runs = 0

            for obj in objects:
                additional = obj.get("_additional", {})
                parent_uuid = additional.get("id", "")
                if not parent_uuid:
                    continue

                # Skip tier2
                version = obj.get("hmm_enrichment_version") or ""
                if version.startswith("tier2"):
                    continue

                # Find enriched children
                content_hash = obj.get("content_hash", "")
                if not content_hash:
                    continue

                child_gql = f"""{{
                    Get {{
                        {WEAVIATE_CLASS}(
                            where: {{ operator: And, operands: [
                                {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }},
                                {{ path: ["scale"], operator: Equal, valueText: "{child_scale}" }},
                                {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
                            ] }}
                            limit: 50
                        ) {{
                            dominant_motifs
                            motif_data_json
                        }}
                    }}
                }}"""

                try:
                    r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                                     json={"query": child_gql}, timeout=15)
                    if r.status_code != 200:
                        continue

                    children = (r.json().get("data", {}).get("Get", {})
                                .get(WEAVIATE_CLASS, []))
                    if not children:
                        continue

                    # Union dominant_motifs, merge motif_data (max similarity)
                    all_dominant = set()
                    motif_max = {}  # motif_id -> max similarity

                    for child in children:
                        dm = child.get("dominant_motifs") or []
                        all_dominant.update(dm)

                        md_json = child.get("motif_data_json") or "[]"
                        try:
                            md = json.loads(md_json)
                            for entry in md:
                                mid = entry["motif_id"]
                                sim = entry.get("similarity", 0)
                                if mid not in motif_max or sim > motif_max[mid]:
                                    motif_max[mid] = sim
                        except (json.JSONDecodeError, KeyError):
                            pass

                    if not motif_max:
                        continue

                    # Build parent motif_data from merged maximums
                    parent_motif_data = sorted([
                        {"motif_id": mid, "similarity": round(sim, 4),
                         "tier": "tier1_propagated"}
                        for mid, sim in motif_max.items()
                    ], key=lambda m: m["similarity"], reverse=True)

                    payload = {
                        "dominant_motifs": sorted(all_dominant),
                        "motif_data_json": json.dumps(parent_motif_data),
                        "hmm_enriched": True,
                        "hmm_enrichment_version": ENRICHMENT_VERSION,
                        "hmm_enriched_at": datetime.now(timezone.utc).isoformat(),
                        "hmm_consensus": False,
                        "hmm_gate_flags": ["TIER1_VECTOR", "AUTOMATED", "PROPAGATED"],
                    }

                    r = session.patch(
                        f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{parent_uuid}",
                        json={"properties": payload},
                        timeout=10,
                    )
                    if r.status_code in (200, 204):
                        enriched += 1

                except Exception as e:
                    log.warning(f"Propagation error for {parent_uuid}: {e}")

            cursor = next_cursor
            page_num += 1

            if page_num % 10 == 0:
                log.info(f"  {parent_scale} page {page_num}: {enriched} enriched")

        log.info(f"  {parent_scale}: {enriched} tiles enriched via propagation")


def stats(session: requests.Session):
    """Show coverage report."""
    log.info("=== COVERAGE STATS ===")

    # Total tiles by scale
    for scale in ["search_512", "context_2048", "full_4096"]:
        total_gql = f"""{{ Aggregate {{ {WEAVIATE_CLASS}(
            where: {{ path: ["scale"], operator: Equal, valueText: "{scale}" }}
        ) {{ meta {{ count }} }} }} }}"""

        enriched_gql = f"""{{ Aggregate {{ {WEAVIATE_CLASS}(
            where: {{ operator: And, operands: [
                {{ path: ["scale"], operator: Equal, valueText: "{scale}" }},
                {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
            ] }}
        ) {{ meta {{ count }} }} }} }}"""

        try:
            r1 = session.post(f"{WEAVIATE_URL}/v1/graphql",
                              json={"query": total_gql}, timeout=10)
            total = r1.json()["data"]["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]

            r2 = session.post(f"{WEAVIATE_URL}/v1/graphql",
                              json={"query": enriched_gql}, timeout=10)
            enriched = r2.json()["data"]["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]

            pct = (enriched / total * 100) if total > 0 else 0
            log.info(f"  {scale}: {enriched:,}/{total:,} enriched ({pct:.1f}%)")
        except Exception:
            log.warning(f"  {scale}: stats failed")

    # Top motifs by frequency (sample first 100 enriched tiles)
    sample_gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{ operator: And, operands: [
                    {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }},
                    {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                ] }}
                limit: 100
            ) {{
                dominant_motifs
            }}
        }}
    }}"""

    try:
        r = session.post(f"{WEAVIATE_URL}/v1/graphql",
                         json={"query": sample_gql}, timeout=15)
        objects = r.json()["data"]["Get"][WEAVIATE_CLASS]

        motif_counts = {}
        for obj in objects:
            dm = obj.get("dominant_motifs") or []
            for m in dm:
                motif_counts[m] = motif_counts.get(m, 0) + 1

        log.info(f"  Top motifs (from {len(objects)} sample tiles):")
        for mid, count in sorted(motif_counts.items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
            log.info(f"    {mid}: {count}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Tier 1 batch classification")
    parser.add_argument("--check", action="store_true",
                        help="Verify anchors, show stats")
    parser.add_argument("--canary", type=int, metavar="N",
                        help="Test on N tiles")
    parser.add_argument("--run", action="store_true",
                        help="Full classification")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (use with --run)")
    parser.add_argument("--propagate", action="store_true",
                        help="Propagate to parent tiles only")
    parser.add_argument("--stats", action="store_true",
                        help="Coverage report")
    args = parser.parse_args()

    if not any([args.check, args.canary, args.run, args.propagate, args.stats]):
        parser.print_help()
        return

    redis_store = HMMRedisStore()
    session = requests.Session()

    # Verify Weaviate
    try:
        r = session.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
        if r.status_code != 200:
            log.error(f"Weaviate not healthy: {r.status_code}")
            return
    except Exception as e:
        log.error(f"Weaviate unreachable: {e}")
        return

    if args.check:
        check(redis_store, session)

    if args.canary:
        run_classification(redis_store, session, limit=args.canary)

    if args.run:
        run_classification(redis_store, session, resume=args.resume)
        # Auto-propagate after full run
        propagate_to_parents(session)

    if args.propagate:
        propagate_to_parents(session)

    if args.stats:
        stats(session)


if __name__ == "__main__":
    main()
