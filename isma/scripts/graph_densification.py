#!/usr/bin/env python3
"""
ISMA Graph Densification — Phase 6B

Creates RELATES_TO edges between HMMTile nodes in Neo4j that share
dominant_motifs with Jaccard similarity ≥ 0.3.

Algorithm:
  1. Load all HMMTile {tile_id, dominant_motifs} into Python memory (~75MB)
  2. Build Python inverted index: {motif → [tile_ids]}
  3. For each motif with ≤ MAX_TILES_PER_MOTIF tiles:
       - Generate all tile pairs sharing that motif
       - Compute Jaccard(a.motifs, b.motifs) using Python set operations
       - Accumulate per-pair: best Jaccard seen across all shared motifs
  4. Create RELATES_TO {jaccard, source} edges for pairs ≥ JACCARD_THRESHOLD
     via UNWIND batches (idempotent MERGE)

Why Python-side inverted index vs Neo4j traversal:
  - Neo4j cross-joins all-pairs is O(N²) — unbound for dense motifs
  - Python inverted index bounds work to small/specific motifs
  - Dense motifs (TECHNICAL_INFRASTRUCTURE: 17K tiles → 145M pairs) are
    skipped via MAX_TILES_PER_MOTIF, avoiding OOM

Usage:
    python3 graph_densification.py                    # Full run
    python3 graph_densification.py --dry-run          # Count pairs only
    python3 graph_densification.py --threshold 0.4    # Stricter Jaccard
    python3 graph_densification.py --max-tiles 1000   # Only most specific motifs
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neo4j import GraphDatabase

log = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://192.168.100.10:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")

BATCH_SIZE = 500          # Edge creation batch size
JACCARD_THRESHOLD = 0.3   # Minimum Jaccard to create an edge
MAX_TILES_PER_MOTIF = 2000  # Skip motifs with more tiles than this (too generic)
LOAD_BATCH_SIZE = 10_000  # How many HMMTile nodes to load per Neo4j query


def get_driver():
    auth = (NEO4J_USER, NEO4J_PASS) if NEO4J_USER else None
    return GraphDatabase.driver(NEO4J_URI, auth=auth, max_connection_lifetime=3600)


def load_all_tiles(driver) -> Dict[str, Set[str]]:
    """Load all HMMTile dominant_motifs into memory.

    Returns {tile_id: frozenset(dominant_motifs)}.
    Loads in batches to avoid timeout on large result sets.
    """
    tile_motifs: Dict[str, Set[str]] = {}
    offset = 0

    with driver.session() as s:
        # Get total count
        count_result = s.run(
            "MATCH (t:HMMTile) WHERE t.dominant_motifs IS NOT NULL "
            "RETURN count(t) AS cnt"
        )
        total = count_result.single()["cnt"]
        log.info("Loading %d HMMTile nodes with dominant_motifs...", total)

        while True:
            result = s.run("""
                MATCH (t:HMMTile)
                WHERE t.dominant_motifs IS NOT NULL AND size(t.dominant_motifs) > 0
                RETURN t.tile_id AS tid, t.dominant_motifs AS motifs
                ORDER BY t.tile_id
                SKIP $skip LIMIT $limit
            """, skip=offset, limit=LOAD_BATCH_SIZE)

            batch = [(rec["tid"], rec["motifs"]) for rec in result]
            if not batch:
                break

            for tid, motifs in batch:
                tile_motifs[tid] = frozenset(motifs)

            offset += len(batch)
            if offset % 50_000 == 0:
                log.info("  Loaded %d/%d tiles...", offset, total)

            if len(batch) < LOAD_BATCH_SIZE:
                break

    log.info("Loaded %d tiles with motifs", len(tile_motifs))
    return tile_motifs


def build_inverted_index(
    tile_motifs: Dict[str, Set[str]],
    max_tiles: int = MAX_TILES_PER_MOTIF,
) -> Dict[str, List[str]]:
    """Build motif → [tile_ids] inverted index.

    Skips motifs with more than max_tiles tiles (too generic for meaningful Jaccard).
    Returns only processable motifs.
    """
    # First pass: count tiles per motif
    motif_counts: Dict[str, int] = defaultdict(int)
    for motifs in tile_motifs.values():
        for m in motifs:
            motif_counts[m] += 1

    # Filter to motifs within the processable range
    processable = {m for m, cnt in motif_counts.items() if cnt <= max_tiles}
    skipped = {m: cnt for m, cnt in motif_counts.items() if cnt > max_tiles}

    if skipped:
        log.info(
            "Skipping %d dense motifs (> %d tiles): %s",
            len(skipped), max_tiles,
            ", ".join(f"{m.split('.')[-1]}({c})" for m, c in sorted(skipped.items(), key=lambda x: -x[1]))
        )

    # Second pass: build index for processable motifs only
    inv_index: Dict[str, List[str]] = defaultdict(list)
    for tid, motifs in tile_motifs.items():
        for m in motifs:
            if m in processable:
                inv_index[m].append(tid)

    log.info(
        "Built inverted index: %d processable motifs, largest has %d tiles",
        len(inv_index),
        max(len(v) for v in inv_index.values()) if inv_index else 0,
    )
    return dict(inv_index)


def compute_jaccard_pairs(
    tile_motifs: Dict[str, Set[str]],
    inv_index: Dict[str, List[str]],
    threshold: float,
) -> Dict[Tuple[str, str], float]:
    """Compute Jaccard similarity for all tile pairs sharing at least one motif.

    Returns {(a_id, b_id): best_jaccard} for pairs where Jaccard ≥ threshold.
    Uses Python set operations for efficiency.
    """
    # Accumulate best Jaccard per ordered pair (a < b)
    pair_jaccard: Dict[Tuple[str, str], float] = {}
    total_pairs_checked = 0
    total_pairs_kept = 0

    for motif_id, tile_ids in sorted(inv_index.items()):
        n = len(tile_ids)
        pairs_this_motif = n * (n - 1) // 2
        total_pairs_checked += pairs_this_motif

        for i in range(len(tile_ids)):
            for j in range(i + 1, len(tile_ids)):
                a_id, b_id = tile_ids[i], tile_ids[j]
                if a_id > b_id:
                    a_id, b_id = b_id, a_id

                key = (a_id, b_id)
                if key in pair_jaccard:
                    # Already computed and above threshold — may need to recompute
                    # with full shared count, but for simplicity use cached value
                    continue

                motifs_a = tile_motifs.get(a_id, frozenset())
                motifs_b = tile_motifs.get(b_id, frozenset())
                if not motifs_a or not motifs_b:
                    continue

                shared = len(motifs_a & motifs_b)
                union = len(motifs_a | motifs_b)
                if union == 0:
                    continue

                j_score = shared / union
                if j_score >= threshold:
                    pair_jaccard[key] = max(pair_jaccard.get(key, 0.0), j_score)
                    total_pairs_kept += 1

    log.info(
        "Jaccard complete: %d pairs checked, %d pairs ≥ %.2f threshold",
        total_pairs_checked, total_pairs_kept, threshold,
    )
    return pair_jaccard


def create_edges_batch(driver, edges: List[Dict]) -> int:
    """Create RELATES_TO edges via UNWIND batch. Returns count created."""
    if not edges:
        return 0
    with driver.session() as s:
        result = s.run("""
            UNWIND $edges AS edge
            MATCH (a:HMMTile {tile_id: edge.a_id})
            MATCH (b:HMMTile {tile_id: edge.b_id})
            MERGE (a)-[r:RELATES_TO]-(b)
            ON CREATE SET r.jaccard = edge.jaccard,
                          r.source = "motif_overlap",
                          r.created_at = datetime()
            ON MATCH SET r.jaccard = CASE WHEN edge.jaccard > r.jaccard
                                     THEN edge.jaccard ELSE r.jaccard END
            RETURN count(r) AS cnt
        """, edges=edges)
        rec = result.single()
        return rec["cnt"] if rec else 0


def run_densification(
    threshold: float = JACCARD_THRESHOLD,
    max_tiles: int = MAX_TILES_PER_MOTIF,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Main densification pipeline."""
    driver = get_driver()
    t0 = time.time()

    # Step 1: Load all tiles
    tile_motifs = load_all_tiles(driver)

    # Step 2: Build inverted index (skip dense motifs)
    inv_index = build_inverted_index(tile_motifs, max_tiles=max_tiles)

    if not inv_index:
        log.warning("No processable motifs found with max_tiles=%d", max_tiles)
        driver.close()
        return {"edges_created": 0, "reason": "no_processable_motifs"}

    # Estimate total pairs
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in inv_index.values())
    log.info("Estimated total pairs to check: %s", f"{total_pairs:,}")

    if dry_run:
        driver.close()
        return {
            "processable_motifs": len(inv_index),
            "estimated_pairs": total_pairs,
            "dry_run": True,
        }

    # Step 3: Compute Jaccard for all pairs
    pair_jaccard = compute_jaccard_pairs(tile_motifs, inv_index, threshold)
    log.info("Found %d pairs with Jaccard ≥ %.2f", len(pair_jaccard), threshold)

    # Step 4: Create edges in batches
    edges_list = [
        {"a_id": a, "b_id": b, "jaccard": round(j, 4)}
        for (a, b), j in pair_jaccard.items()
    ]

    total_created = 0
    t1 = time.time()
    for i in range(0, len(edges_list), batch_size):
        batch = edges_list[i:i + batch_size]
        created = create_edges_batch(driver, batch)
        total_created += created
        if (i // batch_size + 1) % 20 == 0:
            elapsed = time.time() - t1
            rate = total_created / elapsed if elapsed > 0 else 0
            log.info(
                "[%d/%d edges] created=%d rate=%.0f/s",
                min(i + batch_size, len(edges_list)), len(edges_list),
                total_created, rate,
            )

    elapsed_total = time.time() - t0
    log.info(
        "Densification complete: %d RELATES_TO edges in %.0fs",
        total_created, elapsed_total,
    )

    driver.close()
    return {
        "tiles_loaded": len(tile_motifs),
        "processable_motifs": len(inv_index),
        "pairs_checked": sum(len(v) * (len(v) - 1) // 2 for v in inv_index.values()),
        "pairs_above_threshold": len(pair_jaccard),
        "edges_created": total_created,
        "threshold": threshold,
        "elapsed_seconds": round(elapsed_total, 1),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="ISMA Graph Densification")
    parser.add_argument("--threshold", type=float, default=JACCARD_THRESHOLD,
                        help=f"Jaccard threshold (default: {JACCARD_THRESHOLD})")
    parser.add_argument("--max-tiles", type=int, default=MAX_TILES_PER_MOTIF,
                        help=f"Max tiles per motif to process (default: {MAX_TILES_PER_MOTIF})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count pairs only, don't create edges")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Edge creation batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    result = run_densification(
        threshold=args.threshold,
        max_tiles=args.max_tiles,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    print("\nResult:", result)


if __name__ == "__main__":
    main()
