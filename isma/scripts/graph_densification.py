#!/usr/bin/env python3
"""
ISMA Graph Densification — Phase 6B

Creates RELATES_TO edges between HMMTile nodes in Neo4j that share
dominant_motifs with Jaccard similarity ≥ 0.3.

Algorithm:
  1. Use Neo4j co-EXPRESSES pattern: find tile pairs (a, b) that both
     EXPRESSES at least one common HMMMotif
  2. Compute Jaccard(a.motifs, b.motifs) from shared + union counts
  3. Create RELATES_TO {jaccard: <score>, source: "motif_overlap"} edge
     where Jaccard ≥ 0.3

Uses streaming batches to avoid OOM on 600K tiles × ~10M expected edges.

Usage:
    python3 graph_densification.py                    # Full run
    python3 graph_densification.py --dry-run          # Count edges only
    python3 graph_densification.py --threshold 0.4    # Custom Jaccard threshold
    python3 graph_densification.py --motif HMM.SACRED_TRUST  # Single motif only
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neo4j import GraphDatabase

log = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://192.168.100.10:7689")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")

BATCH_SIZE = 500          # Edge creation batch size (Cypher UNWIND)
JACCARD_THRESHOLD = 0.3   # Minimum Jaccard to create an edge
MAX_PAIRS_PER_MOTIF = 50_000  # Safety cap per motif (very dense motifs skipped)


def get_driver():
    auth = (NEO4J_USER, NEO4J_PASS) if NEO4J_USER else None
    return GraphDatabase.driver(NEO4J_URI, auth=auth, max_connection_lifetime=3600)


def get_all_motifs(driver) -> List[str]:
    """Return all HMMMotif IDs in the graph."""
    with driver.session() as s:
        result = s.run("MATCH (m:HMMMotif) RETURN m.motif_id AS mid ORDER BY mid")
        return [rec["mid"] for rec in result]


def iter_tile_pairs_for_motif(
    driver, motif_id: str, max_pairs: int = MAX_PAIRS_PER_MOTIF
) -> Iterator[Tuple[str, str, int]]:
    """Yield (tile_a_id, tile_b_id, shared_count) pairs for tiles co-expressing motif_id.

    Both tiles must EXPRESSES the same motif. Returns ordered pairs (a < b)
    to avoid creating duplicate edges in both directions.
    """
    with driver.session() as s:
        # Count pairs first to detect runaway motifs
        count_result = s.run("""
            MATCH (a:HMMTile)-[:EXPRESSES]->(m:HMMMotif {motif_id: $mid})<-[:EXPRESSES]-(b:HMMTile)
            WHERE a.tile_id < b.tile_id
            RETURN count(*) AS pair_count
        """, mid=motif_id)
        count = count_result.single()["pair_count"]
        if count > max_pairs:
            log.warning(
                "Motif %s has %d pairs (> %d limit) — skipping to avoid OOM",
                motif_id, count, max_pairs,
            )
            return

        result = s.run("""
            MATCH (a:HMMTile)-[:EXPRESSES]->(m:HMMMotif {motif_id: $mid})<-[:EXPRESSES]-(b:HMMTile)
            WHERE a.tile_id < b.tile_id
            RETURN a.tile_id AS a_id, b.tile_id AS b_id, 1 AS shared
        """, mid=motif_id)
        for rec in result:
            yield rec["a_id"], rec["b_id"], rec["shared"]


def get_motif_counts(driver, tile_ids: List[str]) -> Dict[str, int]:
    """Return {tile_id: motif_count} for a batch of tiles."""
    if not tile_ids:
        return {}
    with driver.session() as s:
        result = s.run("""
            MATCH (t:HMMTile)-[:EXPRESSES]->(:HMMMotif)
            WHERE t.tile_id IN $ids
            RETURN t.tile_id AS tid, count(*) AS cnt
        """, ids=tile_ids)
        return {rec["tid"]: rec["cnt"] for rec in result}


def compute_jaccard(shared: int, count_a: int, count_b: int) -> float:
    """Jaccard = |intersection| / |union| = shared / (|A| + |B| - shared)."""
    union = count_a + count_b - shared
    if union <= 0:
        return 0.0
    return shared / union


def create_edges_batch(driver, edges: List[Dict]) -> int:
    """Create RELATES_TO edges in a single UNWIND batch. Returns count created."""
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
            RETURN count(r) AS created
        """, edges=edges)
        rec = result.single()
        return rec["created"] if rec else 0


def run_densification(
    threshold: float = JACCARD_THRESHOLD,
    motif_filter: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """Main densification loop.

    For each motif:
      1. Get all tile pairs co-expressing it (shared_count=1 per motif)
      2. Accumulate shared counts across motifs for each pair
      3. Fetch total motif counts for pairs above threshold
      4. Create RELATES_TO edges for pairs with Jaccard ≥ threshold
    """
    driver = get_driver()

    if motif_filter:
        motifs = [motif_filter]
    else:
        motifs = get_all_motifs(driver)

    log.info("Processing %d motifs (threshold=%.2f)", len(motifs), threshold)

    # Accumulate shared motif counts per tile pair across all motifs
    # pair_shared: {(a_id, b_id): shared_count}
    pair_shared: Dict[Tuple[str, str], int] = {}
    skipped_motifs = 0

    t0 = time.time()
    for i, motif_id in enumerate(motifs):
        pairs_this_motif = 0
        for a_id, b_id, _ in iter_tile_pairs_for_motif(driver, motif_id):
            key = (a_id, b_id)
            pair_shared[key] = pair_shared.get(key, 0) + 1
            pairs_this_motif += 1

        if pairs_this_motif == 0 and i > 0:
            skipped_motifs += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            log.info(
                "[%d/%d motifs] pairs tracked=%d skipped=%d elapsed=%.0fs",
                i + 1, len(motifs), len(pair_shared), skipped_motifs, elapsed,
            )

    log.info(
        "Motif scan complete: %d unique pairs tracked across %d motifs (%.0fs)",
        len(pair_shared), len(motifs), time.time() - t0,
    )

    if dry_run:
        # Estimate edges without fetching motif counts
        candidate_pairs = [(a, b, s) for (a, b), s in pair_shared.items() if s >= 1]
        log.info("DRY RUN: %d candidate pairs (shared ≥ 1), threshold=%.2f", len(candidate_pairs), threshold)
        driver.close()
        return {"candidate_pairs": len(candidate_pairs), "dry_run": True}

    # Now compute Jaccard for each pair with shared ≥ 1
    # Collect all unique tile IDs we need motif counts for
    candidate_pairs = [(a, b, s) for (a, b), s in pair_shared.items() if s >= 1]
    log.info("Computing Jaccard for %d candidate pairs...", len(candidate_pairs))

    # Batch-fetch motif counts for all tiles involved
    all_tile_ids = list({tid for a, b, _ in candidate_pairs for tid in (a, b)})
    log.info("Fetching motif counts for %d tiles...", len(all_tile_ids))

    motif_counts: Dict[str, int] = {}
    chunk_size = 1000
    for i in range(0, len(all_tile_ids), chunk_size):
        chunk = all_tile_ids[i:i + chunk_size]
        motif_counts.update(get_motif_counts(driver, chunk))
        if (i // chunk_size + 1) % 10 == 0:
            log.info("Fetched counts for %d/%d tiles", min(i + chunk_size, len(all_tile_ids)), len(all_tile_ids))

    # Filter to pairs meeting Jaccard threshold
    edges_to_create = []
    edges_below_threshold = 0
    for a_id, b_id, shared in candidate_pairs:
        count_a = motif_counts.get(a_id, 0)
        count_b = motif_counts.get(b_id, 0)
        if count_a == 0 or count_b == 0:
            continue
        j = compute_jaccard(shared, count_a, count_b)
        if j >= threshold:
            edges_to_create.append({"a_id": a_id, "b_id": b_id, "jaccard": round(j, 4)})
        else:
            edges_below_threshold += 1

    log.info(
        "Jaccard filter: %d edges to create, %d below threshold (%.2f)",
        len(edges_to_create), edges_below_threshold, threshold,
    )

    # Create edges in batches
    total_created = 0
    t1 = time.time()
    for i in range(0, len(edges_to_create), batch_size):
        batch = edges_to_create[i:i + batch_size]
        created = create_edges_batch(driver, batch)
        total_created += created
        if (i // batch_size + 1) % 20 == 0:
            elapsed = time.time() - t1
            rate = total_created / elapsed if elapsed > 0 else 0
            log.info(
                "[%d/%d edges] created=%d rate=%.0f/s",
                min(i + batch_size, len(edges_to_create)), len(edges_to_create),
                total_created, rate,
            )

    elapsed_total = time.time() - t0
    log.info(
        "Densification complete: %d RELATES_TO edges created in %.0fs",
        total_created, elapsed_total,
    )

    driver.close()
    return {
        "motifs_processed": len(motifs),
        "motifs_skipped": skipped_motifs,
        "candidate_pairs": len(candidate_pairs),
        "edges_created": total_created,
        "edges_below_threshold": edges_below_threshold,
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
                        help=f"Jaccard threshold for RELATES_TO (default: {JACCARD_THRESHOLD})")
    parser.add_argument("--motif", type=str, default=None,
                        help="Process only a single motif (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count candidate pairs only, don't create edges")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Edge creation batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    result = run_densification(
        threshold=args.threshold,
        motif_filter=args.motif,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    print("\nResult:", result)


if __name__ == "__main__":
    main()
