#!/usr/bin/env python3
"""Build RELATES_TO edges in Neo4j for tiles sharing 2+ rare motifs.

Rare motifs = motifs expressed by fewer than 5% of HMMTile nodes.
Tiles sharing 2+ rare motifs share non-obvious thematic overlap that the
original HMM cross-reference edges (from enrichment) may not capture.

Run once on Spark 1 or 2. Safe to re-run (MERGE skips duplicates).
Checkpointed via Redis so it can be resumed after interruption.

Usage:
    python3 build_motif_cooccurrence_edges.py [--dry-run] [--reset-checkpoint]
"""

import argparse
import itertools
import json
import logging
import sys
import time
from collections import defaultdict

from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NEO4J_URI = "bolt://192.168.100.10:7689"
RARITY_THRESHOLD_PCT = 5.0   # motifs expressed by < 5% of tiles are "rare"
MIN_SHARED_MOTIFS = 2        # minimum shared rare motifs to create an edge
BATCH_SIZE = 500             # edges to create per Cypher batch
CHECKPOINT_KEY = "isma:motif_cooccurrence:checkpoint"


def get_redis():
    try:
        import redis
        r = redis.Redis(host="192.168.100.10", port=6379, decode_responses=True)
        r.ping()
        return r
    except Exception:
        log.warning("Redis unavailable — checkpointing disabled")
        return None


def load_checkpoint(r, key):
    if r is None:
        return None
    val = r.get(key)
    return json.loads(val) if val else None


def save_checkpoint(r, key, data):
    if r is None:
        return
    r.set(key, json.dumps(data), ex=86400 * 7)  # 7-day TTL


def get_rare_motifs(session, threshold_pct):
    """Return motif_id list for motifs expressed by < threshold_pct % of tiles."""
    result = session.run("MATCH (t:HMMTile) RETURN count(t) as total")
    total = result.single()["total"]
    threshold = int(total * threshold_pct / 100)
    log.info(f"Total HMMTile nodes: {total}, rarity threshold: <={threshold} tiles ({threshold_pct}%)")

    result = session.run("""
        MATCH (t:HMMTile)-[:EXPRESSES]->(m:HMMMotif)
        RETURN m.motif_id as motif_id, count(t) as tile_count
        ORDER BY tile_count ASC
    """)
    rare = []
    for rec in result:
        if rec["tile_count"] <= threshold:
            rare.append(rec["motif_id"])
            log.info(f"  Rare: {rec['motif_id']} ({rec['tile_count']} tiles)")
    log.info(f"Identified {len(rare)} rare motifs")
    return rare


def load_tile_motifs(session, rare_motifs):
    """Return dict: tile_id -> frozenset of rare motifs it expresses."""
    rare_set = set(rare_motifs)
    log.info("Loading tiles with rare motifs from Neo4j...")
    result = session.run("""
        MATCH (t:HMMTile)-[:EXPRESSES]->(m:HMMMotif)
        WHERE m.motif_id IN $rare_motifs
        RETURN t.tile_id as tile_id, collect(m.motif_id) as motifs
    """, rare_motifs=list(rare_motifs))

    tile_motifs = {}
    for rec in result:
        tile_id = rec["tile_id"]
        motifs = frozenset(m for m in rec["motifs"] if m in rare_set)
        if motifs:
            tile_motifs[tile_id] = motifs

    log.info(f"Loaded {len(tile_motifs)} tiles expressing at least one rare motif")
    return tile_motifs


def find_cooccurrence_pairs(tile_motifs, min_shared):
    """Find (tile_id_a, tile_id_b, shared_motifs) where |shared| >= min_shared.

    Uses motif-pair index to avoid O(N^2) brute force:
    For each pair of rare motifs, find tiles expressing both, then tally shared motif
    counts across motif-pair groups.
    """
    log.info("Building motif → tile index...")
    motif_to_tiles = defaultdict(set)
    for tile_id, motifs in tile_motifs.items():
        for motif in motifs:
            motif_to_tiles[motif].add(tile_id)

    log.info("Finding tile pairs with 2+ shared rare motifs...")
    # pair -> set of shared motifs
    pair_motifs = defaultdict(set)
    motif_list = sorted(motif_to_tiles.keys())

    for m1, m2 in itertools.combinations(motif_list, 2):
        shared_tiles = motif_to_tiles[m1] & motif_to_tiles[m2]
        for t1, t2 in itertools.combinations(sorted(shared_tiles), 2):
            key = (t1, t2) if t1 < t2 else (t2, t1)
            pair_motifs[key].add(m1)
            pair_motifs[key].add(m2)

    qualifying = [
        (t1, t2, sorted(motifs))
        for (t1, t2), motifs in pair_motifs.items()
        if len(motifs) >= min_shared
    ]
    log.info(f"Found {len(qualifying)} tile pairs with {min_shared}+ shared rare motifs")
    return qualifying


def count_existing_cooccurrence_edges(session):
    result = session.run("""
        MATCH ()-[r:RELATES_TO {type: 'motif_cooccurrence'}]->()
        RETURN count(r) as cnt
    """)
    return result.single()["cnt"]


def create_edges_batch(session, batch, dry_run):
    """MERGE a batch of RELATES_TO edges. Returns count created."""
    if dry_run:
        return len(batch)

    result = session.run("""
        UNWIND $pairs as pair
        MATCH (t1:HMMTile {tile_id: pair.t1})
        MATCH (t2:HMMTile {tile_id: pair.t2})
        MERGE (t1)-[r:RELATES_TO {type: 'motif_cooccurrence'}]-(t2)
        ON CREATE SET
            r.motifs = pair.motifs,
            r.shared_count = pair.shared_count,
            r.created_at = datetime()
        RETURN count(r) as created
    """, pairs=[
        {"t1": t1, "t2": t2, "motifs": motifs, "shared_count": len(motifs)}
        for t1, t2, motifs in batch
    ])
    rec = result.single()
    return rec["created"] if rec else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute pairs but don't write to Neo4j")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Ignore saved checkpoint and start fresh")
    args = parser.parse_args()

    r = get_redis()
    if args.reset_checkpoint and r:
        r.delete(CHECKPOINT_KEY)
        log.info("Checkpoint cleared")

    checkpoint = load_checkpoint(r, CHECKPOINT_KEY) if not args.reset_checkpoint else None

    driver = GraphDatabase.driver(NEO4J_URI, auth=None)
    try:
        with driver.session() as session:
            # Step 1: Get rare motifs
            rare_motifs = get_rare_motifs(session, RARITY_THRESHOLD_PCT)
            if not rare_motifs:
                log.error("No rare motifs found — check threshold or graph data")
                sys.exit(1)

            # Step 2: Load tile → rare-motif mapping
            tile_motifs = load_tile_motifs(session, rare_motifs)

            # Step 3: Find qualifying pairs
            pairs = find_cooccurrence_pairs(tile_motifs, MIN_SHARED_MOTIFS)

            if not pairs:
                log.info("No qualifying pairs found — nothing to create")
                return

            # Step 4: Batch-create edges with checkpointing
            existing = count_existing_cooccurrence_edges(session)
            log.info(f"Existing motif_cooccurrence RELATES_TO edges: {existing}")

            start_idx = checkpoint.get("next_idx", 0) if checkpoint else 0
            if start_idx > 0:
                log.info(f"Resuming from checkpoint at index {start_idx}/{len(pairs)}")

            total_created = checkpoint.get("total_created", 0) if checkpoint else 0
            t0 = time.monotonic()

            for batch_start in range(start_idx, len(pairs), BATCH_SIZE):
                batch = pairs[batch_start: batch_start + BATCH_SIZE]
                created = create_edges_batch(session, batch, args.dry_run)
                total_created += created
                next_idx = batch_start + len(batch)

                elapsed = time.monotonic() - t0
                rate = (next_idx - start_idx) / elapsed if elapsed > 0 else 0
                remaining = (len(pairs) - next_idx) / rate if rate > 0 else 0

                log.info(
                    f"Batch {batch_start}-{next_idx}/{len(pairs)} | "
                    f"edges created: {total_created} | "
                    f"{rate:.0f} pairs/s | "
                    f"ETA: {remaining:.0f}s"
                )

                if not args.dry_run:
                    save_checkpoint(r, CHECKPOINT_KEY, {
                        "next_idx": next_idx,
                        "total_created": total_created,
                        "total_pairs": len(pairs),
                    })

            if args.dry_run:
                log.info(f"DRY RUN complete: would create up to {len(pairs)} edges")
            else:
                log.info(f"Done. Created {total_created} motif_cooccurrence RELATES_TO edges")
                if r:
                    r.delete(CHECKPOINT_KEY)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
