#!/usr/bin/env python3
"""
Neo4j HMMTile Backfill

Reads all hmm_enriched tiles from Weaviate (682K) and creates the missing
HMMTile nodes + EXPRESSES edges in Neo4j (currently only 33K exist).

Pipeline gap: pre-Phase-5.5 triple-write fix left ~649K tiles in Weaviate
without corresponding Neo4j nodes. This script closes that gap idempotently.

Usage:
    python3 neo4j_backfill_hmm.py
    python3 neo4j_backfill_hmm.py --dry-run   # count only, no writes
"""

import json
import sys
import time
import argparse
from datetime import datetime

import requests
from neo4j import GraphDatabase

WEAVIATE_URL = "http://192.168.100.10:8088"
NEO4J_URI = "bolt://192.168.100.10:7689"
CLASS = "ISMA_Quantum"
PAGE_SIZE = 200
NEO4J_BATCH = 500

def weaviate_page(after_id=None):
    """Cursor-paginate via REST API (supports after without where filter limitation)."""
    params = {"class": CLASS, "limit": PAGE_SIZE}
    if after_id:
        params["after"] = after_id
    r = requests.get(f"{WEAVIATE_URL}/v1/objects", params=params, timeout=60)
    r.raise_for_status()
    objects = r.json().get("objects", [])
    # Normalize to same shape as GraphQL response
    result = []
    for obj in objects:
        props = obj.get("properties", {})
        if not props.get("hmm_enriched"):
            continue  # skip non-enriched tiles
        result.append({
            "content_hash": props.get("content_hash"),
            "dominant_motifs": props.get("dominant_motifs") or [],
            "rosetta_summary": props.get("rosetta_summary") or "",
            "platform": props.get("platform") or "",
            "source_file": props.get("source_file") or "",
            "session_id": props.get("session_id") or "",
            "_additional": {"id": obj.get("id", "")},
        })
    return result, objects  # return raw objects for cursor


def load_existing_hashes(driver):
    """Load all existing HMMTile content_hash values into a set."""
    print("Loading existing Neo4j HMMTile hashes...", flush=True)
    with driver.session() as s:
        result = s.run("MATCH (t:HMMTile) RETURN t.content_hash AS h")
        hashes = {row["h"] for row in result if row["h"]}
    print(f"  {len(hashes):,} existing HMMTile nodes", flush=True)
    return hashes


def write_batch(driver, tiles):
    """Batch upsert HMMTile nodes + EXPRESSES edges."""
    # Step 1: upsert HMMTile nodes
    node_query = """
    UNWIND $tiles AS t
    MERGE (n:HMMTile {tile_id: t.content_hash})
    ON CREATE SET
        n.created_at     = datetime(),
        n.content_hash   = t.content_hash,
        n.artifact_id    = 'backfill:' + t.content_hash,
        n.dominant_motifs = t.dominant_motifs,
        n.rosetta_summary = t.rosetta_summary,
        n.platform       = t.platform,
        n.source_file    = t.source_file,
        n.session_id     = t.session_id,
        n.enrichment_version = 'weaviate_backfill_v1',
        n.index          = 0,
        n.start_char     = 0,
        n.end_char       = 0,
        n.estimated_tokens = 0,
        n.layer          = '',
        n.scale          = ''
    ON MATCH SET
        n.dominant_motifs = t.dominant_motifs,
        n.rosetta_summary = t.rosetta_summary
    """

    # Step 2: create EXPRESSES edges (tile -> motif)
    edge_query = """
    UNWIND $pairs AS p
    MATCH (t:HMMTile {tile_id: p.tile_id})
    MERGE (m:HMMMotif {motif_id: p.motif_id})
    ON CREATE SET m.created_at = datetime()
    MERGE (t)-[e:EXPRESSES]->(m)
    ON CREATE SET e.amp = 0.7, e.confidence = 0.7, e.source = 'backfill'
    """

    pairs = []
    for t in tiles:
        for motif in (t.get("dominant_motifs") or []):
            if motif:
                pairs.append({"tile_id": t["content_hash"], "motif_id": motif})

    with driver.session() as s:
        s.run(node_query, tiles=tiles)
        if pairs:
            s.run(edge_query, pairs=pairs)


def run(dry_run=False):
    driver = GraphDatabase.driver(NEO4J_URI, auth=None)

    existing = load_existing_hashes(driver)
    print(f"\nFetching from Weaviate (hmm_enriched=True, page_size={PAGE_SIZE})...\n", flush=True)

    total_seen = 0
    total_new = 0
    total_written = 0
    after_id = None
    pending = []
    start = time.time()

    while True:
        enriched, raw_objects = weaviate_page(after_id)
        if not raw_objects:
            break

        for obj in enriched:
            ch = obj.get("content_hash")
            if not ch:
                continue
            total_seen += 1
            if ch not in existing:
                total_new += 1
                pending.append({
                    "content_hash": ch,
                    "dominant_motifs": obj.get("dominant_motifs") or [],
                    "rosetta_summary": obj.get("rosetta_summary") or "",
                    "platform": obj.get("platform") or "",
                    "source_file": obj.get("source_file") or "",
                    "session_id": obj.get("session_id") or "",
                })

            if not dry_run and len(pending) >= NEO4J_BATCH:
                write_batch(driver, pending)
                total_written += len(pending)
                pending = []

        after_id = raw_objects[-1]["id"]
        elapsed = time.time() - start
        rate = total_seen / elapsed if elapsed > 0 else 0
        print(
            f"  seen={total_seen:,}  new={total_new:,}  written={total_written:,}"
            f"  rate={rate:.0f}/s  after={after_id[:8] if after_id else '?'}",
            flush=True,
        )

    # flush remainder
    if not dry_run and pending:
        write_batch(driver, pending)
        total_written += len(pending)

    elapsed = time.time() - start
    print(f"\n{'DRY RUN — ' if dry_run else ''}Done in {elapsed:.1f}s", flush=True)
    print(f"  Total seen:    {total_seen:,}", flush=True)
    print(f"  New to write:  {total_new:,}", flush=True)
    print(f"  Written:       {total_written:,}", flush=True)

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
