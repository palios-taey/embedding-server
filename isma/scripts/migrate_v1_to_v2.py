"""
Migrate ISMA_Quantum (v1) tiles to ISMA_Quantum_v2 canonical memory objects.

Two-pass migration:
  Pass 1: Scan all v1 tiles, build content_hash groups (lightweight metadata only)
  Pass 2: For each group, fetch representative vector + build v2 object + write

~74K unique content_hashes, ~96K vector fetches. Estimated runtime: 30-45 minutes.
Checkpointed to Redis. Resumable.

Usage:
    python3 -m isma.scripts.migrate_v1_to_v2 [--batch-size 100] [--resume] [--dry-run]
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import redis
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

WEAVIATE_URL = "http://192.168.100.10:8088"
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379
CHECKPOINT_KEY = "isma:v2_migration:checkpoint"
V2_CLASS = "ISMA_Quantum_v2"


def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def graphql(query: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                json={"query": query},
                timeout=120,
            )
            data = r.json()
            if "errors" in data:
                log.warning("GraphQL error: %s", data["errors"])
            return data
        except Exception as e:
            log.warning("GraphQL request failed (attempt %d): %s", attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(2)
    return {}


def fetch_object(uuid: str) -> Optional[dict]:
    """Fetch a single v1 object with vector and all properties."""
    try:
        r = requests.get(
            f"{WEAVIATE_URL}/v1/objects/ISMA_Quantum/{uuid}",
            params={"include": "vector"},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug("Failed to fetch %s: %s", uuid, e)
    return None


# ── Pass 1: Scan and Group ──────────────────────────────────────

class ContentGroup:
    """Lightweight representation of a content_hash group."""
    __slots__ = [
        "content_hash", "rep_uuid", "rosetta_uuid",
        "tile_ids_512", "tile_ids_2048", "tile_ids_4096",
        "platform", "source_type", "source_file", "session_id",
        "document_id", "loaded_at", "total_tokens",
        "hmm_enriched", "hmm_enriched_at",
    ]

    def __init__(self, content_hash: str):
        self.content_hash = content_hash
        self.rep_uuid = None  # UUID of representative search_512 tile (tile_index=0)
        self.rosetta_uuid = None  # UUID of rosetta tile
        self.tile_ids_512: List[str] = []
        self.tile_ids_2048: List[str] = []
        self.tile_ids_4096: List[str] = []
        self.platform = ""
        self.source_type = ""
        self.source_file = ""
        self.session_id = ""
        self.document_id = ""
        self.loaded_at = ""
        self.total_tokens = 0
        self.hmm_enriched = False
        self.hmm_enriched_at = ""


def pass1_scan() -> Dict[str, ContentGroup]:
    """Scan all v1 tiles and build content_hash groups."""
    groups: Dict[str, ContentGroup] = {}
    cursor = None
    total = 0
    t0 = time.monotonic()

    log.info("Pass 1: Scanning all v1 tiles...")

    while True:
        if cursor:
            q = (
                '{ Get { ISMA_Quantum(limit: 10000, after: "%s") '
                "{ content_hash scale tile_index token_count "
                "platform source_type source_file session_id document_id "
                "loaded_at hmm_enriched hmm_enriched_at "
                "_additional { id } } } }" % cursor
            )
        else:
            q = (
                "{ Get { ISMA_Quantum(limit: 10000) "
                "{ content_hash scale tile_index token_count "
                "platform source_type source_file session_id document_id "
                "loaded_at hmm_enriched hmm_enriched_at "
                "_additional { id } } } }"
            )

        data = graphql(q)
        tiles = data.get("data", {}).get("Get", {}).get("ISMA_Quantum", [])

        if not tiles:
            break

        for tile in tiles:
            ch = tile.get("content_hash")
            if not ch:
                continue

            uuid = tile["_additional"]["id"]
            scale = tile.get("scale", "")

            if ch not in groups:
                groups[ch] = ContentGroup(ch)
            g = groups[ch]

            # Track tile UUIDs by scale
            if scale == "search_512":
                g.tile_ids_512.append(uuid)
                # Pick representative: lowest tile_index
                if g.rep_uuid is None or (tile.get("tile_index") or 999) == 0:
                    g.rep_uuid = uuid
            elif scale == "context_2048":
                g.tile_ids_2048.append(uuid)
            elif scale == "full_4096":
                g.tile_ids_4096.append(uuid)
            elif scale == "rosetta":
                g.rosetta_uuid = uuid
                g.hmm_enriched = True

            # Metadata from first tile seen
            if not g.platform and tile.get("platform"):
                g.platform = tile["platform"]
            if not g.source_type and tile.get("source_type"):
                g.source_type = tile["source_type"]
            if not g.source_file and tile.get("source_file"):
                g.source_file = tile["source_file"]
            if not g.session_id and tile.get("session_id"):
                g.session_id = tile["session_id"]
            if not g.document_id and tile.get("document_id"):
                g.document_id = tile["document_id"]
            if not g.loaded_at and tile.get("loaded_at"):
                g.loaded_at = tile["loaded_at"]
            if tile.get("hmm_enriched"):
                g.hmm_enriched = True
            if not g.hmm_enriched_at and tile.get("hmm_enriched_at"):
                g.hmm_enriched_at = tile["hmm_enriched_at"]

            g.total_tokens += tile.get("token_count") or 0

        total += len(tiles)
        cursor = tiles[-1]["_additional"]["id"]

        if total % 100000 == 0:
            elapsed = time.monotonic() - t0
            log.info(
                "  Pass 1: scanned %d tiles, %d groups (%.0fs)",
                total, len(groups), elapsed,
            )

        if len(tiles) < 10000:
            break

    elapsed = time.monotonic() - t0
    log.info(
        "Pass 1 complete: %d tiles -> %d unique content_hashes (%.0fs)",
        total, len(groups), elapsed,
    )

    # Ensure all groups have a representative UUID
    for ch, g in groups.items():
        if not g.rep_uuid and g.tile_ids_512:
            g.rep_uuid = g.tile_ids_512[0]
        elif not g.rep_uuid and g.tile_ids_2048:
            g.rep_uuid = g.tile_ids_2048[0]
        elif not g.rep_uuid and g.tile_ids_4096:
            g.rep_uuid = g.tile_ids_4096[0]

    return groups


# ── Pass 2: Build and Write v2 Objects ──────────────────────────

def build_v2_object(
    group: ContentGroup,
    rep_data: dict,
    rosetta_data: Optional[dict] = None,
) -> Optional[dict]:
    """Build a v2 object from group metadata + fetched tile data."""
    rep_props = rep_data.get("properties", {})
    rep_vector = rep_data.get("vector")

    # Use all_tiles_content if provided (full document, no truncation)
    # Falls back to representative tile content if not provided
    content = rep_props.get("content", "")
    if hasattr(group, "_all_content") and group._all_content:
        content = group._all_content

    # Rosetta data
    rosetta_summary = ""
    rosetta_vector = None
    dominant_motifs = []
    motif_data_json = ""
    motif_annotations = ""
    hmm_phi = None
    hmm_trust = None

    if rosetta_data:
        ros_props = rosetta_data.get("properties", {})
        rosetta_summary = ros_props.get("rosetta_summary") or ""
        rosetta_vector = rosetta_data.get("vector")
        dominant_motifs = ros_props.get("dominant_motifs") or []
        motif_data_json = ros_props.get("motif_data_json") or ""
        hmm_phi = ros_props.get("hmm_phi")
        hmm_trust = ros_props.get("hmm_trust")

    # Also check representative tile for HMM data
    if not rosetta_summary and rep_props.get("rosetta_summary"):
        rosetta_summary = rep_props["rosetta_summary"]
    if not dominant_motifs and rep_props.get("dominant_motifs"):
        dominant_motifs = rep_props["dominant_motifs"]
    if not motif_data_json and rep_props.get("motif_data_json"):
        motif_data_json = rep_props["motif_data_json"]
    if hmm_phi is None and rep_props.get("hmm_phi") is not None:
        hmm_phi = rep_props["hmm_phi"]
    if hmm_trust is None and rep_props.get("hmm_trust") is not None:
        hmm_trust = rep_props["hmm_trust"]

    # Build motif annotations as natural language
    if motif_data_json:
        try:
            motif_data = json.loads(motif_data_json)
            parts = []
            if isinstance(motif_data, list):
                for m in motif_data:
                    mid = m.get("motif_id", "")
                    amp = m.get("amplitude", 0)
                    desc = m.get("definition", m.get("description", ""))
                    if mid:
                        parts.append(
                            f"{mid} ({amp:.2f}): {desc}" if desc else f"{mid} ({amp:.2f})"
                        )
            elif isinstance(motif_data, dict):
                for mid, mval in motif_data.items():
                    if isinstance(mval, dict):
                        amp = mval.get("amplitude", 0)
                        desc = mval.get("definition", mval.get("description", ""))
                        parts.append(
                            f"{mid} ({amp:.2f}): {desc}" if desc else f"{mid} ({amp:.2f})"
                        )
            motif_annotations = "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass

    properties = {
        "content": content,
        "content_hash": group.content_hash,
        "platform": group.platform,
        "source_type": group.source_type,
        "source_file": group.source_file,
        "session_id": group.session_id,
        "document_id": group.document_id,
        "loaded_at": group.loaded_at,
        "hmm_enriched": group.hmm_enriched,
        "tile_count_512": len(group.tile_ids_512),
        "tile_count_2048": len(group.tile_ids_2048),
        "tile_count_4096": len(group.tile_ids_4096),
        "total_tokens": group.total_tokens,
        "tile_ids_512": group.tile_ids_512,
        "tile_ids_2048": group.tile_ids_2048,
        "tile_ids_4096": group.tile_ids_4096,
    }

    if rosetta_summary:
        properties["rosetta_summary"] = rosetta_summary
    if motif_annotations:
        properties["motif_annotations"] = motif_annotations
    if dominant_motifs:
        properties["dominant_motifs"] = dominant_motifs
    if motif_data_json:
        properties["motif_data_json"] = motif_data_json
    if hmm_phi is not None:
        properties["hmm_phi"] = hmm_phi
    if hmm_trust is not None:
        properties["hmm_trust"] = hmm_trust
    if group.rosetta_uuid:
        properties["rosetta_tile_id"] = group.rosetta_uuid
    if group.hmm_enriched_at:
        properties["hmm_enriched_at"] = group.hmm_enriched_at

    obj = {"class": V2_CLASS, "properties": properties}

    vectors = {}
    if rep_vector:
        vectors["raw"] = rep_vector
    if rosetta_vector:
        vectors["rosetta"] = rosetta_vector
    if vectors:
        obj["vectors"] = vectors

    return obj


def write_batch(objects: List[dict]) -> int:
    """Write a batch of v2 objects. Returns success count."""
    if not objects:
        return 0
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/batch/objects",
            json={"objects": objects},
            timeout=120,
        )
        if r.status_code == 200:
            results = r.json()
            errors = 0
            for obj in results:
                errs = obj.get("result", {}).get("errors")
                if errs:
                    errors += 1
                    if errors <= 3:
                        log.warning("Write error: %s", errs)
            return len(results) - errors
        else:
            log.error("Batch write failed: %d %s", r.status_code, r.text[:200])
            return 0
    except Exception as e:
        log.error("Batch write exception: %s", e)
        return 0


def pass2_write(
    groups: Dict[str, ContentGroup],
    batch_size: int = 100,
    resume: bool = False,
    dry_run: bool = False,
):
    """Build and write v2 objects from groups."""
    rdb = get_redis()

    # Resume support
    seen_hashes: Set[str] = set()
    migrated = 0
    if resume:
        seen_hashes = set(rdb.smembers("isma:v2_migration:written") or [])
        migrated = len(seen_hashes)
        log.info("Resuming: %d already written", migrated)

    total = len(groups)
    remaining = {ch: g for ch, g in groups.items() if ch not in seen_hashes}
    log.info("Pass 2: Writing %d v2 objects (%d already done)...", len(remaining), migrated)

    batch_objects = []
    batch_hashes = []
    batch_num = 0
    t0 = time.monotonic()

    for i, (ch, group) in enumerate(remaining.items()):
        if not group.rep_uuid:
            continue

        # Fetch representative tile with vector
        rep_data = fetch_object(group.rep_uuid)
        if not rep_data:
            log.debug("Skipping %s: could not fetch rep tile", ch[:16])
            continue

        # Fetch rosetta tile if exists
        rosetta_data = None
        if group.rosetta_uuid:
            rosetta_data = fetch_object(group.rosetta_uuid)

        v2_obj = build_v2_object(group, rep_data, rosetta_data)
        if v2_obj:
            batch_objects.append(v2_obj)
            batch_hashes.append(ch)

        # Write batch
        if len(batch_objects) >= batch_size:
            if dry_run:
                written = len(batch_objects)
                log.info("DRY RUN: would write %d objects", written)
            else:
                written = write_batch(batch_objects)
                if written > 0:
                    rdb.sadd("isma:v2_migration:written", *batch_hashes[:written])
                    seen_hashes.update(batch_hashes[:written])

            migrated += written
            batch_num += 1
            elapsed = time.monotonic() - t0
            rate = migrated / elapsed if elapsed > 0 else 0
            eta = (total - migrated) / rate if rate > 0 else 0

            if batch_num % 5 == 0 or batch_num <= 3:
                log.info(
                    "Batch %d: %d/%d migrated (%.0f/s, ETA %.0fm)",
                    batch_num, migrated, total, rate, eta / 60,
                )

            # Checkpoint
            if not dry_run:
                rdb.hset(CHECKPOINT_KEY, mapping={
                    "migrated": migrated,
                    "total": total,
                    "timestamp": time.time(),
                })

            batch_objects = []
            batch_hashes = []

    # Write remaining
    if batch_objects:
        if dry_run:
            migrated += len(batch_objects)
        else:
            written = write_batch(batch_objects)
            if written > 0:
                rdb.sadd("isma:v2_migration:written", *batch_hashes[:written])
            migrated += written

    elapsed = time.monotonic() - t0
    log.info(
        "Pass 2 complete: %d/%d v2 objects written (%.0fs)",
        migrated, total, elapsed,
    )

    # Final checkpoint
    if not dry_run:
        rdb.hset(CHECKPOINT_KEY, mapping={
            "migrated": migrated,
            "total": total,
            "completed": "true",
            "completed_at": time.time(),
        })

    return migrated


def migrate(batch_size: int = 100, resume: bool = False, dry_run: bool = False):
    """Run the full v1 -> v2 migration."""
    groups = pass1_scan()
    return pass2_write(groups, batch_size=batch_size, resume=resume, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v1 tiles to v2 canonical objects")
    parser.add_argument("--batch-size", type=int, default=100, help="Objects per write batch")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Weaviate")
    args = parser.parse_args()

    count = migrate(
        batch_size=args.batch_size,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    log.info("Total migrated: %d v2 objects", count)
