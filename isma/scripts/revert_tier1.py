#!/usr/bin/env python3
"""Revert all Tier 1 batch classification enrichment from Weaviate tiles.

Finds all tiles with hmm_enrichment_version == "tier1_1.0.0" and clears
the HMM properties back to defaults.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("revert_tier1")

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
WEAVIATE_CLASS = "ISMA_Quantum"
PAGE_SIZE = 100

# Properties to clear
CLEAR_PATCH = {
    "dominant_motifs": [],
    "motif_data_json": "",
    "hmm_enriched": False,
    "hmm_enrichment_version": "",
    "hmm_enriched_at": "",
    "hmm_consensus": False,
    "hmm_gate_flags": [],
}

session = requests.Session()


def count_tier1_tiles() -> int:
    """Count tiles with tier1 enrichment."""
    gql = """{
        Aggregate {
            ISMA_Quantum(where: {
                path: ["hmm_enrichment_version"],
                operator: Equal,
                valueText: "tier1_1.0.0"
            }) {
                meta { count }
            }
        }
    }"""
    r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql})
    r.raise_for_status()
    data = r.json()
    return data["data"]["Aggregate"]["ISMA_Quantum"][0]["meta"]["count"]


def fetch_tier1_page(cursor: str = None) -> tuple:
    """Fetch a page of tier1-enriched tiles."""
    after_clause = f', after: "{cursor}"' if cursor else ""
    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                limit: {PAGE_SIZE}
                where: {{
                    path: ["hmm_enrichment_version"],
                    operator: Equal,
                    valueText: "tier1_1.0.0"
                }}
                {after_clause}
            ) {{
                _additional {{ id }}
            }}
        }}
    }}"""
    r = session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql})
    r.raise_for_status()
    data = r.json()
    objects = data.get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
    last_id = objects[-1]["_additional"]["id"] if objects else None
    return objects, last_id


def clear_tile(uuid: str) -> bool:
    """Clear HMM properties on a single tile."""
    url = f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{uuid}"
    try:
        r = session.patch(url, json={"properties": CLEAR_PATCH})
        return r.status_code == 204
    except Exception as e:
        log.warning(f"Error clearing {uuid}: {e}")
        return False


def main():
    total = count_tier1_tiles()
    log.info(f"Found {total:,} tiles with tier1_1.0.0 enrichment to revert")

    if total == 0:
        log.info("Nothing to revert")
        return

    reverted = 0
    failed = 0
    cursor = None
    page = 0
    t0 = time.time()

    while True:
        objects, next_cursor = fetch_tier1_page(cursor)
        if not objects:
            break

        for obj in objects:
            uuid = obj["_additional"]["id"]
            if clear_tile(uuid):
                reverted += 1
            else:
                failed += 1

        page += 1
        elapsed = time.time() - t0
        rate = reverted / elapsed if elapsed > 0 else 0

        if page % 10 == 0:
            log.info(
                f"Page {page}: {reverted:,} reverted, {failed} failed, "
                f"{rate:.0f} tiles/s"
            )

        # Don't use cursor pagination with where - fetch fresh each time
        # since we're removing the filter condition as we go
        cursor = None

    elapsed = time.time() - t0
    log.info(
        f"DONE: {reverted:,} reverted, {failed} failed in {elapsed:.1f}s "
        f"({reverted/elapsed:.0f} tiles/s)"
    )

    # Verify
    remaining = count_tier1_tiles()
    log.info(f"Remaining tier1 tiles: {remaining:,}")


if __name__ == "__main__":
    main()
