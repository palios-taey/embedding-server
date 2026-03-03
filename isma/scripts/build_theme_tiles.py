#!/usr/bin/env python3
"""Build 24 theme-level RAPTOR tiles in Weaviate ISMA_Quantum.

Each theme tile is the "root" level of the RAPTOR hierarchy above HMM Rosetta tiles.
Content = theme definition + top-N anchor rosetta summaries.
Embedded via Qwen3 LB and stored with scale='theme'.

These tiles enable:
- Cold-start query routing (match theme before individual tiles)
- Conceptual R@10 improvement via pre-filter on conceptual queries

Usage:
    python3 build_theme_tiles.py [--theme THEME_ID] [--dry-run]
    python3 build_theme_tiles.py              # Build all 24 themes
    python3 build_theme_tiles.py --theme 001  # Rebuild one theme
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

THEME_INDEX_PATH = "/var/spark/isma/theme_search_index.json"
WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_CLASS = "ISMA_Quantum"
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

TOP_N_ROSETTAS = 10  # rosetta summaries to include in theme tile content


def embed_text(text: str):
    """Embed text via Qwen3 LB. Returns list of floats or None on error."""
    try:
        r = requests.post(
            EMBEDDING_URL,
            json={"model": EMBEDDING_MODEL, "input": text, "encoding_format": "float"},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
        log.error(f"Embedding API HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Embedding error: {e}")
    return None


def theme_content_hash(theme_id: str) -> str:
    """Deterministic content_hash for a theme tile."""
    return hashlib.sha256(f"theme_tile_{theme_id}".encode()).hexdigest()[:16]


def build_theme_content(theme: dict) -> str:
    """Build the text content for a theme tile."""
    lines = [
        f"# Theme {theme['theme_id']}: {theme['display_name']}",
        "",
        f"**Description**: {theme['description']}",
        "",
    ]

    required = theme.get("required_motifs", [])
    supporting = theme.get("supporting_motifs", [])
    if required:
        lines.append(f"**Required motifs**: {', '.join(required)}")
    if supporting:
        lines.append(f"**Supporting motifs**: {', '.join(supporting)}")
    lines.append("")

    anchors = theme.get("anchor_rosettas", [])
    if anchors:
        lines.append("## Canonical Examples (Rosetta Summaries)")
        lines.append("")
        for i, anchor in enumerate(anchors[:TOP_N_ROSETTAS]):
            rosetta = anchor.get("rosetta_summary", "").strip()
            src = anchor.get("source_file", "").split("/")[-1]
            if rosetta:
                lines.append(f"{i+1}. [{src}] {rosetta}")
                lines.append("")

    return "\n".join(lines)


def get_existing_tile(theme_id: str):
    """Return existing Weaviate tile UUID for this theme, or None."""
    content_hash = theme_content_hash(theme_id)
    gql = {
        "query": f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{path: ["content_hash"] operator: Equal valueText: "{content_hash}"}}
                    limit: 1
                ) {{ _additional {{ id }} scale }}
            }}
        }}"""
    }
    try:
        r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json=gql, timeout=15)
        results = r.json().get("data", {}).get("Get", {}).get(WEAVIATE_CLASS, [])
        if results:
            return results[0]["_additional"]["id"]
    except Exception as e:
        log.error(f"Weaviate lookup error: {e}")
    return None


def store_theme_tile(theme: dict, content: str, vector: list, dry_run: bool) -> bool:
    """Store or update a theme tile in Weaviate. Returns True on success."""
    theme_id = theme["theme_id"]
    content_hash = theme_content_hash(theme_id)
    now = datetime.now(timezone.utc).isoformat()

    properties = {
        "content": content,
        "content_hash": content_hash,
        "scale": "theme",
        "source_file": f"theme_{theme_id}_{theme['display_name'].lower().replace(' ', '_')}",
        "platform": "corpus",
        "layer": 0,
        "dominant_motifs": (
            theme.get("required_motifs", []) + theme.get("supporting_motifs", [])
        )[:5],
        "rosetta_summary": (
            f"Theme {theme['theme_id']} — {theme['display_name']}: {theme['description']}"
        ),
        "hmm_enriched": True,
        "hmm_enrichment_version": "theme_tile_1.0",
        "hmm_enriched_at": now,
        "loaded_at": now,
        "token_count": len(content.split()),
    }

    if dry_run:
        log.info(f"[DRY RUN] Would store theme tile {theme_id} ({len(content)} chars)")
        return True

    existing_uuid = get_existing_tile(theme_id)

    if existing_uuid:
        # Update existing tile
        r = requests.patch(
            f"{WEAVIATE_URL}/v1/objects/{WEAVIATE_CLASS}/{existing_uuid}",
            json={"properties": properties},
            timeout=15,
        )
        if r.status_code in (200, 204):
            log.info(f"Updated theme tile {theme_id} (uuid={existing_uuid})")
            return True
        log.error(f"Update failed for {theme_id}: {r.status_code} {r.text[:200]}")
        return False
    else:
        # Create new tile with vector
        r = requests.post(
            f"{WEAVIATE_URL}/v1/objects",
            json={
                "class": WEAVIATE_CLASS,
                "properties": properties,
                "vector": vector,
            },
            timeout=15,
        )
        if r.status_code in (200, 201):
            new_uuid = r.json().get("id", "?")
            log.info(f"Created theme tile {theme_id} (uuid={new_uuid})")
            return True
        log.error(f"Create failed for {theme_id}: {r.status_code} {r.text[:200]}")
        return False


def process_theme(theme: dict, dry_run: bool) -> bool:
    """Build, embed, and store a single theme tile. Returns True on success."""
    theme_id = theme["theme_id"]
    name = theme["display_name"]
    log.info(f"Processing theme {theme_id}: {name}")

    content = build_theme_content(theme)
    log.info(f"  Content length: {len(content)} chars, {len(content.split())} words")

    vector = embed_text(content)
    if vector is None:
        log.error(f"  Embedding failed for theme {theme_id}")
        return False
    log.info(f"  Embedded: {len(vector)}-dim vector")

    return store_theme_tile(theme, content, vector, dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", help="Build only this theme ID (e.g. 001)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute content and embed but don't write to Weaviate")
    args = parser.parse_args()

    with open(THEME_INDEX_PATH) as f:
        theme_index = json.load(f)

    themes = list(theme_index.values())
    if args.theme:
        themes = [t for t in themes if t["theme_id"] == args.theme]
        if not themes:
            log.error(f"Theme {args.theme!r} not found in index")
            sys.exit(1)

    log.info(f"Building {len(themes)} theme tile(s)...")
    t0 = time.monotonic()
    success, failure = 0, 0

    for theme in themes:
        ok = process_theme(theme, args.dry_run)
        if ok:
            success += 1
        else:
            failure += 1

    elapsed = time.monotonic() - t0
    log.info(
        f"Done in {elapsed:.1f}s — "
        f"{'(dry run) ' if args.dry_run else ''}"
        f"{success} OK, {failure} failed"
    )
    sys.exit(0 if failure == 0 else 1)


if __name__ == "__main__":
    main()
