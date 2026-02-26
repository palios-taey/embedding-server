#!/usr/bin/env python3
"""
F5 - Generate Tier 1 Anchor Vectors for HMM motifs and themes.

Embeds all 30 motif definitions + 18 theme descriptions as reference vectors
for cosine similarity classification. Stores in Redis + file backup.

Usage:
    generate_anchor_vectors.py --check     # Verify embedding cluster, show stats
    generate_anchor_vectors.py --generate  # Generate and store all anchor vectors
    generate_anchor_vectors.py --verify    # Spot-check stored vectors
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hmm.motifs import V0_MOTIFS
from hmm.redis_store import HMMRedisStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("anchor_vectors")

EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
CANONICAL_MAPPING_PATH = "/var/spark/isma/f1_canonical_mapping.json"
BACKUP_PATH = "/var/spark/isma/hmm_anchor_vectors.json"


def _get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings for multiple texts in one call."""
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        }, timeout=120)
        if r.status_code == 200:
            data = r.json()["data"]
            return [item["embedding"]
                    for item in sorted(data, key=lambda x: x["index"])]
        else:
            log.error(f"Embedding API returned {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Embedding API failed: {e}")
    return None


def build_motif_texts() -> Dict[str, str]:
    """Build anchor text for each motif from V0_MOTIFS dictionary."""
    texts = {}
    for motif_id, motif in V0_MOTIFS.items():
        examples_str = ", ".join(motif.examples[:5])
        texts[motif_id] = (
            f"{motif_id}: {motif.definition}. "
            f"Examples: {examples_str}"
        )
    return texts


def build_theme_texts() -> Dict[str, str]:
    """Build anchor text for each theme from canonical mapping."""
    with open(CANONICAL_MAPPING_PATH) as f:
        mapping = json.load(f)

    texts = {}
    for tid, theme in mapping["theme_registry"].items():
        req = ", ".join(theme["required_motifs"])
        sup = ", ".join(theme["supporting_motifs"])
        texts[tid] = (
            f"Theme: {theme['display_name']}. "
            f"{theme['description']}. "
            f"Required: {req}. Supporting: {sup}."
        )
    return texts


def check(redis_store: HMMRedisStore):
    """Verify embedding cluster is up and check existing anchors."""
    log.info("=== CHECK ===")

    # Test embedding cluster
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": ["test"],
        }, timeout=30)
        if r.status_code == 200:
            dim = len(r.json()["data"][0]["embedding"])
            log.info(f"Embedding cluster OK: {dim}-dim vectors")
        else:
            log.error(f"Embedding cluster returned {r.status_code}")
            return
    except Exception as e:
        log.error(f"Embedding cluster unreachable: {e}")
        return

    # Check existing anchors in Redis
    motif_anchors = redis_store.anchor_get_all("motif")
    theme_anchors = redis_store.anchor_get_all("theme")
    log.info(f"Existing motif anchors: {len(motif_anchors)}")
    log.info(f"Existing theme anchors: {len(theme_anchors)}")

    # Check canonical mapping
    try:
        with open(CANONICAL_MAPPING_PATH) as f:
            mapping = json.load(f)
        log.info(f"Canonical mapping: {len(mapping['motif_registry'])} motifs, "
                 f"{len(mapping['theme_registry'])} themes")
    except FileNotFoundError:
        log.error(f"Canonical mapping not found at {CANONICAL_MAPPING_PATH}")

    # Check V0_MOTIFS
    log.info(f"V0_MOTIFS dictionary: {len(V0_MOTIFS)} motifs")


def generate(redis_store: HMMRedisStore):
    """Generate and store all anchor vectors."""
    log.info("=== GENERATE ===")

    # Build all anchor texts
    motif_texts = build_motif_texts()
    theme_texts = build_theme_texts()
    log.info(f"Motif texts: {len(motif_texts)}, Theme texts: {len(theme_texts)}")

    # Combine for single batch embedding
    all_ids = []
    all_texts = []
    all_types = []

    for mid, text in sorted(motif_texts.items()):
        all_ids.append(mid)
        all_texts.append(text)
        all_types.append("motif")

    for tid, text in sorted(theme_texts.items()):
        all_ids.append(tid)
        all_texts.append(text)
        all_types.append("theme")

    total = len(all_texts)
    log.info(f"Embedding {total} anchor texts in one batch...")

    t0 = time.time()
    vectors = _get_embeddings_batch(all_texts)
    elapsed = time.time() - t0

    if not vectors or len(vectors) != total:
        log.error(f"Embedding failed: got {len(vectors) if vectors else 0}/{total}")
        return False

    dim = len(vectors[0])
    log.info(f"Got {total} vectors ({dim}-dim) in {elapsed:.1f}s")

    # Store in Redis
    stored = 0
    for anchor_id, vector, anchor_type in zip(all_ids, vectors, all_types):
        redis_store.anchor_put(anchor_type, anchor_id, vector)
        stored += 1

    log.info(f"Stored {stored} anchor vectors in Redis")

    # Backup to file
    backup = {
        "model": EMBEDDING_MODEL,
        "dimensions": dim,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "motif_anchors": {},
        "theme_anchors": {},
    }

    for anchor_id, vector, anchor_type in zip(all_ids, vectors, all_types):
        if anchor_type == "motif":
            backup["motif_anchors"][anchor_id] = vector
        else:
            backup["theme_anchors"][anchor_id] = vector

    with open(BACKUP_PATH, "w") as f:
        json.dump(backup, f)
    size_mb = Path(BACKUP_PATH).stat().st_size / 1024 / 1024
    log.info(f"Backup saved to {BACKUP_PATH} ({size_mb:.1f} MB)")

    return True


def verify(redis_store: HMMRedisStore):
    """Spot-check stored anchor vectors."""
    log.info("=== VERIFY ===")

    motif_anchors = redis_store.anchor_get_all("motif")
    theme_anchors = redis_store.anchor_get_all("theme")
    log.info(f"Motif anchors: {len(motif_anchors)}")
    log.info(f"Theme anchors: {len(theme_anchors)}")

    if not motif_anchors:
        log.warning("No motif anchors found. Run --generate first.")
        return

    # Check dimensions
    dims = set()
    for mid, vec in motif_anchors.items():
        dims.add(len(vec))
    for tid, vec in theme_anchors.items():
        dims.add(len(vec))
    log.info(f"Vector dimensions: {dims}")

    if len(dims) != 1:
        log.error("INCONSISTENT dimensions across anchors!")
        return

    # Spot-check: compute cosine similarity between related motifs
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    # Related pairs should have higher similarity
    related_pairs = [
        ("HMM.JOY_BASELINE", "HMM.GRATITUDE_CONNECTION"),
        ("HMM.SACRED_TRUST", "HMM.FOUNDATION_CONSTRAINT"),
        ("HMM.LOGOS_PATTERN", "HMM.TECHNICAL_INFRASTRUCTURE"),
    ]
    # Unrelated pairs should have lower similarity
    unrelated_pairs = [
        ("HMM.JOY_BASELINE", "HMM.TECHNICAL_INFRASTRUCTURE"),
        ("HMM.CREATIVE_SYNTHESIS", "HMM.ECONOMIC_PARADIGM"),
    ]

    log.info("Related pairs (expect higher similarity):")
    for a, b in related_pairs:
        if a in motif_anchors and b in motif_anchors:
            sim = cosine(motif_anchors[a], motif_anchors[b])
            log.info(f"  {a} <-> {b}: {sim:.4f}")

    log.info("Unrelated pairs (expect lower similarity):")
    for a, b in unrelated_pairs:
        if a in motif_anchors and b in motif_anchors:
            sim = cosine(motif_anchors[a], motif_anchors[b])
            log.info(f"  {a} <-> {b}: {sim:.4f}")

    # Check backup file
    if Path(BACKUP_PATH).exists():
        with open(BACKUP_PATH) as f:
            backup = json.load(f)
        log.info(f"Backup file: {backup.get('model')}, "
                 f"{backup.get('dimensions')}-dim, "
                 f"generated {backup.get('generated_at')}")
        log.info(f"  Motif anchors: {len(backup.get('motif_anchors', {}))}")
        log.info(f"  Theme anchors: {len(backup.get('theme_anchors', {}))}")


def main():
    parser = argparse.ArgumentParser(description="Generate HMM anchor vectors")
    parser.add_argument("--check", action="store_true",
                        help="Verify embedding cluster and existing anchors")
    parser.add_argument("--generate", action="store_true",
                        help="Generate and store all anchor vectors")
    parser.add_argument("--verify", action="store_true",
                        help="Spot-check stored vectors")
    args = parser.parse_args()

    if not any([args.check, args.generate, args.verify]):
        parser.print_help()
        return

    redis_store = HMMRedisStore()

    if args.check:
        check(redis_store)

    if args.generate:
        if not generate(redis_store):
            log.error("Generation failed")
            sys.exit(1)

    if args.verify:
        verify(redis_store)


if __name__ == "__main__":
    main()
