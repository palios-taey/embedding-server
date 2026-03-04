#!/usr/bin/env python3
"""6B-3: ColBERT Pilot — Create ISMA_ColBERT_Pilot class and ingest top 20K enriched tiles.

Uses jinaai/jina-colbert-v2 (560M, 64-dim Matryoshka late interaction).
Stores multi-vector per-token embeddings in Weaviate with MaxSim aggregation.

Usage:
    python3 colbert_pilot_ingest.py [--create-class] [--ingest] [--limit 20000]
    python3 colbert_pilot_ingest.py --create-class         # Create class only
    python3 colbert_pilot_ingest.py --ingest --limit 5000  # Ingest N tiles
    python3 colbert_pilot_ingest.py --create-class --ingest # Full pilot setup
    python3 colbert_pilot_ingest.py --stats                # Show ingest status
"""

import argparse
import json
import logging
import sys
import time
from typing import List, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

WEAVIATE_URL = "http://192.168.100.10:8088"
PILOT_CLASS = "ISMA_ColBERT_Pilot"
SOURCE_CLASS = "ISMA_Quantum"

COLBERT_MODEL = "jinaai/jina-colbert-v2"
COLBERT_DIM = 64       # Matryoshka: use first 64 dims for storage efficiency
MAX_DOC_TOKENS = 512   # Max tokens per passage (ColBERT recommendation)
MAX_QUERY_TOKENS = 64  # Max tokens per query
BATCH_SIZE = 8         # Inference batch size (CPU)

# Redis for checkpoint
CHECKPOINT_KEY = "isma:colbert_pilot:checkpoint"


# =============================================================================
# COLBERT MODEL
# =============================================================================

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    log.info(f"Loading {COLBERT_MODEL}...")
    _tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL)
    _model = AutoModel.from_pretrained(COLBERT_MODEL, trust_remote_code=True)
    _model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    log.info(f"Model loaded on {device}")
    return _model, _tokenizer


def encode_passages(texts: List[str], model=None, tokenizer=None) -> List[List[List[float]]]:
    """Encode a batch of passages. Returns list of multi-vectors (one per text).
    Each multi-vector is a list of 64-dim token vectors (non-padding tokens only).
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    device = next(model.parameters()).device

    # Tokenize with passage prefix (ColBERT convention: [unused1] for passages)
    prefixed = [f"[unused1]{t}" for t in texts]
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        max_length=MAX_DOC_TOKENS,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Per-token embeddings [batch, seq_len, hidden_dim]
    token_embs = outputs.last_hidden_state

    # Slice to COLBERT_DIM (Matryoshka)
    token_embs = token_embs[:, :, :COLBERT_DIM]

    # L2 normalize per token
    token_embs = F.normalize(token_embs, p=2, dim=-1)

    # Return non-padding token vectors per passage
    attention_mask = inputs["attention_mask"]  # [batch, seq_len]
    result = []
    for i in range(len(texts)):
        mask = attention_mask[i].bool()
        vecs = token_embs[i][mask]  # [n_tokens, COLBERT_DIM]
        result.append(vecs.cpu().tolist())
    return result


def encode_query(text: str, model=None, tokenizer=None) -> List[List[float]]:
    """Encode a single query. Returns multi-vector."""
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    device = next(model.parameters()).device

    # Query prefix (ColBERT: [unused0] for queries)
    prefixed = f"[unused0]{text}"
    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        max_length=MAX_QUERY_TOKENS,
        truncation=True,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    token_embs = outputs.last_hidden_state[:, :, :COLBERT_DIM]  # [1, seq_len, dim]
    token_embs = F.normalize(token_embs, p=2, dim=-1)

    mask = inputs["attention_mask"][0].bool()
    return token_embs[0][mask].cpu().tolist()


# =============================================================================
# WEAVIATE CLASS
# =============================================================================

def class_exists() -> bool:
    r = requests.get(f"{WEAVIATE_URL}/v1/schema/{PILOT_CLASS}", timeout=10)
    return r.status_code == 200


def create_class():
    """Create ISMA_ColBERT_Pilot with multi-vector ColBERT config."""
    if class_exists():
        log.info(f"{PILOT_CLASS} already exists — skipping create")
        return

    schema = {
        "class": PILOT_CLASS,
        "description": (
            "ColBERT pilot: 20K enriched tiles with per-token 64-dim vectors "
            "for MaxSim late interaction retrieval. Part of ISMA Phase 6B-3."
        ),
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
            "cleanupIntervalSeconds": 60,
            "stopwords": {"preset": "en"},
            "usingBlockMaxWAND": True,
            "indexTimestamps": True,
        },
        "properties": [
            {"name": "content_hash", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
            {"name": "content", "dataType": ["text"],
             "tokenization": "word", "indexSearchable": True, "indexFilterable": False},
            {"name": "rosetta_summary", "dataType": ["text"],
             "tokenization": "word", "indexSearchable": True, "indexFilterable": False},
            {"name": "source_file", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
            {"name": "platform", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
            {"name": "dominant_motifs", "dataType": ["text[]"],
             "indexFilterable": True, "indexSearchable": True},
            {"name": "n_tokens", "dataType": ["int"],
             "indexFilterable": True, "indexRangeFilters": True},
            {"name": "source_uuid", "dataType": ["text"],
             "tokenization": "field", "indexFilterable": True, "indexSearchable": False},
        ],
        "vectorConfig": {
            "colbert": {
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "dot",
                    "efConstruction": 128,
                    "maxConnections": 32,
                    "ef": -1,
                    "multivector": {
                        "enabled": True,
                        "aggregation": "maxSim",
                    },
                },
                "vectorizer": {"none": {}},
            }
        },
    }

    r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, timeout=30)
    if r.status_code in (200, 201):
        log.info(f"Created class {PILOT_CLASS}")
    else:
        log.error(f"Failed to create class: {r.status_code} {r.text[:300]}")
        sys.exit(1)


# =============================================================================
# TILE FETCHING
# =============================================================================

def fetch_source_tiles(limit: int = 20000) -> list:
    """Fetch top N enriched tiles from ISMA_Quantum, prioritizing information density.

    Selection strategy: hmm_enriched=True, sorted by Rosetta quality.
    Targets diverse platforms and scale levels.

    NOTE: Weaviate cursor API (after + where) is incompatible — uses offset pagination.
    """
    log.info(f"Fetching {limit} tiles from {SOURCE_CLASS}...")

    # Use offset-based pagination (cursor pagination incompatible with where filter)
    all_tiles = []
    batch_size = 500
    offset = 0

    while len(all_tiles) < limit:
        fetch_limit = min(batch_size, limit - len(all_tiles) + 50)

        # Build GraphQL query with offset pagination
        gql = f"""{{
            Get {{
                {SOURCE_CLASS}(
                    where: {{
                        operator: And
                        operands: [
                            {{path: ["hmm_enriched"], operator: Equal, valueBoolean: true}}
                            {{path: ["scale"], operator: NotEqual, valueText: "theme"}}
                        ]
                    }}
                    limit: {fetch_limit}
                    offset: {offset}
                ) {{
                    content_hash
                    content
                    rosetta_summary
                    source_file
                    platform
                    dominant_motifs
                    scale
                    _additional {{ id }}
                }}
            }}
        }}"""

        try:
            r = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                json={"query": gql},
                timeout=30,
            )
            results = (
                r.json().get("data", {}).get("Get", {}).get(SOURCE_CLASS, []) or []
            )
        except Exception as e:
            log.error(f"Fetch error at offset {offset}: {e}")
            break

        if not results:
            break

        for item in results:
            # Build text from rosetta_summary + content (prefer rosetta for ColBERT)
            text = (item.get("rosetta_summary") or "").strip()
            if len(text) < 50:
                content = (item.get("content") or "").strip()
                text = (text + " " + content).strip()[:2000]

            if len(text) < 20:
                continue

            all_tiles.append({
                "uuid": item["_additional"]["id"],
                "content_hash": item.get("content_hash", ""),
                "text": text,
                "source_file": item.get("source_file", ""),
                "platform": item.get("platform", ""),
                "dominant_motifs": item.get("dominant_motifs") or [],
                "rosetta_summary": item.get("rosetta_summary", ""),
                "scale": item.get("scale", ""),
            })

        offset += len(results)
        log.info(f"  Fetched {len(all_tiles)}/{limit} tiles (offset={offset})...")

        if len(results) < fetch_limit:
            break

    log.info(f"Fetched {len(all_tiles)} source tiles")
    return all_tiles[:limit]


# =============================================================================
# CHECKPOINT / DEDUP
# =============================================================================

def get_redis():
    try:
        import redis
        r = redis.Redis(host="192.168.100.10", port=6379, decode_responses=True)
        r.ping()
        return r
    except Exception:
        log.warning("Redis unavailable — checkpointing disabled")
        return None


def get_existing_hashes(sample_size: int = 500) -> set:
    """Return set of content_hashes already ingested into ISMA_ColBERT_Pilot."""
    gql = f"""{{
        Get {{
            {PILOT_CLASS}(
                limit: {sample_size}
            ) {{ content_hash }}
        }}
    }}"""
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": gql},
            timeout=20,
        )
        results = r.json().get("data", {}).get("Get", {}).get(PILOT_CLASS, []) or []
        return {t["content_hash"] for t in results if t.get("content_hash")}
    except Exception as e:
        log.warning(f"Could not fetch existing hashes: {e}")
        return set()


def ingest_tile(tile: dict, vectors: List[List[float]]) -> bool:
    """Store a single tile with multi-vector. Returns True on success."""
    obj = {
        "class": PILOT_CLASS,
        "properties": {
            "content_hash": tile["content_hash"],
            "content": (tile.get("rosetta_summary") or "")[:2000],
            "rosetta_summary": (tile.get("rosetta_summary") or ""),
            "source_file": tile["source_file"],
            "platform": tile["platform"],
            "dominant_motifs": tile["dominant_motifs"][:10],
            "n_tokens": len(vectors),
            "source_uuid": tile["uuid"],
        },
        "vectors": {
            "colbert": vectors,
        },
    }
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/objects",
            json=obj,
            timeout=15,
        )
        return r.status_code in (200, 201)
    except Exception as e:
        log.error(f"Ingest error for {tile['content_hash'][:8]}: {e}")
        return False


# =============================================================================
# STATS
# =============================================================================

def show_stats():
    r = requests.post(
        f"{WEAVIATE_URL}/v1/graphql",
        json={"query": f"{{ Aggregate {{ {PILOT_CLASS} {{ meta {{ count }} }} }} }}"},
        timeout=15,
    )
    count = (
        r.json().get("data", {}).get("Aggregate", {})
        .get(PILOT_CLASS, [{}])[0]
        .get("meta", {}).get("count", "?")
    )
    print(f"{PILOT_CLASS}: {count} tiles ingested")
    print(f"Class exists: {class_exists()}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-class", action="store_true",
                        help="Create ISMA_ColBERT_Pilot Weaviate class")
    parser.add_argument("--ingest", action="store_true",
                        help="Ingest tiles into the class")
    parser.add_argument("--limit", type=int, default=20000,
                        help="Number of tiles to ingest (default: 20000)")
    parser.add_argument("--stats", action="store_true",
                        help="Show current ingest stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if not args.create_class and not args.ingest:
        parser.print_help()
        sys.exit(1)

    if args.create_class:
        create_class()

    if not args.ingest:
        return

    # Load ColBERT model
    model, tokenizer = load_model()

    # Fetch source tiles
    tiles = fetch_source_tiles(args.limit)
    if not tiles:
        log.error("No tiles fetched")
        sys.exit(1)

    # Skip already-ingested tiles
    existing = get_existing_hashes(sample_size=min(1000, args.limit))
    if existing:
        log.info(f"Skipping {len(existing)} already-ingested tiles")
        tiles = [t for t in tiles if t["content_hash"] not in existing]
        log.info(f"Remaining to ingest: {len(tiles)}")

    if not tiles:
        log.info("All tiles already ingested")
        return

    # Batch encode + ingest
    t0 = time.monotonic()
    success = failed = 0

    for i in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[i : i + BATCH_SIZE]
        texts = [t["text"] for t in batch]

        try:
            vectors_batch = encode_passages(texts, model, tokenizer)
        except Exception as e:
            log.error(f"Encoding error at batch {i}: {e}")
            failed += len(batch)
            continue

        for tile, vectors in zip(batch, vectors_batch):
            if len(vectors) == 0:
                log.warning(f"Zero tokens for {tile['content_hash'][:8]}, skipping")
                failed += 1
                continue
            ok = ingest_tile(tile, vectors)
            if ok:
                success += 1
            else:
                failed += 1

        elapsed = time.monotonic() - t0
        done = i + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(tiles) - done) / rate if rate > 0 else 0
        log.info(
            f"Progress: {done}/{len(tiles)} | "
            f"{rate:.1f} tiles/s | ETA {eta:.0f}s | "
            f"ok={success} failed={failed}"
        )

    elapsed = time.monotonic() - t0
    log.info(f"Done in {elapsed:.1f}s — {success} ingested, {failed} failed")


if __name__ == "__main__":
    main()
