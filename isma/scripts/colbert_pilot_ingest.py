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
BATCH_SIZE = 32        # Inference batch size (GPU)

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
# STREAMING INGEST — cursor → encode → write with no accumulation
# =============================================================================

def _tile_from_obj(obj: dict) -> dict | None:
    """Extract tile dict from Weaviate REST object. Returns None if text too short."""
    props = obj.get("properties", {})
    text = (props.get("rosetta_summary") or "").strip()
    if len(text) < 50:
        text = (text + " " + (props.get("content") or "")).strip()[:2000]
    if len(text) < 20:
        return None
    return {
        "uuid": obj["id"],
        "content_hash": props.get("content_hash", ""),
        "text": text,
        "source_file": props.get("source_file", ""),
        "platform": props.get("platform", ""),
        "dominant_motifs": props.get("dominant_motifs") or [],
        "rosetta_summary": props.get("rosetta_summary", ""),
        "scale": props.get("scale", ""),
    }


def run_streaming_ingest(model, tokenizer, scale: str, limit: int,
                         existing_uuids: set, cursor_start: str | None = None,
                         shard_id: int = 0, num_shards: int = 1):
    """Cursor-based streaming ingest: fetch → encode → write with no accumulation.

    Uses REST /v1/objects cursor (O(1) per page regardless of depth).
    Filters by scale in Python. Encodes BATCH_SIZE tiles, writes immediately.

    shard_id / num_shards: hash-based sharding so multiple nodes can run
    simultaneously on the same scale without coordination or duplicates.
    Each node writes only tiles where int(uuid_hex, 16) % num_shards == shard_id.
    """
    log.info(f"Streaming ingest: scale={scale} limit={limit} "
             f"shard={shard_id}/{num_shards} cursor_start={cursor_start or 'beginning'}")
    cursor = cursor_start
    t0 = time.monotonic()
    success = failed = found = scanned = 0
    encode_buf: list = []  # fixed max BATCH_SIZE, flushed immediately — not accumulation

    while found < limit:
        url = f"{WEAVIATE_URL}/v1/objects?class={SOURCE_CLASS}&limit=500"
        if cursor:
            url += f"&after={cursor}"
        try:
            r = requests.get(url, timeout=30)
            objects = r.json().get("objects", [])
        except Exception as e:
            log.error(f"Fetch error at cursor {cursor}: {e}")
            break

        if not objects:
            break

        cursor = objects[-1]["id"]
        scanned += len(objects)

        for obj in objects:
            if obj.get("properties", {}).get("scale") != scale:
                continue
            if num_shards > 1 and int(obj["id"].replace("-", ""), 16) % num_shards != shard_id:
                continue
            if obj["id"] in existing_uuids:
                continue
            tile = _tile_from_obj(obj)
            if tile is None:
                continue

            encode_buf.append(tile)
            found += 1

            if len(encode_buf) >= BATCH_SIZE:
                try:
                    vectors = encode_passages([t["text"] for t in encode_buf], model, tokenizer)
                except Exception as e:
                    log.error(f"Encode error: {e}")
                    failed += len(encode_buf)
                    encode_buf.clear()
                    continue
                ok, bad = ingest_batch(encode_buf, vectors)
                success += ok
                failed += bad
                encode_buf.clear()

                elapsed = time.monotonic() - t0
                rate = found / elapsed if elapsed > 0 else 0
                log.info(f"Progress: {found} | {rate:.1f} tiles/s | "
                         f"scanned={scanned} ok={success} failed={failed}")

        if len(objects) < 500:
            log.info("Cursor exhausted — all objects scanned")
            break

    # Flush tail (at most BATCH_SIZE-1 tiles — not accumulation, just remainder)
    if encode_buf:
        try:
            vectors = encode_passages([t["text"] for t in encode_buf], model, tokenizer)
            ok, bad = ingest_batch(encode_buf, vectors)
            success += ok
            failed += bad
        except Exception as e:
            log.error(f"Final encode error: {e}")

    elapsed = time.monotonic() - t0
    log.info(f"Done in {elapsed:.1f}s — {success} ingested, {failed} failed | "
             f"scanned {scanned} objects to find {found} {scale} tiles")


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


def get_existing_source_uuids() -> set:
    """Return set of source_uuids already ingested (dedup by source tile UUID, not content_hash).

    This allows multiple entries per content_hash (different passages from same document)
    while preventing re-ingestion of the same source tile on resume.
    """
    all_uuids = set()
    offset = 0
    page_size = 2000
    while True:
        gql = f"""{{
            Get {{
                {PILOT_CLASS}(limit: {page_size} offset: {offset}) {{ source_uuid }}
            }}
        }}"""
        try:
            r = requests.post(
                f"{WEAVIATE_URL}/v1/graphql",
                json={"query": gql},
                timeout=30,
            )
            results = r.json().get("data", {}).get("Get", {}).get(PILOT_CLASS, []) or []
        except Exception as e:
            log.warning(f"Could not fetch existing UUIDs at offset {offset}: {e}")
            break
        if not results:
            break
        for t in results:
            if t.get("source_uuid"):
                all_uuids.add(t["source_uuid"])
        if len(results) < page_size:
            break
        offset += len(results)
    log.info(f"Found {len(all_uuids)} already-ingested source tiles")
    return all_uuids


def ingest_batch(tiles: list, vectors_list: List[List[List[float]]]) -> tuple[int, int]:
    """Batch-write tiles to Weaviate using /v1/batch/objects. Returns (success, failed)."""
    objects = []
    for tile, vectors in zip(tiles, vectors_list):
        if len(vectors) == 0:
            continue
        objects.append({
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
            "vectors": {"colbert": vectors},
        })
    if not objects:
        return 0, len(tiles)
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/batch/objects",
            json={"objects": objects},
            timeout=60,
        )
        if r.status_code not in (200, 201):
            log.error(f"Batch write failed: {r.status_code} {r.text[:200]}")
            return 0, len(objects)
        results = r.json()
        if isinstance(results, list):
            ok = sum(1 for res in results if res.get("result", {}).get("status") == "SUCCESS")
            failed = len(results) - ok
        else:
            ok = len(objects)
            failed = 0
        return ok, failed
    except Exception as e:
        log.error(f"Batch write error: {e}")
        return 0, len(objects)


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
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index (0-based). With --num-shards, each node writes a distinct subset.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards. Splits load across nodes without coordination.")
    parser.add_argument("--scale", default="search_512",
                        choices=["rosetta", "search_512", "context_2048", "full_4096"],
                        help="Source scale to ingest (default: search_512 for full passage coverage)")
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

    # UUID dedup (fast for empty class, needed for safe resume)
    existing_uuids = get_existing_source_uuids()

    # Stream: cursor → encode → write, no accumulation
    run_streaming_ingest(model, tokenizer, scale=args.scale,
                         limit=args.limit, existing_uuids=existing_uuids,
                         shard_id=args.shard_id, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
