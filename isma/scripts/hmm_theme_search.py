#!/usr/bin/env python3
"""
HMM Theme Search Index Builder

For each of the 18 canonical themes, searches Weaviate for related unenriched
content using nearVector (embedding the theme/motif descriptions via vLLM)
and tier1 motif_data_json similarity scores. Builds an index that the package
builder uses to create theme-coherent packages.

Usage:
    python3 hmm_theme_search.py --build          # Build full index
    python3 hmm_theme_search.py --stats           # Show index stats
    python3 hmm_theme_search.py --theme 001       # Rebuild one theme

Output: /var/spark/isma/theme_search_index.json
"""

import sys
import os
import json
import time
import logging
import argparse
import requests
from typing import Dict, List, Set, Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
WEAVIATE_CLASS = "ISMA_Quantum"
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

CANONICAL_MAPPING = "/var/spark/isma/f1_canonical_mapping.json"
INDEX_PATH = "/var/spark/isma/theme_search_index.json"

# Search parameters
NEAR_VECTOR_LIMIT = 10000          # Max tiles per nearVector query
NEAR_VECTOR_CERTAINTY = 0.55       # Min certainty for nearVector results
MOTIF_SIMILARITY_THRESHOLD = 0.35  # Min tier1 similarity score for motif matches
OFFSET_PAGE_SIZE = 500             # Page size for offset pagination
MAX_OFFSET = 40000                 # Weaviate offset limit safety margin

# Layers that are already compressed (kernel-layer_2) — serve as anchors
ENRICHED_LAYERS = {"kernel", "layer_0", "v0", "chewy", "chewy-gallery", "layer_1", "layer_2"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("theme_search")

# ============================================================================
# Embedding Helper
# ============================================================================

_embed_cache: Dict[str, List[float]] = {}


def embed_text(text: str) -> Optional[List[float]]:
    """Embed text via vLLM API. Caches results."""
    if text in _embed_cache:
        return _embed_cache[text]

    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": text,
            "encoding_format": "float",
        }, timeout=30)
        if r.status_code == 200:
            vec = r.json()["data"][0]["embedding"]
            _embed_cache[text] = vec
            return vec
        log.error(f"Embedding API HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Embedding error: {e}")
    return None


# ============================================================================
# Weaviate Helpers
# ============================================================================

_session = requests.Session()


def weaviate_gql(query: str, timeout: int = 120) -> Optional[dict]:
    """Execute Weaviate GraphQL query."""
    try:
        r = _session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data.get("errors"):
                log.error(f"GraphQL errors: {json.dumps(data['errors'])[:300]}")
                return None
            return data.get("data", {})
        log.error(f"Weaviate HTTP {r.status_code}")
    except Exception as e:
        log.error(f"Weaviate error: {e}")
    return None


def _source_layer(source_file: str) -> str:
    """Extract the layer/stage from a source file path."""
    s = source_file.lower()
    if "/kernel/" in s: return "kernel"
    if "/layer_0/" in s or "/v0" in s: return "layer_0"
    if "/chewy-consciousness" in s: return "chewy-gallery"
    if "/chewy/" in s: return "chewy"
    if "/layer_1/" in s: return "layer_1"
    if "/layer_2/" in s: return "layer_2"
    if "/github-repos/" in s: return "github-repos"
    if "/expansion_md/" in s: return "expansion_md"
    if "/mira_md" in s: return "mira_md"
    if "/mac_all_md/" in s: return "mac_all_md"
    if "/spark_loose/" in s: return "spark_loose"
    if "/parsed/" in s or "transcript" in s: return "transcript"
    return "unknown"


def _is_already_compressed(source_file: str) -> bool:
    """Check if this item is from kernel-layer_2 (already compressed)."""
    return _source_layer(source_file) in ENRICHED_LAYERS


# ============================================================================
# Theme Index Builder
# ============================================================================

def load_canonical_mapping() -> dict:
    """Load the 18 themes and their motif mappings."""
    with open(CANONICAL_MAPPING) as f:
        return json.load(f)


def find_items_for_theme(theme_id: str, theme_info: dict, motif_registry: dict) -> dict:
    """Find all unenriched items related to a theme.

    Two search strategies:
    1. nearVector: Embed theme description, search Weaviate for similar tiles
    2. nearVector per motif: Embed each motif definition, search for matches
    Both use the same embedding model that produced the tile vectors.
    """
    theme_name = theme_info["display_name"]
    description = theme_info["description"]
    required = theme_info.get("required_motifs", [])
    supporting = theme_info.get("supporting_motifs", [])
    all_motifs = required + supporting

    log.info(f"Theme {theme_id} ({theme_name}): {len(all_motifs)} motifs")

    # Strategy 1: Search by theme description embedding
    items_by_desc = _search_by_vector(description, f"theme_{theme_id}_desc")

    # Strategy 2: Search by each motif's definition embedding
    items_by_motifs: Dict[str, dict] = {}
    for motif_id in all_motifs:
        motif_info = motif_registry.get(motif_id, {})
        motif_def = motif_info.get("definition", "")
        if not motif_def:
            continue
        # Use motif_id + definition as search text for richer embedding
        search_text = f"{motif_id}: {motif_def}"
        motif_results = _search_by_vector(search_text, f"motif_{motif_id}")
        for ch, info in motif_results.items():
            if ch not in items_by_motifs or info["score"] > items_by_motifs[ch]["score"]:
                items_by_motifs[ch] = info

    # Merge all results — keep highest score per content_hash
    all_hashes: Dict[str, dict] = {}
    for source_results in [items_by_desc, items_by_motifs]:
        for ch, info in source_results.items():
            if ch not in all_hashes or info["score"] > all_hashes[ch].get("score", 0):
                all_hashes[ch] = info

    # Separate into categories
    unenriched_corpus = []
    unenriched_exchanges = []
    anchor_hashes = []

    for ch, info in all_hashes.items():
        source = info.get("source_file", "")
        layer = _source_layer(source)

        if _is_already_compressed(source):
            anchor_hashes.append(ch)
        elif layer == "transcript":
            unenriched_exchanges.append({
                "content_hash": ch,
                "source_file": source,
                "score": round(info.get("score", 0), 4),
                "token_estimate": info.get("token_estimate", 0),
            })
        else:
            unenriched_corpus.append({
                "content_hash": ch,
                "source_file": source,
                "score": round(info.get("score", 0), 4),
                "layer": layer,
                "token_estimate": info.get("token_estimate", 0),
            })

    # Sort by score descending
    unenriched_corpus.sort(key=lambda x: x["score"], reverse=True)
    unenriched_exchanges.sort(key=lambda x: x["score"], reverse=True)

    # Get anchor rosetta summaries for kernel-layer_2 items
    anchors = _get_anchor_rosettas(anchor_hashes)

    log.info(f"  Results: {len(unenriched_corpus)} corpus, "
             f"{len(unenriched_exchanges)} exchanges, {len(anchors)} anchors")

    return {
        "theme_id": theme_id,
        "display_name": theme_name,
        "description": description,
        "required_motifs": required,
        "supporting_motifs": supporting,
        "unenriched_corpus": unenriched_corpus,
        "unenriched_exchanges": unenriched_exchanges,
        "anchor_rosettas": anchors,
        "total_items": len(unenriched_corpus) + len(unenriched_exchanges),
    }


def _search_by_vector(text: str, label: str) -> Dict[str, dict]:
    """Embed text and search Weaviate via nearVector.

    Returns dict of content_hash -> {source_file, score, token_estimate}.
    Only returns search_512 scale tiles (one per content_hash, best score).
    """
    vec = embed_text(text)
    if not vec:
        log.warning(f"  Failed to embed {label}")
        return {}

    vec_str = json.dumps(vec)
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                nearVector: {{ vector: {vec_str}, certainty: {NEAR_VECTOR_CERTAINTY} }}
                where: {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                limit: {NEAR_VECTOR_LIMIT}
            ) {{
                content_hash source_file token_count
                _additional {{ certainty }}
            }}
        }}
    }}"""

    data = weaviate_gql(q, timeout=180)
    if not data:
        log.warning(f"  nearVector query failed for {label}")
        return {}

    tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
    results: Dict[str, dict] = {}

    for t in tiles:
        ch = t.get("content_hash", "")
        if not ch:
            continue
        cert = t.get("_additional", {}).get("certainty", 0) or 0
        if ch not in results or cert > results[ch]["score"]:
            results[ch] = {
                "source_file": t.get("source_file", ""),
                "score": cert,
                "token_estimate": t.get("token_count", 0) or 0,
            }

    log.info(f"    {label}: {len(tiles)} tiles → {len(results)} unique hashes")
    return results


def _get_anchor_rosettas(content_hashes: List[str]) -> List[dict]:
    """Get rosetta summaries for already-enriched items (context anchors)."""
    anchors = []

    for ch in content_hashes[:30]:  # Limit to 30 potential anchors per theme
        q = f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{
                        operator: And,
                        operands: [
                            {{ path: ["content_hash"], operator: Equal, valueText: "{ch}" }},
                            {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                        ]
                    }}
                    limit: 1
                ) {{
                    rosetta_summary dominant_motifs source_file motif_data_json
                }}
            }}
        }}"""

        data = weaviate_gql(q)
        if not data:
            continue

        tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
        if not tiles:
            continue

        t = tiles[0]
        rosetta = t.get("rosetta_summary", "")
        # Include even without rosetta — the anchor still has motif data
        motif_data = t.get("motif_data_json", "")
        if not rosetta and not motif_data:
            continue

        anchors.append({
            "content_hash": ch,
            "source_file": t.get("source_file", ""),
            "rosetta_summary": rosetta or "",
            "dominant_motifs": t.get("dominant_motifs", []) or [],
        })

    return anchors


# ============================================================================
# Token Estimation (batch via Aggregate)
# ============================================================================

def estimate_tokens_for_items(items: List[dict]) -> None:
    """Update token_estimate for items using full_4096 tile Aggregate counts.

    Modifies items in-place. Only queries items where token_estimate is 0.
    """
    need_estimate = [it for it in items if not it.get("token_estimate")]
    if not need_estimate:
        return

    log.info(f"  Estimating tokens for {len(need_estimate)} items...")
    for it in need_estimate:
        ch = it["content_hash"]
        q = f"""{{
            Aggregate {{
                {WEAVIATE_CLASS}(
                    where: {{
                        operator: And,
                        operands: [
                            {{ path: ["content_hash"], operator: Equal, valueText: "{ch}" }},
                            {{ path: ["scale"], operator: Equal, valueText: "full_4096" }}
                        ]
                    }}
                ) {{
                    token_count {{ sum }}
                    meta {{ count }}
                }}
            }}
        }}"""

        data = weaviate_gql(q)
        if data:
            agg = data.get("Aggregate", {}).get(WEAVIATE_CLASS, [{}])
            if agg:
                tc_sum = agg[0].get("token_count", {}).get("sum")
                tile_count = agg[0].get("meta", {}).get("count", 0)
                if tc_sum:
                    it["token_estimate"] = int(tc_sum)
                    it["tile_count"] = tile_count


# ============================================================================
# Build Full Index
# ============================================================================

def build_index(theme_filter: str = None):
    """Build the complete theme search index."""
    mapping = load_canonical_mapping()
    themes = mapping["theme_registry"]
    motif_registry = mapping.get("motif_registry", {})

    if theme_filter:
        themes = {k: v for k, v in themes.items() if k == theme_filter}

    log.info(f"Building theme search index for {len(themes)} themes...")
    t0 = time.time()

    index = {}

    for theme_id, theme_info in sorted(themes.items()):
        result = find_items_for_theme(theme_id, theme_info, motif_registry)

        # Estimate tokens for items without estimates
        estimate_tokens_for_items(result["unenriched_corpus"])
        estimate_tokens_for_items(result["unenriched_exchanges"])

        theme_key = f"theme_{theme_id}_{theme_info['display_name'].lower().replace(' ', '_')}"
        index[theme_key] = result

    elapsed = time.time() - t0

    # Save index
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, INDEX_PATH)

    log.info(f"\nIndex saved to {INDEX_PATH} ({elapsed:.1f}s)")
    show_stats(index)


def show_stats(index: dict = None):
    """Display index statistics."""
    if index is None:
        if not os.path.exists(INDEX_PATH):
            print("No index found. Run --build first.")
            return
        with open(INDEX_PATH) as f:
            index = json.load(f)

    print(f"\n{'='*80}")
    print(f"Theme Search Index — {len(index)} themes")
    print(f"{'='*80}")

    total_corpus = 0
    total_exchanges = 0
    total_anchors = 0
    total_tokens = 0
    all_corpus_hashes = set()
    all_exchange_hashes = set()

    for key, theme in sorted(index.items()):
        n_corpus = len(theme.get("unenriched_corpus", []))
        n_exchanges = len(theme.get("unenriched_exchanges", []))
        n_anchors = len(theme.get("anchor_rosettas", []))
        tk = sum(it.get("token_estimate", 0) for it in theme.get("unenriched_corpus", []))
        tk += sum(it.get("token_estimate", 0) for it in theme.get("unenriched_exchanges", []))
        total_corpus += n_corpus
        total_exchanges += n_exchanges
        total_anchors += n_anchors
        total_tokens += tk

        for it in theme.get("unenriched_corpus", []):
            all_corpus_hashes.add(it["content_hash"])
        for it in theme.get("unenriched_exchanges", []):
            all_exchange_hashes.add(it["content_hash"])

        print(f"  {theme['theme_id']} {theme['display_name']:25s} "
              f"corpus={n_corpus:5d}  exchanges={n_exchanges:5d}  "
              f"anchors={n_anchors:2d}  tokens={tk:>10,}")

    print(f"\n  TOTAL (with overlap): corpus={total_corpus:,}  exchanges={total_exchanges:,}  anchors={total_anchors}")
    print(f"  UNIQUE: corpus={len(all_corpus_hashes):,}  exchanges={len(all_exchange_hashes):,}")
    print(f"  TOTAL tokens: {total_tokens:,}")
    print(f"  Note: Items may appear in multiple themes")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HMM Theme Search Index Builder")
    parser.add_argument("--build", action="store_true", help="Build full index")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    parser.add_argument("--theme", type=str, help="Build index for one theme (e.g., 001)")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.build or args.theme:
        build_index(theme_filter=args.theme)
        return

    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nInterrupted")
    except Exception:
        log.error("Fatal error:", exc_info=True)
        sys.exit(1)
