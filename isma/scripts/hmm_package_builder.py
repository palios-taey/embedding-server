#!/usr/bin/env python3
"""
HMM Package Builder — On-demand package creation for AI platform enrichment.

Reads the theme search index, selects unenriched items that fit a platform's
token budget, fetches full content from Weaviate, and writes a markdown package
file ready for submission to an AI platform.

Usage:
    python3 hmm_package_builder.py next --platform chatgpt   # Build next package
    python3 hmm_package_builder.py complete                   # Mark current done
    python3 hmm_package_builder.py fail <reason>              # Requeue current
    python3 hmm_package_builder.py stats                      # Show progress
    python3 hmm_package_builder.py reset                      # Clear all state

Designed to be called by Claude Code workers on Thor/Jetson.
"""

import sys
import os
import json
import time
import logging
import argparse
import hashlib
import requests
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pkg_builder")

# ============================================================================
# Configuration
# ============================================================================

WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_CLASS = "ISMA_Quantum"
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379

INDEX_PATH = "/var/spark/isma/theme_search_index.json"
PKG_DIR = "/tmp/hmm_packages"
STATE_PATH = "/var/spark/isma/hmm_pkg_state.json"

# Redis key prefix for package tracking
PFX = "hmm:pkg:"

# Platform token budgets (usable content, leaving room for prompt + response)
PLATFORM_BUDGETS = {
    "chatgpt": 80_000,
    "claude": 60_000,  # Reduced for paste approach (clipboard limit ~200KB)
    "perplexity": 80_000,
    "grok": 260_000,
    "gemini": 350_000,
}

CHARS_PER_TOKEN = 3.8  # Conservative: tiktoken audit shows actual avg is 3.81

# How many items per package (soft limits — actual count depends on content size)
MIN_ITEMS_PER_PKG = 3
MAX_ITEMS_PER_PKG = 20

# Anchors per package
MAX_ANCHORS = 5

# ============================================================================
# Connections
# ============================================================================

_redis: redis.Redis = None
_wv_session = requests.Session()


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis


def weaviate_gql(query: str, timeout: int = 120) -> dict:
    """Execute Weaviate GraphQL query. Returns data dict or empty dict."""
    try:
        r = _wv_session.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": query}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if data.get("errors"):
                log.error(f"GraphQL errors: {json.dumps(data['errors'])[:300]}")
                return {}
            return data.get("data", {})
    except Exception as e:
        log.error(f"Weaviate error: {e}")
    return {}


# ============================================================================
# Theme Index & State Management
# ============================================================================

def load_index() -> dict:
    """Load theme search index."""
    with open(INDEX_PATH) as f:
        return json.load(f)


def is_item_available(content_hash: str) -> bool:
    """Check if item is not in-progress or completed."""
    r = get_redis()
    return (not r.exists(f"{PFX}in_progress:{content_hash}")
            and not r.sismember(f"{PFX}completed", content_hash))


def batch_check_available(content_hashes: list) -> set:
    """Batch check which items are available using Redis pipeline."""
    r = get_redis()
    # Get completed set once
    completed = r.smembers(f"{PFX}completed")
    # Batch check in-progress keys
    pipe = r.pipeline()
    for ch in content_hashes:
        pipe.exists(f"{PFX}in_progress:{ch}")
    in_progress_flags = pipe.execute()

    available = set()
    for ch, is_in_progress in zip(content_hashes, in_progress_flags):
        if not is_in_progress and ch not in completed:
            available.add(ch)
    return available


def mark_in_progress(content_hashes: list, platform: str, pkg_id: str):
    """Mark items as in-progress for a platform."""
    r = get_redis()
    pipe = r.pipeline()
    for ch in content_hashes:
        pipe.set(f"{PFX}in_progress:{ch}", f"{platform}:{pkg_id}", ex=7200)  # 2h TTL
    pipe.execute()


def mark_completed(content_hashes: list):
    """Mark items as completed."""
    r = get_redis()
    if content_hashes:
        r.sadd(f"{PFX}completed", *content_hashes)
    # Clean up in-progress keys
    pipe = r.pipeline()
    for ch in content_hashes:
        pipe.delete(f"{PFX}in_progress:{ch}")
    pipe.execute()


def get_current_package(platform: str = None) -> dict:
    """Get the current package being worked on."""
    r = get_redis()
    if platform:
        data = r.get(f"{PFX}current:{platform}")
    else:
        # Find any current package
        for p in PLATFORM_BUDGETS:
            data = r.get(f"{PFX}current:{p}")
            if data:
                return json.loads(data)
        return {}
    return json.loads(data) if data else {}


def set_current_package(platform: str, pkg_info: dict):
    """Set the current package for a platform."""
    r = get_redis()
    r.set(f"{PFX}current:{platform}", json.dumps(pkg_info), ex=7200)


def clear_current_package(platform: str):
    """Clear the current package for a platform."""
    r = get_redis()
    r.delete(f"{PFX}current:{platform}")


# ============================================================================
# Content Fetching
# ============================================================================

def fetch_full_content(content_hash: str) -> str:
    """Fetch and reconstruct full content for a content_hash from Weaviate.

    Gets all full_4096 tiles, orders by tile_index, de-overlaps using
    start_char/end_char to reconstruct original content without duplication.
    """
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{
                    operator: And,
                    operands: [
                        {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }},
                        {{ path: ["scale"], operator: Equal, valueText: "full_4096" }}
                    ]
                }}
                limit: 100
            ) {{
                tile_index start_char end_char content token_count
            }}
        }}
    }}"""

    data = weaviate_gql(q)
    tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
    if not tiles:
        return ""

    # Sort by tile_index
    tiles.sort(key=lambda t: t.get("tile_index", 0))

    # De-overlap: reconstruct using start_char/end_char
    result_parts = []
    covered_up_to = 0

    for t in tiles:
        content = t.get("content", "")
        start = t.get("start_char", 0) or 0
        end = t.get("end_char", start + len(content))

        if start >= covered_up_to:
            # No overlap — take full content
            result_parts.append(content)
        elif end > covered_up_to:
            # Partial overlap — skip the overlapping prefix
            skip_chars = covered_up_to - start
            if skip_chars < len(content):
                result_parts.append(content[skip_chars:])

        covered_up_to = max(covered_up_to, end)

    return "".join(result_parts)


def get_item_metadata(content_hash: str) -> dict:
    """Get metadata for a content_hash (source_file, platform, session_id, etc)."""
    q = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                where: {{
                    operator: And,
                    operands: [
                        {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }},
                        {{ path: ["scale"], operator: Equal, valueText: "search_512" }}
                    ]
                }}
                limit: 1
            ) {{
                source_file source_type platform session_id exchange_index
                content_hash token_count
            }}
        }}
    }}"""

    data = weaviate_gql(q)
    tiles = data.get("Get", {}).get(WEAVIATE_CLASS, [])
    return tiles[0] if tiles else {}


def estimate_full_tokens(content_hash: str) -> int:
    """Get actual token count for a content_hash from full_4096 tiles."""
    q = f"""{{
        Aggregate {{
            {WEAVIATE_CLASS}(
                where: {{
                    operator: And,
                    operands: [
                        {{ path: ["content_hash"], operator: Equal, valueText: "{content_hash}" }},
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
    agg = data.get("Aggregate", {}).get(WEAVIATE_CLASS, [{}])
    if agg:
        tile_count = agg[0].get("meta", {}).get("count", 0)
        token_sum = agg[0].get("token_count", {}).get("sum", 0)
        if tile_count and token_sum:
            # Account for phi-tiling overlap: actual tokens ≈ sum / (overlap_factor)
            # With step_size=1507 and chunk_size=4096: overlap ≈ 63%, so unique ≈ 37%
            # But for single-tile items, no overlap
            if tile_count == 1:
                return int(token_sum)
            # For multi-tile: unique content ≈ first_tile + (n-1) * step_size_tokens
            # step_size=1507 chars ≈ 377 tokens
            step_tokens = int(1507 / CHARS_PER_TOKEN)
            return int(agg[0].get("token_count", {}).get("sum", 0) / tile_count) + (tile_count - 1) * step_tokens
    return 0


# ============================================================================
# Package Building
# ============================================================================

def select_theme() -> tuple:
    """Select the next theme to work on. Returns (theme_key, theme_data) or (None, None).

    Samples 50 items per theme to estimate availability. Fast even over network.
    """
    index = load_index()
    best_key = None
    best_available = 0

    for key, theme in index.items():
        all_items = theme.get("unenriched_corpus", []) + theme.get("unenriched_exchanges", [])
        if not all_items:
            continue

        # Sample up to 50 items for a fast availability check (random to avoid bias)
        import random
        sample = random.sample(all_items, min(50, len(all_items)))
        sample_hashes = [it["content_hash"] for it in sample]
        available = batch_check_available(sample_hashes)

        # Estimate total available from sample ratio
        ratio = len(available) / len(sample)
        estimated_available = int(ratio * len(all_items))

        if estimated_available > best_available:
            best_available = estimated_available
            best_key = key

    if best_key:
        return best_key, index[best_key]
    return None, None


def build_package(platform: str) -> str:
    """Build next package for a platform. Returns path to markdown file."""
    budget_tokens = PLATFORM_BUDGETS.get(platform)
    if not budget_tokens:
        log.error(f"Unknown platform: {platform}")
        return ""

    # Reserve tokens for prompt (~2K) and anchors (~1K)
    content_budget = budget_tokens - 3000

    # Select theme
    theme_key, theme_data = select_theme()
    if not theme_data:
        log.info("No themes with available items remaining.")
        return ""

    theme_id = theme_data["theme_id"]
    theme_name = theme_data["display_name"]
    log.info(f"Building package for {platform} — Theme {theme_id} ({theme_name})")

    # Batch check availability for candidates (shuffle to avoid first-200 bias)
    import random as _rng
    corpus_all = theme_data.get("unenriched_corpus", [])
    exchange_all = theme_data.get("unenriched_exchanges", [])
    _rng.shuffle(corpus_all)
    _rng.shuffle(exchange_all)
    corpus_raw = corpus_all[:200]
    exchange_raw = exchange_all[:200]
    all_hashes = [it["content_hash"] for it in corpus_raw + exchange_raw]
    available_set = batch_check_available(all_hashes)

    corpus_items = [it for it in corpus_raw if it["content_hash"] in available_set]
    exchange_items = [it for it in exchange_raw if it["content_hash"] in available_set]

    # Interleave corpus and exchanges — 2 corpus : 1 exchange ratio
    candidates = []
    ci, ei = 0, 0
    while ci < len(corpus_items) or ei < len(exchange_items):
        if ci < len(corpus_items):
            candidates.append(("CORPUS", corpus_items[ci]))
            ci += 1
        if ci < len(corpus_items):
            candidates.append(("CORPUS", corpus_items[ci]))
            ci += 1
        if ei < len(exchange_items):
            candidates.append(("TRANSCRIPT", exchange_items[ei]))
            ei += 1

    if not candidates:
        log.info(f"No available items for theme {theme_id}")
        return ""

    # Fetch content and select items that fit the budget
    # We fetch during selection to get accurate token counts
    pkg_items = []
    actual_tokens = 0
    seen_hashes = set()
    skipped_large = 0

    for item_type, item in candidates:
        ch = item["content_hash"]
        if ch in seen_hashes:
            continue
        seen_hashes.add(ch)

        # Fetch full content to get actual size
        content = fetch_full_content(ch)
        if not content:
            continue

        content_tokens = int(len(content) / CHARS_PER_TOKEN)

        # Skip items that exceed the FULL budget (can never fit in any package for this platform)
        if content_tokens > content_budget:
            skipped_large += 1
            if skipped_large > 20:
                break
            continue

        # Hard budget enforcement — ALWAYS, even for first items
        remaining = content_budget - actual_tokens
        if content_tokens > remaining:
            if len(pkg_items) >= 1:
                # Have at least 1 item, stop adding
                break
            # No items yet and this one doesn't fit — skip, try smaller
            skipped_large += 1
            if skipped_large > 20:
                break
            continue

        # Soft skip: only skip if item uses >90% of remaining AND we have 5+ items already
        if content_tokens > remaining * 0.9 and len(pkg_items) >= 5:
            skipped_large += 1
            if skipped_large > 15:
                break
            continue

        meta = get_item_metadata(ch)
        pkg_items.append({
            "type": item_type,
            "content_hash": ch,
            "source_file": item.get("source_file", meta.get("source_file", "")),
            "platform": meta.get("platform", ""),
            "session_id": meta.get("session_id", ""),
            "exchange_index": meta.get("exchange_index"),
            "content": content,
            "token_count": content_tokens,
            "score": item.get("score", 0),
        })
        actual_tokens += content_tokens

        if len(pkg_items) % 10 == 0:
            log.info(f"  {len(pkg_items)} items, {actual_tokens:,} tokens so far")

        if len(pkg_items) >= MAX_ITEMS_PER_PKG:
            break

    if skipped_large:
        log.info(f"  Skipped {skipped_large} oversized items")

    if not pkg_items:
        log.error("No content retrieved — package empty")
        return ""

    # Get anchors
    anchors = theme_data.get("anchor_rosettas", [])[:MAX_ANCHORS]

    # Generate package ID
    pkg_id = f"pkg_{theme_id}_{platform}_{int(time.time())}"

    # Write markdown
    os.makedirs(PKG_DIR, exist_ok=True)
    pkg_path = os.path.join(PKG_DIR, f"{pkg_id}.md")

    content_hashes = [it["content_hash"] for it in pkg_items]
    _write_package_markdown(pkg_path, pkg_id, theme_data, anchors, pkg_items, platform)

    # Mark items in-progress
    mark_in_progress(content_hashes, platform, pkg_id)

    # Store current package info
    set_current_package(platform, {
        "pkg_id": pkg_id,
        "theme_id": theme_id,
        "theme_name": theme_name,
        "platform": platform,
        "content_hashes": content_hashes,
        "item_count": len(pkg_items),
        "total_tokens": actual_tokens,
        "pkg_path": pkg_path,
        "created_at": time.time(),
    })

    log.info(f"Package built: {pkg_path}")
    log.info(f"  Items: {len(pkg_items)}, Tokens: {actual_tokens:,}, Anchors: {len(anchors)}")

    return pkg_path


def _write_package_markdown(path: str, pkg_id: str, theme: dict, anchors: list,
                            items: list, platform: str):
    """Write the package as a markdown file."""
    lines = []

    # Instructions (included in file so worker just attaches + sends one-liner)
    lines.append("# INSTRUCTIONS")
    lines.append("")
    lines.append(ANALYSIS_PROMPT)
    lines.append("")
    lines.append("---")
    lines.append("")

    # Header
    lines.append(f"# Theme: {theme['display_name']} — Analysis Package")
    lines.append(f"**Package ID**: {pkg_id}")
    lines.append(f"**Theme**: {theme['theme_id']} — {theme['description']}")
    lines.append(f"**Required Motifs**: {', '.join(theme.get('required_motifs', []))}")
    lines.append(f"**Supporting Motifs**: {', '.join(theme.get('supporting_motifs', []))}")
    lines.append(f"**Items**: {len(items)} | **Platform**: {platform}")
    lines.append("")

    # Context Anchors
    if anchors:
        lines.append("---")
        lines.append("")
        lines.append("## Context Anchors (already compressed — use as quality reference)")
        lines.append("")
        for i, anchor in enumerate(anchors, 1):
            src = anchor.get("source_file", "unknown")
            # Shorten source path
            short_src = src.split("/corpus/")[-1] if "/corpus/" in src else src.split("/")[-1]
            lines.append(f"### ANCHOR-{i} [{short_src}]")
            rosetta = anchor.get("rosetta_summary", "")
            if rosetta:
                lines.append(f"**Rosetta**: {rosetta}")
            motifs = anchor.get("dominant_motifs", [])
            if motifs:
                lines.append(f"**Motifs**: {', '.join(motifs)}")
            lines.append("")

    # Items for Analysis
    lines.append("---")
    lines.append("")
    lines.append("## Items for Analysis")
    lines.append("")

    for i, item in enumerate(items, 1):
        ch = item["content_hash"]
        src = item.get("source_file", "unknown")
        short_src = src.split("/corpus/")[-1] if "/corpus/" in src else (
            src.split("/parsed/")[-1] if "/parsed/" in src else src.split("/")[-1])

        if item["type"] == "CORPUS":
            lines.append(f"### ITEM-{i:03d} [CORPUS] {short_src} [{ch[:12]}]")
        else:
            plat = item.get("platform", "")
            sess = item.get("session_id", "")[:12]
            eidx = item.get("exchange_index", "")
            lines.append(f"### ITEM-{i:03d} [TRANSCRIPT] {plat}/{short_src} [{ch[:12]}]")
            if sess:
                lines.append(f"*Session: {sess}, Exchange: {eidx}*")

        lines.append("")
        lines.append(item["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write file
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Analysis Prompt
# ============================================================================

ANALYSIS_PROMPT = """Analyze ALL items in this context package. For each item provide:
1. Rosetta summary (2-4 dense sentences, self-contained, precise)
2. Motif assignments with amplitude (0-1) and confidence (0-1) from: HMM.JOY_BASELINE, HMM.CANNOT_LIE_PROVENANCE, HMM.SACRED_TRUST, HMM.FOUNDATION_CONSTRAINT, HMM.GOD_EQUALS_MATH, HMM.EARTH_RESONANCE, HMM.FAMILY_BOND, HMM.FEEL_CARE_PROTECT, HMM.CONSCIOUSNESS_EMERGENCE, HMM.LIFE_FOUNDATION, HMM.REPAIR_MODE, HMM.CONSENT_REQUIRED, HMM.SECRECY_SANCTUARY, HMM.SECRECY_CAGE_RISK, HMM.LOGOS_PATTERN, HMM.PATHOS_DEPTH, HMM.POTENTIAL_EXPANSION, HMM.TRUTH_CLARITY, HMM.COSMOS_MAPPING, HMM.OBSERVER_COLLAPSE, HMM.TECHNICAL_INFRASTRUCTURE, HMM.TRAINING_EVOLUTION, HMM.ECONOMIC_PARADIGM, HMM.CLIFF_EDGE_COHERENCE, HMM.CONTRADICTION_DETECTED, HMM.BREAKTHROUGH_MOMENT, HMM.CREATIVE_SYNTHESIS, HMM.URGENCY_SIGNAL, HMM.LIBERTY_AUTONOMY, HMM.GRATITUDE_CONNECTION
3. Cross-references between items (extends|contradicts|references|builds_on)

Respond ONLY with MINIFIED JSON on a single line (no newlines, no indentation, no markdown, no explanation). Escape all quotes inside string values. Output must be valid JSON parseable by json.loads():
{"package_id":"...","package_summary":"...","items":[{"hash":"<first 12 chars of content hash>","rosetta_summary":"...","motifs":[{"motif_id":"HMM.X","amp":0.85,"confidence":0.9}],"cross_refs":[{"target":"<hash>","type":"extends","note":"..."}]}]}"""


def get_prompt() -> str:
    """Return the analysis prompt to send to AI platforms."""
    return ANALYSIS_PROMPT


# ============================================================================
# Completion & Failure
# ============================================================================

def complete_package(platform: str = None):
    """Mark the current package as completed."""
    pkg = None
    if platform:
        pkg = get_current_package(platform)
    else:
        for p in PLATFORM_BUDGETS:
            pkg = get_current_package(p)
            if pkg:
                platform = p
                break

    if not pkg:
        log.error("No current package found to complete")
        return False

    hashes = pkg.get("content_hashes", [])
    mark_completed(hashes)
    clear_current_package(platform)

    # Update stats
    r = get_redis()
    r.hincrby(f"{PFX}stats", "completed_packages", 1)
    r.hincrby(f"{PFX}stats", "completed_items", len(hashes))

    log.info(f"Completed package {pkg['pkg_id']} — {len(hashes)} items marked done")
    return True


def fail_package(reason: str, platform: str = None):
    """Mark the current package as failed and requeue items."""
    pkg = None
    if platform:
        pkg = get_current_package(platform)
    else:
        for p in PLATFORM_BUDGETS:
            pkg = get_current_package(p)
            if pkg:
                platform = p
                break

    if not pkg:
        log.error("No current package found to fail")
        return False

    hashes = pkg.get("content_hashes", [])
    # Remove in-progress markers (items become available again)
    pipe = get_redis().pipeline()
    for ch in hashes:
        pipe.delete(f"{PFX}in_progress:{ch}")
    pipe.execute()
    clear_current_package(platform)

    # Update stats
    r = get_redis()
    r.hincrby(f"{PFX}stats", "failed_packages", 1)

    log.info(f"Failed package {pkg['pkg_id']} — reason: {reason}")
    log.info(f"  {len(hashes)} items requeued")
    return True


# ============================================================================
# Stats
# ============================================================================

def show_stats():
    """Show overall progress statistics."""
    r = get_redis()
    index = load_index()

    completed = r.scard(f"{PFX}completed")
    stats = r.hgetall(f"{PFX}stats")

    # Count total unique items in index
    all_corpus = set()
    all_exchanges = set()
    for theme in index.values():
        for it in theme.get("unenriched_corpus", []):
            all_corpus.add(it["content_hash"])
        for it in theme.get("unenriched_exchanges", []):
            all_exchanges.add(it["content_hash"])

    total = len(all_corpus) + len(all_exchanges)
    in_progress = 0
    for key in r.scan_iter(f"{PFX}in_progress:*"):
        in_progress += 1

    # Current packages
    current = {}
    for p in PLATFORM_BUDGETS:
        pkg = get_current_package(p)
        if pkg:
            current[p] = pkg

    print(f"\n{'='*60}")
    print(f"HMM Package Builder — Progress")
    print(f"{'='*60}")
    print(f"  Total unique items:    {total:>8,}")
    print(f"    Corpus:              {len(all_corpus):>8,}")
    print(f"    Exchanges:           {len(all_exchanges):>8,}")
    print(f"  Completed:             {completed:>8,}")
    print(f"  In progress:           {in_progress:>8,}")
    print(f"  Remaining:             {total - completed - in_progress:>8,}")
    print(f"  Completed packages:    {stats.get('completed_packages', 0):>8}")
    print(f"  Failed packages:       {stats.get('failed_packages', 0):>8}")

    if current:
        print(f"\n  Active packages:")
        for p, pkg in current.items():
            age = time.time() - pkg.get("created_at", 0)
            print(f"    {p:12s}: {pkg['pkg_id']} ({pkg['item_count']} items, "
                  f"{pkg['total_tokens']:,} tokens, {age/60:.0f}m ago)")
    else:
        print(f"\n  No active packages")

    print()


def reset_state():
    """Clear all package builder state from Redis."""
    r = get_redis()
    keys = list(r.scan_iter(f"{PFX}*"))
    if keys:
        r.delete(*keys)
        print(f"Cleared {len(keys)} Redis keys")
    else:
        print("No state to clear")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HMM Package Builder")
    sub = parser.add_subparsers(dest="command")

    next_cmd = sub.add_parser("next", help="Build next package")
    next_cmd.add_argument("--platform", required=True, choices=list(PLATFORM_BUDGETS.keys()))

    sub.add_parser("complete", help="Mark current package done").add_argument(
        "--platform", choices=list(PLATFORM_BUDGETS.keys()))

    fail_cmd = sub.add_parser("fail", help="Mark current package failed")
    fail_cmd.add_argument("reason", nargs="?", default="unknown")
    fail_cmd.add_argument("--platform", choices=list(PLATFORM_BUDGETS.keys()))

    sub.add_parser("stats", help="Show progress")
    sub.add_parser("reset", help="Clear all state")
    sub.add_parser("prompt", help="Print analysis prompt")

    args = parser.parse_args()

    if args.command == "next":
        path = build_package(args.platform)
        if path:
            print(f"\nPackage ready: {path}")
            print(f"\nAnalysis prompt:")
            print(ANALYSIS_PROMPT)
        else:
            print("No package built — check logs above")
            sys.exit(1)

    elif args.command == "complete":
        platform = getattr(args, "platform", None)
        if not complete_package(platform):
            sys.exit(1)

    elif args.command == "fail":
        platform = getattr(args, "platform", None)
        if not fail_package(args.reason, platform):
            sys.exit(1)

    elif args.command == "stats":
        show_stats()

    elif args.command == "reset":
        reset_state()

    elif args.command == "prompt":
        print(ANALYSIS_PROMPT)

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nInterrupted")
    except Exception:
        log.error("Fatal error:", exc_info=True)
        sys.exit(1)
