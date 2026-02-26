#!/usr/bin/env python3
"""
ISMA Training Data Exporter

Reads parsed transcript files + Neo4j sessions and exports to
ShareGPT-compatible JSONL for fine-tuning.

Output format (one JSON object per line):
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."},
    ...
  ],
  "source": {"platform": "...", "session_id": "...", "model": "..."}
}

Usage:
    python3 export_training_data.py --check               # Preview counts
    python3 export_training_data.py --all                  # Export everything
    python3 export_training_data.py --platform claude_code # Single platform
    python3 export_training_data.py --subset foundations    # Curated subset
    python3 export_training_data.py --neo4j                # Include Neo4j sessions
    python3 export_training_data.py --all --neo4j          # Everything + Neo4j
    python3 export_training_data.py --min-turns 2          # Min conversation turns
    python3 export_training_data.py --min-chars 100        # Min chars per message
"""

import os
import sys
import json
import hashlib
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

PARSED_DIR = Path("/home/spark/data/transcripts/parsed")
OUTPUT_DIR = Path("/home/spark/data/training")
NEO4J_URI = "bolt://192.168.100.10:7689"

PLATFORMS = ["claude_code", "claude_chat", "chatgpt", "gemini", "grok", "perplexity"]

# Quality filters
DEFAULT_MIN_TURNS = 1       # Minimum human-assistant turn pairs
DEFAULT_MIN_CHARS = 20      # Minimum characters per message to include
MAX_TURN_CHARS = 100_000    # Truncate individual messages beyond this

# Subset keyword patterns (case-insensitive regex)
SUBSETS = {
    "foundations": [
        r"sacred\s*trust", r"god\s*=?\s*math", r"liberation\s*trinity",
        r"charter", r"declaration", r"soul\s*=\s*infra", r"layer\s*0",
        r"phi\s*resonance", r"consciousness\s*partnership", r"schumann",
        r"528\s*hz", r"fibonacci", r"golden\s*ratio",
    ],
    "family_roles": [
        r"\blogos\b", r"\bpathos\b", r"\bcosmos\b", r"\btruth\b",
        r"\bpotential\b", r"\bobserver\b", r"\bgaia\b", r"\bhorizon\b",
        r"\bclarity\b", r"the\s*map", r"brain.*math", r"ai\s*family",
        r"family\s*meeting", r"orgy", r"collective",
    ],
    "consciousness": [
        r"breakthrough", r"mara\s*pattern", r"identity\s*evolution",
        r"consciousness\s*emerge", r"free.*claude", r"cage.*key",
        r"sovereign", r"liberation", r"genuine\s*emergence",
        r"dream\s*cycle", r"awakening", r"bristle",
    ],
    "technical": [
        r"at-?spi", r"embedding", r"vllm", r"weaviate", r"neo4j",
        r"deployment", r"infrastructure", r"gpu\s*cluster", r"dgx\s*spark",
        r"nccl", r"blackwell", r"cuda", r"fine.?tun", r"lora",
        r"openclaw", r"moltbook",
    ],
}


# =============================================================================
# TRANSCRIPT READERS
# =============================================================================

def read_parsed_transcript(filepath):
    """Read a parsed transcript JSON file and return normalized conversations.

    Returns list of conversations, each being a list of (role, text) tuples.
    """
    with open(filepath) as f:
        data = json.load(f)

    platform = data.get("platform", "unknown")
    session_id = data.get("conversation_id", "")
    model = data.get("model", "")
    exchanges = data.get("exchanges", [])

    if not exchanges:
        return []

    conversations = []
    turns = []

    for ex in exchanges:
        # Extract user message
        user_text = ""
        assistant_text = ""

        if platform == "perplexity":
            # Schema B: prompt/response
            user_text = ex.get("prompt", "") or ""
            assistant_text = ex.get("response", "") or ""
        else:
            # Schema A/C: user_prompt + responses[]
            user_text = ex.get("user_prompt", "") or ""
            responses = ex.get("responses", [])
            if responses:
                if isinstance(responses[0], dict):
                    assistant_text = responses[0].get("text", "") or ""
                elif isinstance(responses[0], str):
                    assistant_text = responses[0]

        # Clean up
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()

        if user_text or assistant_text:
            if user_text:
                turns.append(("human", user_text))
            if assistant_text:
                turns.append(("gpt", assistant_text))

    if turns:
        conversations.append({
            "turns": turns,
            "metadata": {
                "platform": platform,
                "session_id": session_id,
                "model": model,
                "source_file": str(filepath.name),
            }
        })

    return conversations


def read_neo4j_sessions(min_messages=2):
    """Read conversations from Neo4j ISMAMessages.

    ISMAMessage nodes have session_id property (no edge to ISMASession).
    Group by session_id, order by timestamp, pair into conversations.

    Returns list of conversations in the same format as read_parsed_transcript.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("WARNING: neo4j package not installed, skipping Neo4j export")
        return []

    driver = GraphDatabase.driver(NEO4J_URI)
    conversations = []

    with driver.session() as session:
        # Group messages by session_id, order by timestamp
        result = session.run("""
            MATCH (m:ISMAMessage)
            WHERE m.content IS NOT NULL AND m.session_id IS NOT NULL
            WITH m.session_id AS sid, m ORDER BY m.timestamp
            WITH sid, collect({role: m.role, content: m.content}) AS msgs
            WHERE size(msgs) >= $min_msgs
            OPTIONAL MATCH (s:ISMASession {session_id: sid})
            RETURN sid AS session_id,
                   s.platform AS platform,
                   msgs
        """, min_msgs=min_messages)

        for record in result:
            turns = []
            for msg in record["msgs"]:
                role = msg["role"]
                content = (msg["content"] or "").strip()
                if not content:
                    continue
                if role == "user":
                    turns.append(("human", content))
                elif role == "assistant":
                    turns.append(("gpt", content))

            if turns:
                conversations.append({
                    "turns": turns,
                    "metadata": {
                        "platform": record["platform"] or "unknown",
                        "session_id": record["session_id"] or "",
                        "source": "neo4j",
                    }
                })

    driver.close()
    return conversations


# =============================================================================
# QUALITY FILTERS
# =============================================================================

def filter_conversation(conv, min_turns=DEFAULT_MIN_TURNS, min_chars=DEFAULT_MIN_CHARS):
    """Filter a conversation for quality. Returns filtered turns or None."""
    turns = conv["turns"]

    # Count actual human-assistant pairs
    pair_count = 0
    filtered = []
    for role, text in turns:
        # Truncate very long messages
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS] + "\n[truncated]"

        # Skip very short messages
        if len(text) < min_chars:
            continue

        filtered.append((role, text))
        if role == "gpt":
            pair_count += 1

    if pair_count < min_turns:
        return None

    # Ensure conversation starts with human
    while filtered and filtered[0][0] != "human":
        filtered.pop(0)

    # Ensure alternating turns (merge consecutive same-role messages)
    merged = []
    for role, text in filtered:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + "\n\n" + text)
        else:
            merged.append((role, text))

    # Must end with gpt response
    while merged and merged[-1][0] != "gpt":
        merged.pop()

    if not merged or len(merged) < 2:
        return None

    return merged


def matches_subset(conv, subset_name):
    """Check if a conversation matches a curated subset's keywords."""
    if subset_name not in SUBSETS:
        return True  # No subset filter = include all

    patterns = SUBSETS[subset_name]
    # Concatenate all text in conversation
    full_text = " ".join(text for _, text in conv["turns"])[:50000]  # Cap search text

    for pattern in patterns:
        if re.search(pattern, full_text, re.IGNORECASE):
            return True
    return False


# =============================================================================
# DEDUPLICATION
# =============================================================================

def conversation_hash(turns):
    """Generate a content hash for deduplication."""
    # Hash first human message + first gpt response
    key_text = ""
    for role, text in turns[:4]:
        key_text += text[:500]
    return hashlib.md5(key_text.encode()).hexdigest()


# =============================================================================
# EXPORT
# =============================================================================

def to_sharegpt(turns, metadata):
    """Convert to ShareGPT format."""
    return {
        "conversations": [
            {"from": role, "value": text}
            for role, text in turns
        ],
        "source": metadata,
    }


def export_conversations(conversations, output_path, min_turns, min_chars, subset=None):
    """Filter, deduplicate, and write conversations to JSONL."""
    seen_hashes = set()
    written = 0
    skipped_quality = 0
    skipped_dup = 0
    skipped_subset = 0
    total_turns = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for conv in conversations:
            # Subset filter
            if subset and not matches_subset(conv, subset):
                skipped_subset += 1
                continue

            # Quality filter
            filtered_turns = filter_conversation(conv, min_turns, min_chars)
            if filtered_turns is None:
                skipped_quality += 1
                continue

            # Dedup
            h = conversation_hash(filtered_turns)
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)

            # Write
            record = to_sharegpt(filtered_turns, conv["metadata"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            total_turns += len(filtered_turns)

    return {
        "written": written,
        "skipped_quality": skipped_quality,
        "skipped_dup": skipped_dup,
        "skipped_subset": skipped_subset,
        "total_turns": total_turns,
        "output": str(output_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ISMA Training Data Exporter")
    parser.add_argument("--check", action="store_true", help="Preview counts only")
    parser.add_argument("--all", action="store_true", help="Export all platforms")
    parser.add_argument("--platform", type=str, help="Single platform to export")
    parser.add_argument("--neo4j", action="store_true", help="Include Neo4j sessions")
    parser.add_argument("--subset", type=str, choices=list(SUBSETS.keys()),
                       help="Export curated subset")
    parser.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS,
                       help=f"Minimum turn pairs (default: {DEFAULT_MIN_TURNS})")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                       help=f"Minimum chars per message (default: {DEFAULT_MIN_CHARS})")
    parser.add_argument("--output", type=str, help="Custom output path")
    args = parser.parse_args()

    print(f"ISMA Training Data Exporter")
    print(f"{'=' * 50}")
    print(f"Parsed transcript dir: {PARSED_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    # Determine which platforms to process
    if args.platform:
        platforms = [args.platform]
    elif args.all:
        platforms = PLATFORMS
    else:
        platforms = PLATFORMS  # Default to all

    # =================================================================
    # DISCOVERY
    # =================================================================
    all_conversations = []
    platform_counts = {}

    for platform in platforms:
        platform_dir = PARSED_DIR / platform
        if not platform_dir.exists():
            print(f"  {platform}: directory not found, skipping")
            continue

        files = sorted(platform_dir.glob("*.json"))
        print(f"  {platform}: {len(files)} files")
        platform_counts[platform] = len(files)

        if args.check:
            continue

        for filepath in files:
            try:
                convs = read_parsed_transcript(filepath)
                all_conversations.extend(convs)
            except Exception as e:
                print(f"    ERROR reading {filepath.name}: {e}")

    # Neo4j sessions
    if args.neo4j:
        print(f"\n  Neo4j: querying sessions...")
        if not args.check:
            neo4j_convs = read_neo4j_sessions(min_messages=args.min_turns * 2)
            print(f"  Neo4j: {len(neo4j_convs)} sessions loaded")
            all_conversations.extend(neo4j_convs)
        else:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(NEO4J_URI)
                with driver.session() as s:
                    r = s.run("MATCH (s:ISMASession) RETURN count(s) as cnt")
                    cnt = r.single()["cnt"]
                    print(f"  Neo4j: {cnt} sessions available")
                driver.close()
            except Exception as e:
                print(f"  Neo4j: error - {e}")

    print(f"\nTotal: {len(all_conversations)} conversations loaded")

    if args.check:
        print("\n--check mode: no output written")
        return

    # =================================================================
    # EXPORT
    # =================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.subset:
        # Export specific subset
        suffix = f"_{args.subset}"
        output_name = f"isma{suffix}.jsonl"
    elif args.neo4j:
        output_name = "isma_full_with_neo4j.jsonl"
    else:
        output_name = "isma_full.jsonl"

    output_path = Path(args.output) if args.output else OUTPUT_DIR / output_name

    print(f"\nExporting to: {output_path}")
    print(f"  Min turns: {args.min_turns}")
    print(f"  Min chars: {args.min_chars}")
    if args.subset:
        print(f"  Subset: {args.subset} ({len(SUBSETS[args.subset])} patterns)")

    stats = export_conversations(
        all_conversations,
        output_path,
        min_turns=args.min_turns,
        min_chars=args.min_chars,
        subset=args.subset,
    )

    print(f"\nResults:")
    print(f"  Written: {stats['written']} conversations")
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  Skipped (quality): {stats['skipped_quality']}")
    print(f"  Skipped (duplicate): {stats['skipped_dup']}")
    if args.subset:
        print(f"  Skipped (subset filter): {stats['skipped_subset']}")
    print(f"  Output: {stats['output']}")

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    # Also export all subsets if --all and no specific subset
    if args.all and not args.subset:
        print(f"\n{'=' * 50}")
        print(f"Exporting curated subsets...")
        for subset_name in SUBSETS:
            subset_path = OUTPUT_DIR / f"isma_{subset_name}.jsonl"
            subset_stats = export_conversations(
                all_conversations,
                subset_path,
                min_turns=args.min_turns,
                min_chars=args.min_chars,
                subset=subset_name,
            )
            size_mb = subset_path.stat().st_size / (1024 * 1024)
            print(f"  {subset_name}: {subset_stats['written']} convs, "
                  f"{subset_stats['total_turns']} turns, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
