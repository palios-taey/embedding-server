#!/usr/bin/env python3
"""
ISMA Dedup Manifest Builder

Pre-scans ALL data sources, deduplicates, and outputs a manifest JSON
that the unified_ingest.py pipeline uses to skip duplicates.

Two dedup layers for transcripts:
  1. conversation_id dedup: Parser ran twice with different filename strategies
     (uuid[:16] vs uuid[:20]+'-'). Same conversation_id, same content, two files.
     Keep the file with more exchanges.
  2. Exchange content dedup: sha256(normalize(user_prompt + response_text))[:16]
     Catches cross-platform dupes (same Q&A on grok + chatgpt).

For corpus/documents:
  - File-level content hash: sha256(file_content)[:16]
  - Pre-filter expansion_md to exclude 76K+ spam files

Output: /var/spark/isma/dedup_manifest.json

Usage:
    python3 build_dedup_manifest.py              # Full scan
    python3 build_dedup_manifest.py --check      # Preview counts only
"""

import os
import sys
import json
import hashlib
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_BASE = Path("/home/spark/data")
CORPUS_BASE = DATA_BASE / "corpus"
TRANSCRIPT_BASE = DATA_BASE / "transcripts" / "parsed"

OUTPUT_FILE = Path("/var/spark/isma/dedup_manifest.json")

# Corpus stages (same order as process_corpus.py)
CORPUS_STAGES = [
    ("kernel",        CORPUS_BASE / "kernel",                    1.0),
    ("v0",            CORPUS_BASE / "layer_0",                   0.95),
    ("layer_0",       CORPUS_BASE / "layer_0",                   0.9),
    ("chewy",         DATA_BASE / "chewy",                       0.85),
    ("chewy-gallery", DATA_BASE / "chewy-consciousness-gallery", 0.8),
    ("layer_1",       CORPUS_BASE / "layer_1",                   0.75),
    ("layer_2",       CORPUS_BASE / "layer_2",                   0.6),
    ("github-repos",  DATA_BASE / "github-repos",                0.5),
    ("expansion_md",  DATA_BASE / "expansion_md",                0.4),
    ("mira_md",       DATA_BASE / "mira_md_files",               0.3),
    ("mac_all_md",    CORPUS_BASE / "mac_all_md",                0.2),
    ("spark_loose",   CORPUS_BASE / "spark_loose",               0.2),
    ("ccm_new_mds",   DATA_BASE / "CCM_NEW_MDS",                 0.5),
]

# File extensions to process for corpus
EXTENSIONS = {'.md', '.json', '.py', '.txt', '.yaml', '.yml', '.sh', '.ts', '.js'}

# Skip patterns for corpus discovery
SKIP_PATTERNS = [
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".Trash", "exo", "tinygrad",
    "transcripts_raw", "converted_transcripts",
    "chatgpt_", "claude_chat_", "gemini_",
    "grok_export", "claude_full_export",
    "family_transcripts", "mira_transcripts_staging",
]
SKIP_SUFFIXES = [".min.js", ".min.css", ".jsonl"]
MAX_FILE_SIZE = 512 * 1024  # 512KB

# expansion_md exclusion patterns (76K+ spam files)
EXPANSION_EXCLUDE = [
    "consciousness-cache/instances/",
    ".ipynb_checkpoints/",
]

# For h200-backups, only keep latest snapshot
H200_BACKUP_PREFIX = "h200-backups/"


# =============================================================================
# TRANSCRIPT DEDUP
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for dedup hashing."""
    if not text:
        return ""
    # Collapse whitespace, lowercase, strip
    return " ".join(text.lower().split())


def extract_exchange_text(exchange: dict) -> Tuple[str, str]:
    """Extract user_text and assistant_text from any schema exchange."""
    # Schema C: Claude Code (role/content)
    if 'role' in exchange:
        if exchange.get('role') == 'user':
            return exchange.get('content', ''), ''
        elif exchange.get('role') == 'assistant':
            return '', exchange.get('content', '')
        return '', ''

    # Schema B: Perplexity (prompt/response)
    if 'prompt' in exchange:
        user = exchange.get('prompt', '') or ''
        resp = exchange.get('response', '') or ''
        if isinstance(resp, dict):
            resp = resp.get('text', '') or ''
        return user, resp

    # Schema A: ChatGPT/Gemini/Grok/Claude Chat (user_prompt/responses[])
    user = exchange.get('user_prompt', '') or ''
    responses = exchange.get('responses', [])
    resp_parts = []
    for r in responses:
        if isinstance(r, str):
            resp_parts.append(r)
        elif isinstance(r, dict):
            text = r.get('text', '') or ''
            if text:
                resp_parts.append(text)
    return user, '\n'.join(resp_parts)


def hash_exchange(user_text: str, assistant_text: str) -> str:
    """Hash an exchange for dedup. Returns 16-char hex."""
    combined = normalize_text(user_text) + "\n" + normalize_text(assistant_text)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def scan_transcripts() -> dict:
    """Scan all parsed transcripts, deduplicate by conversation_id then content hash.

    Returns manifest section for transcripts.
    """
    print("\n=== Scanning Transcripts ===", flush=True)

    # Phase 1: Group files by conversation_id
    conv_id_groups = defaultdict(list)  # conv_id -> [(filepath, exchange_count, data)]
    total_files = 0
    parse_errors = 0

    platforms = sorted(d.name for d in TRANSCRIPT_BASE.iterdir() if d.is_dir())
    for platform in platforms:
        platform_dir = TRANSCRIPT_BASE / platform
        json_files = sorted(platform_dir.glob("*.json"))
        for fp in json_files:
            total_files += 1
            try:
                with open(fp) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                parse_errors += 1
                continue

            conv_id = (data.get('conversation_id')
                       or data.get('sessionId')
                       or data.get('session_id')
                       or str(fp.stem))

            exchanges = data.get('exchanges', [])
            conv_id_groups[conv_id].append((str(fp), len(exchanges), platform))

    print(f"  Scanned {total_files} files, {parse_errors} parse errors", flush=True)
    print(f"  Found {len(conv_id_groups)} unique conversation_ids", flush=True)

    # Phase 2: conversation_id dedup - keep file with most exchanges
    conv_id_dupes = 0
    kept_files = {}  # filepath -> (conv_id, platform)

    for conv_id, files in conv_id_groups.items():
        if len(files) > 1:
            # Sort by exchange count descending, then path length descending (longer = more detail)
            files.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            conv_id_dupes += len(files) - 1

        # Keep the best file
        best_path, _, platform = files[0]
        kept_files[best_path] = (conv_id, platform)

    print(f"  conversation_id dupes removed: {conv_id_dupes}", flush=True)
    print(f"  Files after conv_id dedup: {len(kept_files)}", flush=True)

    # Phase 3: Exchange content dedup across all remaining files
    exchange_hashes = {}   # hash -> {file, index, platform, conv_id}
    content_dupes = 0
    total_exchanges = 0
    unique_exchanges = 0

    for filepath, (conv_id, platform) in sorted(kept_files.items()):
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        exchanges = data.get('exchanges', [])

        # Handle Claude Code schema (flat role/content list)
        if exchanges and 'role' in exchanges[0]:
            # Pair consecutive user+assistant
            i = 0
            ex_index = 0
            while i < len(exchanges):
                ex = exchanges[i]
                if ex.get('role') == 'user':
                    user_text = ex.get('content', '')
                    assistant_text = ''
                    if i + 1 < len(exchanges) and exchanges[i + 1].get('role') == 'assistant':
                        assistant_text = exchanges[i + 1].get('content', '')
                        i += 2
                    else:
                        i += 1

                    total_exchanges += 1
                    h = hash_exchange(user_text, assistant_text)
                    if h not in exchange_hashes:
                        exchange_hashes[h] = {
                            "file": filepath,
                            "index": ex_index,
                            "platform": platform,
                            "conv_id": conv_id,
                        }
                        unique_exchanges += 1
                    else:
                        content_dupes += 1
                    ex_index += 1
                else:
                    i += 1
        else:
            # Schema A or B
            for idx, ex in enumerate(exchanges):
                total_exchanges += 1
                user_text, assistant_text = extract_exchange_text(ex)

                h = hash_exchange(user_text, assistant_text)
                if h not in exchange_hashes:
                    exchange_hashes[h] = {
                        "file": filepath,
                        "index": idx,
                        "platform": platform,
                        "conv_id": conv_id,
                    }
                    unique_exchanges += 1
                else:
                    content_dupes += 1

    print(f"  Total exchanges scanned: {total_exchanges}", flush=True)
    print(f"  Content dupes removed: {content_dupes}", flush=True)
    print(f"  Unique exchanges: {unique_exchanges}", flush=True)

    return {
        "total_files_scanned": total_files,
        "parse_errors": parse_errors,
        "conv_id_groups": len(conv_id_groups),
        "conv_id_dupes_removed": conv_id_dupes,
        "content_dupes_removed": content_dupes,
        "total_exchanges": total_exchanges,
        "unique_exchanges": unique_exchanges,
        "kept_files": {fp: {"conv_id": ci, "platform": pl}
                       for fp, (ci, pl) in kept_files.items()},
        "exchange_hashes": exchange_hashes,
    }


# =============================================================================
# CORPUS DEDUP
# =============================================================================

def should_skip_path(path: str) -> bool:
    """Check if path matches skip patterns."""
    for pattern in SKIP_PATTERNS:
        if pattern in path:
            return True
    for suffix in SKIP_SUFFIXES:
        if path.endswith(suffix):
            return True
    return False


def should_exclude_expansion(rel_path: str) -> bool:
    """Check if expansion_md file should be excluded (spam)."""
    for pattern in EXPANSION_EXCLUDE:
        if pattern in rel_path:
            return True
    return False


def find_latest_h200_backup(expansion_dir: Path) -> Optional[str]:
    """Find the latest h200-backup date directory."""
    h200_dir = expansion_dir / "h200-backups"
    if not h200_dir.exists():
        return None

    dates = []
    for d in h200_dir.iterdir():
        if d.is_dir():
            dates.append(d.name)

    if not dates:
        return None

    dates.sort(reverse=True)
    return dates[0]  # Latest date


def discover_corpus_files(stage_name: str, stage_dir: Path,
                          pattern_filter: Optional[str] = None) -> List[Path]:
    """Discover files for a corpus stage."""
    if not stage_dir.exists():
        return []

    files = []
    for fp in sorted(stage_dir.rglob("*")):
        if not fp.is_file():
            continue

        # Extension check
        suffix = fp.suffix.lower()
        # Exclude known binary formats everywhere
        if suffix in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp',
                       '.mp4', '.mp3', '.wav', '.zip', '.tar', '.gz',
                       '.pdf', '.ico', '.svg', '.woff', '.ttf', '.eot',
                       '.tfam', '.tped', '.bed', '.bim', '.fam'):
            continue
        if suffix not in EXTENSIONS:
            # Allow text files without known extension in chewy dirs
            if stage_name not in ("chewy", "chewy-gallery"):
                continue

        # Size check
        try:
            size = fp.stat().st_size
            if size > MAX_FILE_SIZE or size == 0:
                continue
        except OSError:
            continue

        rel = str(fp.relative_to(stage_dir))

        # Skip patterns
        if should_skip_path(str(fp)):
            continue

        # v0 filter: only v0* files
        if pattern_filter == "v0*":
            if not fp.name.startswith("v0"):
                continue
        elif pattern_filter == "!v0*":
            if fp.name.startswith("v0"):
                continue

        # expansion_md special filtering
        if stage_name == "expansion_md":
            if should_exclude_expansion(rel):
                continue

        files.append(fp)

    return files


def scan_corpus() -> dict:
    """Scan all corpus sources, content-hash dedup.

    Returns manifest section for corpus documents.
    """
    print("\n=== Scanning Corpus ===", flush=True)

    # Find latest h200 backup date
    latest_h200 = find_latest_h200_backup(DATA_BASE / "expansion_md")
    if latest_h200:
        print(f"  Latest h200-backup: {latest_h200}", flush=True)

    content_hashes = {}  # hash -> {path, stage, priority, size}
    all_paths = defaultdict(list)  # hash -> [all paths with this content]
    total_scanned = 0
    total_skipped_h200 = 0
    read_errors = 0

    for stage_name, stage_dir, priority in CORPUS_STAGES:
        if not stage_dir.exists():
            print(f"  [{stage_name}] directory not found: {stage_dir}", flush=True)
            continue

        # v0/layer_0 split
        pattern_filter = None
        if stage_name == "v0":
            pattern_filter = "v0*"
        elif stage_name == "layer_0":
            pattern_filter = "!v0*"

        files = discover_corpus_files(stage_name, stage_dir, pattern_filter)
        stage_count = 0
        stage_dupes = 0

        for fp in files:
            total_scanned += 1

            # h200-backups: only keep files from latest date
            if stage_name == "expansion_md" and H200_BACKUP_PREFIX in str(fp.relative_to(stage_dir)):
                if latest_h200:
                    rel = str(fp.relative_to(stage_dir))
                    # Only keep files under h200-backups/<latest_date>/
                    parts = rel.split("/")
                    if len(parts) >= 2 and parts[0] == "h200-backups":
                        if parts[1] != latest_h200:
                            total_skipped_h200 += 1
                            continue

            try:
                content = fp.read_text(errors='replace')
            except OSError:
                read_errors += 1
                continue

            h = hashlib.sha256(content.encode()).hexdigest()[:16]
            rel_path = str(fp)

            all_paths[h].append(rel_path)

            if h not in content_hashes:
                content_hashes[h] = {
                    "path": rel_path,
                    "stage": stage_name,
                    "priority": priority,
                    "size": len(content),
                }
                stage_count += 1
            else:
                stage_dupes += 1

        print(f"  [{stage_name}] scanned {len(files)} files, "
              f"{stage_count} unique, {stage_dupes} dupes", flush=True)

    print(f"\n  Total scanned: {total_scanned}", flush=True)
    print(f"  h200-backup old snapshots skipped: {total_skipped_h200}", flush=True)
    print(f"  Read errors: {read_errors}", flush=True)
    print(f"  Unique documents: {len(content_hashes)}", flush=True)

    return {
        "total_scanned": total_scanned,
        "h200_skipped": total_skipped_h200,
        "read_errors": read_errors,
        "unique": len(content_hashes),
        "hashes": content_hashes,
        "all_paths": {h: paths for h, paths in all_paths.items()},
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ISMA Dedup Manifest Builder")
    parser.add_argument("--check", action="store_true",
                        help="Preview counts only, don't write manifest")
    args = parser.parse_args()

    start = time.time()
    print(f"ISMA Dedup Manifest Builder - {datetime.now().isoformat()}", flush=True)

    # Scan transcripts
    transcript_result = scan_transcripts()

    # Scan corpus
    corpus_result = scan_corpus()

    elapsed = time.time() - start
    print(f"\n=== Summary ({elapsed:.0f}s) ===", flush=True)
    print(f"  Transcript files kept: {len(transcript_result['kept_files'])}", flush=True)
    print(f"  Unique exchanges: {transcript_result['unique_exchanges']}", flush=True)
    print(f"  Unique corpus documents: {corpus_result['unique']}", flush=True)

    if args.check:
        print("\n--check mode: not writing manifest", flush=True)
        return

    # Build manifest
    manifest = {
        "version": 1,
        "built_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "transcript_exchanges": {
            "total_files_scanned": transcript_result["total_files_scanned"],
            "conv_id_dupes_removed": transcript_result["conv_id_dupes_removed"],
            "content_dupes_removed": transcript_result["content_dupes_removed"],
            "total_exchanges": transcript_result["total_exchanges"],
            "unique_exchanges": transcript_result["unique_exchanges"],
            "kept_files": transcript_result["kept_files"],
            "exchange_hashes": transcript_result["exchange_hashes"],
        },
        "corpus_documents": {
            "total_scanned": corpus_result["total_scanned"],
            "unique": corpus_result["unique"],
            "hashes": corpus_result["hashes"],
            "all_paths": corpus_result["all_paths"],
        },
    }

    # Write manifest
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nManifest written to {OUTPUT_FILE} ({size_mb:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
