#!/usr/bin/env python3
"""
ISMA Nightly Ingestion Pipeline

Runs via cron at 2am daily. Discovers and processes new data from:
1. Spark .claude/projects/ (new Claude Code sessions)
2. Mac .claude/projects/ (rsync + parse)
3. Drop directory (/home/spark/data/incoming/)
4. Neo4j new sessions (since last checkpoint)

Usage:
    python3 nightly_ingest.py          # Full nightly run
    python3 nightly_ingest.py --check  # Preview what would run
    python3 nightly_ingest.py --skip-rsync  # Skip Mac rsync
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPTS_DIR = Path(__file__).parent
STATE_DIR = Path("/var/spark/isma")
STATE_FILE = STATE_DIR / "nightly_state.json"

# Source directories
SPARK_CLAUDE_PROJECTS = Path("/home/spark/.claude/projects")
MAC_STAGING = Path("/home/spark/data/staging/mac_claude_projects")
INCOMING_DIR = Path("/home/spark/data/incoming")
PARSED_DIR = Path("/home/spark/data/transcripts/parsed")
CORPUS_DIR = Path("/home/spark/data/corpus")

# Mac rsync target
MAC_HOST = "jesselarose@10.0.0.12"
MAC_CLAUDE_PROJECTS = "/Users/jesselarose/.claude/projects/"
MAC_HOME = "/Users/jesselarose/"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(STATE_DIR / "nightly.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("nightly")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "last_spark_mtime": 0, "last_neo4j_checkpoint": None}


def save_state(state):
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def run_cmd(cmd, description, check_only=False):
    """Run a shell command with logging."""
    log.info(f"{'[CHECK] ' if check_only else ''}Running: {description}")
    if check_only:
        log.info(f"  Would run: {cmd}")
        return True
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=7200
        )
        if result.returncode != 0:
            log.error(f"  FAILED: {result.stderr[:500]}")
            return False
        if result.stdout:
            # Log last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                log.info(f"  {line}")
        return True
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after 2 hours")
        return False


# =============================================================================
# PHASE 1: New Spark Claude Code sessions
# =============================================================================

def discover_new_spark_sessions(state, check_only=False):
    """Find new .jsonl main sessions on Spark."""
    last_mtime = state.get("last_spark_mtime", 0)
    new_files = []

    for jsonl in SPARK_CLAUDE_PROJECTS.rglob("*.jsonl"):
        # Skip subagent files
        if "/subagents/" in str(jsonl) or jsonl.name.startswith("agent-"):
            continue
        # Check if newer than last run
        mtime = jsonl.stat().st_mtime
        if mtime > last_mtime:
            new_files.append(jsonl)

    log.info(f"Phase 1: Found {len(new_files)} new Spark Claude Code sessions")

    if new_files and not check_only:
        # Parse each new file
        for f in new_files:
            run_cmd(
                f"cd {SCRIPTS_DIR} && python3 parse_raw_exports.py "
                f"--input '{f}' --output '{PARSED_DIR}'",
                f"Parse {f.name}"
            )
        # Update mtime checkpoint
        state["last_spark_mtime"] = max(f.stat().st_mtime for f in new_files)

    return len(new_files)


# =============================================================================
# PHASE 2: Mac rsync + parse
# =============================================================================

def rsync_and_parse_mac(state, check_only=False, skip_rsync=False):
    """rsync Mac .claude/projects/ and parse new sessions."""
    if skip_rsync:
        log.info("Phase 2: Skipping Mac rsync (--skip-rsync)")
        return 0

    MAC_STAGING.mkdir(parents=True, exist_ok=True)

    # rsync .jsonl files from Mac
    success = run_cmd(
        f"rsync -avz --include='*.jsonl' --include='*/' --exclude='*' "
        f"{MAC_HOST}:{MAC_CLAUDE_PROJECTS} {MAC_STAGING}/",
        "rsync Mac .claude/projects/",
        check_only=check_only
    )
    if not success and not check_only:
        log.warning("Mac rsync failed - Mac may be offline")
        return 0

    if check_only:
        return 0

    # Find main sessions (not subagents)
    new_files = []
    for jsonl in MAC_STAGING.rglob("*.jsonl"):
        if "/subagents/" in str(jsonl) or jsonl.name.startswith("agent-"):
            continue
        # Check if already parsed
        session_id = jsonl.stem[:16]
        parsed = PARSED_DIR / "claude_code" / f"{session_id}.json"
        if not parsed.exists():
            new_files.append(jsonl)

    log.info(f"Phase 2: Found {len(new_files)} unparsed Mac sessions")

    for f in new_files:
        run_cmd(
            f"cd {SCRIPTS_DIR} && python3 parse_raw_exports.py "
            f"--input '{f}' --output '{PARSED_DIR}'",
            f"Parse Mac {f.name}"
        )

    return len(new_files)


# =============================================================================
# PHASE 3: Incoming drop directory
# =============================================================================

def process_incoming(check_only=False):
    """Process files dropped in /home/spark/data/incoming/."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    (INCOMING_DIR / "transcripts").mkdir(exist_ok=True)
    (INCOMING_DIR / "corpus").mkdir(exist_ok=True)

    # Count files
    transcript_files = list((INCOMING_DIR / "transcripts").rglob("*"))
    transcript_files = [f for f in transcript_files if f.is_file()]
    corpus_files = list((INCOMING_DIR / "corpus").rglob("*"))
    corpus_files = [f for f in corpus_files if f.is_file()]

    log.info(f"Phase 3: {len(transcript_files)} transcript files, "
             f"{len(corpus_files)} corpus files in incoming/")

    if transcript_files and not check_only:
        # Parse transcript files
        for f in transcript_files:
            if f.suffix in ('.jsonl', '.json'):
                run_cmd(
                    f"cd {SCRIPTS_DIR} && python3 parse_raw_exports.py "
                    f"--input '{f}' --output '{PARSED_DIR}'",
                    f"Parse incoming {f.name}"
                )
            elif f.suffix in ('.md', '.txt'):
                # Copy to corpus for processing
                dest = CORPUS_DIR / "spark_loose" / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(f, dest)
                log.info(f"  Moved {f.name} to corpus/spark_loose/")

        # Move processed files to archive
        archive = INCOMING_DIR / "processed" / datetime.now().strftime("%Y%m%d")
        archive.mkdir(parents=True, exist_ok=True)
        for f in transcript_files:
            f.rename(archive / f.name)

    return len(transcript_files) + len(corpus_files)


# =============================================================================
# PHASE 4: Rebuild dedup manifest + embed via unified pipeline
# =============================================================================

def rebuild_manifest_and_embed(check_only=False):
    """Rebuild dedup manifest then run unified_ingest.py --incremental.

    The unified pipeline writes to BOTH Weaviate and Neo4j, replacing the
    old split of process_transcripts.py (Weaviate only) + reprocess_neo4j.py.
    """
    # Rebuild manifest to pick up new files
    success = run_cmd(
        f"cd {SCRIPTS_DIR} && python3 build_dedup_manifest.py",
        "Rebuild dedup manifest",
        check_only=check_only
    )
    if not success and not check_only:
        log.error("Manifest rebuild failed - skipping embed")
        return False

    # Run unified ingest (incremental = idempotent upsert)
    return run_cmd(
        f"cd {SCRIPTS_DIR} && python3 -u unified_ingest.py --incremental",
        "Unified ingest (incremental, dual-write Weaviate+Neo4j)",
        check_only=check_only
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ISMA Nightly Ingestion")
    parser.add_argument("--check", action="store_true", help="Preview only")
    parser.add_argument("--skip-rsync", action="store_true", help="Skip Mac rsync")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"ISMA Nightly Ingestion - {datetime.now().isoformat()}")
    log.info("=" * 60)

    state = load_state()
    if state.get("last_run"):
        log.info(f"Last run: {state['last_run']}")

    start = time.time()

    # Phase 1: Spark Claude Code
    n_spark = discover_new_spark_sessions(state, check_only=args.check)

    # Phase 2: Mac rsync + parse
    n_mac = rsync_and_parse_mac(state, check_only=args.check, skip_rsync=args.skip_rsync)

    # Phase 3: Incoming files
    n_incoming = process_incoming(check_only=args.check)

    # Phase 4: Rebuild manifest + unified ingest (dual-write Weaviate+Neo4j)
    if (n_spark + n_mac + n_incoming) > 0 or not args.check:
        rebuild_manifest_and_embed(check_only=args.check)

    elapsed = time.time() - start

    if not args.check:
        save_state(state)

    log.info(f"\nComplete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"  Spark sessions: {n_spark}")
    log.info(f"  Mac sessions: {n_mac}")
    log.info(f"  Incoming files: {n_incoming}")


if __name__ == "__main__":
    main()
