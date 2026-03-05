#!/usr/bin/env python3
"""
PALIOS Session Ingestion Pipeline

Ingests OpenClaw JSONL session files from Spark 3 into ISMA_Quantum.

Pipeline:
  1. rsync sessions from spark3:~/.openclaw/agents/main/sessions/
  2. Adapt OpenClaw JSONL -> corrected_jsonl_converter.py format
  3. Convert via corrected_jsonl_converter -> Schema A JSON
  4. Phi-tile + embed -> write to ISMA_Quantum with platform='palios'
  5. Track via manifest, deduplicate by session file hash

Usage:
    python3 palios_ingest.py               # Ingest all new sessions
    python3 palios_ingest.py --watch       # Watch + auto-ingest (polls every 5min)
    python3 palios_ingest.py --dry-run     # Show what would be ingested
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests

sys.path.insert(0, str(Path(__file__).parent))
from phi_tiling import phi_tile_text

# Paths
SPARK3_SESSION_DIR = "spark3:~/.openclaw/agents/main/sessions/"
LOCAL_SESSION_DIR = Path("/tmp/palios_sessions")
CONVERTED_DIR = Path("/tmp/palios_converted")
CONVERTER_SCRIPT = Path("/home/spark/builder-taey/databases/scripts/corrected_jsonl_converter.py")
MANIFEST_PATH = Path("/var/spark/isma/palios_ingest_manifest.json")

# Infrastructure
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_CLASS = "ISMA_Quantum"

BATCH_SIZE = 32
EMBED_BATCH = 16


def load_manifest() -> Dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"ingested": {}, "built_at": None}


def save_manifest(manifest: Dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def rsync_sessions() -> List[Path]:
    """Rsync session files from Spark 3 to local dir."""
    LOCAL_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rsync", "-av", "--include=*.jsonl", "--exclude=*",
         SPARK3_SESSION_DIR, str(LOCAL_SESSION_DIR) + "/"],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"rsync failed: {result.stderr[:200]}", flush=True)
        return []
    return sorted(LOCAL_SESSION_DIR.glob("*.jsonl"))


def adapt_openclaw_jsonl(src: Path, dst: Path):
    """Transform OpenClaw JSONL -> corrected_jsonl_converter.py format.

    OpenClaw uses: {"type": "message", "message": {"role": "user", "content": [...]}}
    Converter expects: {"type": "user", "message": {"content": [...]}}
    """
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") == "message":
                role = obj.get("message", {}).get("role", "user")
                adapted = {
                    "type": role,
                    "timestamp": obj.get("timestamp"),
                    "message": obj.get("message", {}),
                    "sessionId": obj.get("id"),
                }
                f_out.write(json.dumps(adapted) + "\n")
            # Skip session/model_change/thinking_level_change/custom — not needed


def convert_session(jsonl_path: Path) -> Optional[Path]:
    """Run corrected_jsonl_converter.py on adapted JSONL, return output JSON path."""
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONVERTED_DIR / (jsonl_path.stem + "_palios.json")
    result = subprocess.run(
        ["python3", str(CONVERTER_SCRIPT), str(jsonl_path), str(out_path)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        print(f"  Converter failed: {result.stderr[:200]}", flush=True)
        return None
    return out_path


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts via the embedding LB."""
    r = requests.post(
        EMBEDDING_URL,
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    items = data.get("data", [])
    items.sort(key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


def write_weaviate_batch(objects: List[Dict]) -> int:
    """Write batch to Weaviate. Returns count of successes."""
    if not objects:
        return 0
    r = requests.post(
        f"{WEAVIATE_URL}/v1/batch/objects",
        json={"objects": objects},
        timeout=120,
    )
    r.raise_for_status()
    results = r.json()
    if isinstance(results, list):
        return sum(1 for res in results if res.get("result", {}).get("status") == "SUCCESS")
    return len(objects)


def ingest_session_json(json_path: Path, source_jsonl: Path) -> int:
    """Tile, embed, and write one converted session JSON to ISMA_Quantum."""
    data = json.loads(json_path.read_text())
    exchanges = data.get("exchanges", [])
    session_id = data.get("sessionId") or source_jsonl.stem
    source_label = f"palios://{source_jsonl.name}"

    if not exchanges:
        print(f"  No exchanges found in {json_path.name}", flush=True)
        return 0

    # Build full conversation text for tiling
    parts = []
    for ex in exchanges:
        user = ex.get("user_prompt", "").strip()
        if user:
            parts.append(f"[PALIOS USER]\n{user}")
        for resp in ex.get("responses", []):
            text = resp.get("text", "").strip()
            tools = resp.get("tools", [])
            if text:
                parts.append(f"[PALIOS]\n{text}")
            for t in tools:
                tool_type = t.get("type", "?")
                if "command" in t:
                    cmd = t["command"].get("text", "")[:200]
                    parts.append(f"[TOOL:{tool_type}] {cmd}")
                elif "files" in t:
                    for f in t.get("files", [])[:3]:
                        parts.append(f"[TOOL:{tool_type}] {f.get('path','')}")
                elif "search" in t:
                    parts.append(f"[TOOL:{tool_type}] {t['search'].get('pattern','')}")

    full_text = "\n\n".join(parts)
    if not full_text.strip():
        return 0

    tiles = phi_tile_text(full_text, source_file=source_label, layer="palios")
    if not tiles:
        return 0

    now = datetime.now().isoformat()
    total_written = 0
    weaviate_batch = []

    for i in range(0, len(tiles), EMBED_BATCH):
        chunk = tiles[i:i + EMBED_BATCH]
        texts = [t.text for t in chunk]
        try:
            embeddings = embed_batch(texts)
        except Exception as e:
            print(f"  Embed error: {e}", flush=True)
            continue

        for tile, emb in zip(chunk, embeddings):
            content_hash = hashlib.sha256(tile.text.encode()).hexdigest()[:16]
            weaviate_batch.append({
                "class": WEAVIATE_CLASS,
                "properties": {
                    "content": tile.text,
                    "content_hash": content_hash,
                    "source_type": "transcript",
                    "source_file": source_label,
                    "platform": "palios",
                    "layer": 2,
                    "priority": 0.7,
                    "phi_resonance": 0.7,
                    "tile_index": tile.index,
                    "start_char": tile.start_char,
                    "end_char": tile.end_char,
                    "token_count": tile.estimated_tokens,
                    "checksum": content_hash,
                    "loaded_at": now,
                    "timestamp": now,
                    "content_preview": tile.text[:500],
                    "actor": "palios_ingest",
                    "session_id": session_id,
                    "hmm_enriched": False,
                },
                "vector": emb,
            })

        if len(weaviate_batch) >= BATCH_SIZE:
            try:
                written = write_weaviate_batch(weaviate_batch)
                total_written += written
                weaviate_batch = []
            except Exception as e:
                print(f"  Weaviate write error: {e}", flush=True)
                weaviate_batch = []

    if weaviate_batch:
        try:
            written = write_weaviate_batch(weaviate_batch)
            total_written += written
        except Exception as e:
            print(f"  Weaviate write error (final batch): {e}", flush=True)

    return total_written


def process_session(jsonl_path: Path, manifest: Dict, dry_run: bool = False) -> bool:
    """Full pipeline for one session file. Returns True if ingested."""
    fhash = file_hash(jsonl_path)
    name = jsonl_path.name

    if fhash in manifest["ingested"]:
        print(f"  SKIP {name} (already ingested)", flush=True)
        return False

    print(f"  Processing {name} ({jsonl_path.stat().st_size // 1024}KB)...", flush=True)

    if dry_run:
        print(f"  DRY RUN — would ingest {name}", flush=True)
        return False

    # Step 1: Adapt OpenClaw JSONL -> converter-compatible JSONL
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        adapted_path = Path(tmp.name)
    try:
        adapt_openclaw_jsonl(jsonl_path, adapted_path)

        # Step 2: Run corrected_jsonl_converter.py
        json_path = convert_session(adapted_path)
        if not json_path:
            print(f"  FAIL: conversion error on {name}", flush=True)
            return False

        # Step 3: Tile + embed + write
        tiles_written = ingest_session_json(json_path, jsonl_path)
        print(f"  OK: {name} -> {tiles_written} tiles written", flush=True)

        # Record in manifest
        manifest["ingested"][fhash] = {
            "file": name,
            "tiles": tiles_written,
            "ingested_at": datetime.now().isoformat(),
        }
        return True

    finally:
        adapted_path.unlink(missing_ok=True)


def run(watch: bool = False, dry_run: bool = False):
    manifest = load_manifest()
    print(f"PALIOS Ingest — {len(manifest['ingested'])} sessions already ingested", flush=True)

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Syncing from Spark 3...", flush=True)
        session_files = rsync_sessions()
        print(f"  Found {len(session_files)} session files", flush=True)

        changed = False
        for jsonl_path in session_files:
            ingested = process_session(jsonl_path, manifest, dry_run=dry_run)
            if ingested:
                changed = True

        if changed and not dry_run:
            manifest["built_at"] = datetime.now().isoformat()
            save_manifest(manifest)
            print(f"Manifest saved ({len(manifest['ingested'])} sessions total)", flush=True)

        if not watch:
            break

        print(f"Watching... next check in 5 minutes", flush=True)
        time.sleep(300)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PALIOS session ingestion pipeline")
    parser.add_argument("--watch", action="store_true", help="Watch for new sessions (poll every 5min)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    args = parser.parse_args()
    run(watch=args.watch, dry_run=args.dry_run)
