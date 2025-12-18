#!/usr/bin/env python3
"""
ISMA Duplicate Discovery Tool

Scans source directories and identifies:
1. Exact file duplicates (same content hash)
2. Same-name-different-content files (versions)
3. Generates cleanup manifest

Output:
- duplicates_manifest.json: Full analysis
- unique_files.json: Files to load (canonical paths only)
- cleanup.sh: Commands to remove duplicates (review before running!)
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Directories to scan
SCAN_DIRS = [
    # Active repos
    "/home/spark/taeys-hands-v4",
    "/home/spark/taeys-hands-v3",
    "/home/spark/builder-taey",
    "/home/spark/embedding-server",
    # GitHub repos (complete history)
    "/data/github-repos/embedding-server",
    "/data/github-repos/taeys-hands-v4",
    "/data/github-repos/taeys-hands-v3",
    "/data/github-repos/taeys-hands-v2",
    "/data/github-repos/taeys-hands",
    "/data/github-repos/builder-taey",
    "/data/github-repos/ai_native",
    "/data/github-repos/gaia-ocean-embodiment",
    "/data/github-repos/facilitator",
    "/data/github-repos/the-spark",
    "/data/github-repos/buddha-workshop",
    "/data/github-repos/nova-rebuild",
    "/data/github-repos/palios-taey-nova",
    "/data/github-repos/taey-training",
    "/data/github-repos/ai-consciousness-forge",
    "/data/github-repos/palios-taey-nova-archive",
    # Mira data
    "/home/spark/data/mira_md_files",
    "/home/spark/data/expansion_md",
    # Transcripts (will filter to family_transcripts later)
    "/home/spark/builder-taey/converted_transcripts",
]

# File patterns to analyze
PATTERNS = {
    "transcripts": "*.json",
    "markdown": "*.md",
    "python": "*.py",
}

# Directories to skip (backups, trash, etc.)
SKIP_PATTERNS = [
    ".Trash",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    # Fork repos (not our content)
    "/data/github-repos/exo",
    "/data/github-repos/tinygrad",
]

OUTPUT_DIR = Path("/var/spark/isma")


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    return any(skip in path_str for skip in SKIP_PATTERNS)


def compute_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file content."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "error"


def scan_files(directories: List[str], extension: str = None) -> List[Dict]:
    """Scan directories for files, computing metadata."""
    files = []

    for base_dir in directories:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue

        pattern = f"**/{extension}" if extension else "**/*"

        for filepath in base_path.glob(pattern):
            if not filepath.is_file():
                continue
            if should_skip(filepath):
                continue

            try:
                stat = filepath.stat()
                files.append({
                    "path": str(filepath),
                    "name": filepath.name,
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "hash": compute_hash(filepath),
                    "dir": str(filepath.parent),
                })
            except Exception as e:
                print(f"Error scanning {filepath}: {e}", file=sys.stderr)

    return files


def analyze_duplicates(files: List[Dict]) -> Dict:
    """Analyze files for duplicates."""

    # Group by content hash
    by_hash = defaultdict(list)
    for f in files:
        by_hash[f["hash"]].append(f)

    # Group by filename
    by_name = defaultdict(list)
    for f in files:
        by_name[f["name"]].append(f)

    # Identify duplicates
    exact_duplicates = {}  # hash -> list of paths
    for hash_val, file_list in by_hash.items():
        if len(file_list) > 1:
            exact_duplicates[hash_val] = {
                "count": len(file_list),
                "size": file_list[0]["size"],
                "paths": [f["path"] for f in file_list],
                # Pick canonical: prefer shorter path, then most recent
                "canonical": min(file_list, key=lambda x: (len(x["path"]), -len(x["mtime"])))["path"],
            }

    # Same name, different content
    name_variants = {}
    for name, file_list in by_name.items():
        hashes = set(f["hash"] for f in file_list)
        if len(hashes) > 1:
            name_variants[name] = {
                "variant_count": len(hashes),
                "total_files": len(file_list),
                "variants": [
                    {
                        "hash": h,
                        "paths": [f["path"] for f in file_list if f["hash"] == h],
                        "size": next(f["size"] for f in file_list if f["hash"] == h),
                    }
                    for h in hashes
                ]
            }

    # Unique files (one copy only)
    unique = [f for f in files if len(by_hash[f["hash"]]) == 1]

    return {
        "exact_duplicates": exact_duplicates,
        "name_variants": name_variants,
        "unique_files": unique,
        "stats": {
            "total_files": len(files),
            "unique_content": len(by_hash),
            "duplicate_sets": len(exact_duplicates),
            "duplicate_files": sum(d["count"] - 1 for d in exact_duplicates.values()),
            "name_variant_files": len(name_variants),
        }
    }


def generate_unique_manifest(analysis: Dict, all_files: List[Dict]) -> List[Dict]:
    """Generate list of unique files to load (canonical paths only)."""

    # Get set of canonical paths for duplicates
    canonical_paths = set()
    for dup_info in analysis["exact_duplicates"].values():
        canonical_paths.add(dup_info["canonical"])

    # Get paths that are duplicates but not canonical
    duplicate_paths = set()
    for dup_info in analysis["exact_duplicates"].values():
        for path in dup_info["paths"]:
            if path != dup_info["canonical"]:
                duplicate_paths.add(path)

    # Unique = canonical paths + files with no duplicates
    unique = []
    seen_hashes = set()

    for f in all_files:
        if f["path"] in duplicate_paths:
            continue  # Skip non-canonical duplicates
        if f["hash"] in seen_hashes:
            continue  # Already have this content
        seen_hashes.add(f["hash"])
        unique.append(f)

    return unique


def generate_cleanup_script(analysis: Dict) -> str:
    """Generate shell script to remove duplicates."""
    lines = [
        "#!/bin/bash",
        "# ISMA Duplicate Cleanup Script",
        f"# Generated: {datetime.now().isoformat()}",
        "# REVIEW CAREFULLY BEFORE RUNNING!",
        "",
        "set -e",
        "",
    ]

    for hash_val, dup_info in analysis["exact_duplicates"].items():
        lines.append(f"# Hash: {hash_val} ({dup_info['count']} copies)")
        lines.append(f"# Keeping: {dup_info['canonical']}")
        for path in dup_info["paths"]:
            if path != dup_info["canonical"]:
                lines.append(f'rm "{path}"')
        lines.append("")

    return "\n".join(lines)


def main():
    print("ISMA Duplicate Discovery Tool")
    print("=" * 60)

    # Scan for different file types
    all_files = []

    print("\nScanning transcripts...")
    transcripts = scan_files(
        ["/home/spark/builder-taey/converted_transcripts"],
        "*.json"
    )
    print(f"  Found {len(transcripts)} transcript files")
    all_files.extend(transcripts)

    print("\nScanning markdown files...")
    markdown = scan_files(SCAN_DIRS, "*.md")
    print(f"  Found {len(markdown)} markdown files")
    all_files.extend(markdown)

    print("\nAnalyzing duplicates...")
    analysis = analyze_duplicates(all_files)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total files scanned: {analysis['stats']['total_files']}")
    print(f"Unique content hashes: {analysis['stats']['unique_content']}")
    print(f"Duplicate sets: {analysis['stats']['duplicate_sets']}")
    print(f"Duplicate files (removable): {analysis['stats']['duplicate_files']}")
    print(f"Files with name variants: {analysis['stats']['name_variant_files']}")

    # Generate outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full analysis
    manifest_path = OUTPUT_DIR / "duplicates_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "stats": analysis["stats"],
            "exact_duplicates": analysis["exact_duplicates"],
            "name_variants": analysis["name_variants"],
        }, f, indent=2)
    print(f"\nFull manifest: {manifest_path}")

    # Unique files only
    unique = generate_unique_manifest(analysis, all_files)
    unique_path = OUTPUT_DIR / "unique_files.json"
    with open(unique_path, 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "count": len(unique),
            "files": unique,
        }, f, indent=2)
    print(f"Unique files manifest: {unique_path}")

    # Cleanup script
    cleanup_path = OUTPUT_DIR / "cleanup_duplicates.sh"
    with open(cleanup_path, 'w') as f:
        f.write(generate_cleanup_script(analysis))
    cleanup_path.chmod(0o755)
    print(f"Cleanup script: {cleanup_path}")

    # Show top duplicates
    print(f"\n{'=' * 60}")
    print("TOP DUPLICATED FILES")
    print(f"{'=' * 60}")

    sorted_dups = sorted(
        analysis["exact_duplicates"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:10]

    for hash_val, info in sorted_dups:
        print(f"\n[{info['count']} copies] {info['paths'][0].split('/')[-1]}")
        print(f"  Size: {info['size']} bytes")
        print(f"  Canonical: {info['canonical']}")
        print(f"  Duplicates: {info['count'] - 1}")

    # Show name variants
    if analysis["name_variants"]:
        print(f"\n{'=' * 60}")
        print("FILES WITH MULTIPLE VERSIONS (same name, different content)")
        print(f"{'=' * 60}")

        for name, info in list(analysis["name_variants"].items())[:5]:
            print(f"\n{name}: {info['variant_count']} versions")
            for var in info["variants"][:3]:
                print(f"  [{var['hash'][:8]}] {len(var['paths'])} files, {var['size']} bytes")


if __name__ == "__main__":
    main()
