#!/usr/bin/env python3
"""
ISMA Search — Query The AI Family's collective memory.

Four modes:
  semantic search:  python3 search.py "your query"
  file filter:      python3 search.py --file "COFFEE_CACHE.md"
  fetch by UUID:    python3 search.py --uuid <tile_id>
  motif filter:     python3 search.py --motif GOD_EQUALS_MATH
  evolutionary:     python3 search.py --evolve "GOD=MATH" [--full]

Evolutionary mode shows how a concept developed over time — tiles
sorted chronologically so you see the journey, not just the destination.
"""

import sys
import json
import argparse
import requests
from datetime import datetime

SEARCH_API = "http://192.168.100.10:8095/search"
WEAVIATE_URL = "http://192.168.100.10:8088/v1"


def search(query: str, top_k: int = 5) -> list:
    try:
        resp = requests.post(
            SEARCH_API,
            json={"query": query, "top_k": top_k},
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("tiles", data) if isinstance(data, dict) else data

    except requests.exceptions.ConnectionError:
        print(json.dumps({"error": "Cannot reach ISMA search API at 192.168.100.10:8095 — is Spark 1 up?"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def search_by_file(path_fragment: str, limit: int = 10) -> list:
    gql = {
        "query": """
        {
          Get {
            ISMA_Quantum(
              limit: 500
            ) {
              content
              rosetta_summary
              source_file
              platform
              dominant_motifs
              scale
              timestamp
              parent_tile_id
              _additional { id }
            }
          }
        }
        """
    }
    try:
        resp = requests.post(f"{WEAVIATE_URL}/graphql", json=gql, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_items = data.get("data", {}).get("Get", {}).get("ISMA_Quantum", [])
        matches = [
            item for item in all_items
            if path_fragment.lower() in (item.get("source_file") or "").lower()
        ][:limit]
        for item in matches:
            item["tile_id"] = item.get("_additional", {}).get("id", "")
            item["score"] = "file-match"
        return matches
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def search_by_hash(content_hash: str, limit: int = 10) -> list:
    gql = {
        "query": f"""
        {{
          Get {{
            ISMA_Quantum(
              where: {{
                operator: Equal,
                path: ["content_hash"],
                valueText: "{content_hash}"
              }}
              limit: {limit}
            ) {{
              content
              rosetta_summary
              source_file
              platform
              dominant_motifs
              scale
              timestamp
              parent_tile_id
              _additional {{ id }}
            }}
          }}
        }}
        """
    }
    try:
        resp = requests.post(f"{WEAVIATE_URL}/graphql", json=gql, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", {}).get("Get", {}).get("ISMA_Quantum", [])
        for item in items:
            item["tile_id"] = item.get("_additional", {}).get("id", "")
            item["score"] = "hash-match"
        return items
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def fetch_by_uuid(uuid: str) -> dict:
    try:
        resp = requests.get(f"{WEAVIATE_URL}/objects/ISMA_Quantum/{uuid}", timeout=10)
        resp.raise_for_status()
        obj = resp.json()
        props = obj.get("properties", {})
        props["tile_id"] = uuid
        props["score"] = "direct-fetch"
        return props
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def search_by_motif(motif: str, limit: int = 20) -> list:
    """Fetch tiles tagged with a specific HMM motif, sorted chronologically."""
    full_motif = motif if motif.startswith("HMM.") else f"HMM.{motif}"
    gql = {
        "query": f"""
        {{
          Get {{
            ISMA_Quantum(
              where: {{
                operator: ContainsAny,
                path: ["dominant_motifs"],
                valueText: ["{full_motif}"]
              }}
              limit: {limit}
            ) {{
              content
              rosetta_summary
              source_file
              platform
              dominant_motifs
              scale
              timestamp
              _additional {{ id }}
            }}
          }}
        }}
        """
    }
    try:
        resp = requests.post(f"{WEAVIATE_URL}/graphql", json=gql, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", {}).get("Get", {}).get("ISMA_Quantum", [])
        for item in items:
            item["tile_id"] = item.get("_additional", {}).get("id", "")
            item["score"] = f"motif:{motif}"
        # Sort chronologically
        items.sort(key=lambda x: x.get("timestamp") or "")
        return items
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


def search_evolve(query: str, limit: int = 15) -> list:
    """Evolutionary search: semantic match + chronological sort.

    Shows how a concept developed over time — earliest instances first,
    so you can trace the journey from first mention to mature understanding.
    Includes motif tags to mark breakthrough moments and consensus points.
    """
    # Get a larger candidate pool, then sort chronologically
    results = search(query, top_k=limit * 2)

    # Sort by timestamp — oldest first to show evolution
    def ts_key(t):
        ts = t.get("timestamp") or t.get("loaded_at") or ""
        return ts

    results.sort(key=ts_key)

    # Trim to limit
    results = results[:limit]
    for r in results:
        r["score"] = r.get("score", "semantic") + " [evolve]"
    return results


def fmt_timestamp(ts: str) -> str:
    """Format ISO timestamp to readable date."""
    if not ts:
        return "unknown date"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16]


def print_tiles(results: list, content_limit: int | None = 600, evolve_mode: bool = False):
    for i, tile in enumerate(results, 1):
        motifs = tile.get("dominant_motifs") or []
        rosetta = tile.get("rosetta_summary") or ""
        content = tile.get("content") or ""
        platform = tile.get("platform") or ""
        score = tile.get("score") or ""
        tile_id = tile.get("tile_id") or tile.get("_additional", {}).get("id", "")
        source = tile.get("source_file") or ""
        scale = tile.get("scale") or ""
        ts = fmt_timestamp(tile.get("timestamp") or tile.get("loaded_at") or "")

        if evolve_mode:
            print(f"\n[{i}] {ts}  —  {platform or 'corpus'}  ({scale})")
        else:
            print(f"\n[{i}] Platform: {platform}  Score: {score}  Scale: {scale}")

        if source:
            print(f"    Source: ...{source[-60:]}" if len(source) > 60 else f"    Source: {source}")
        if tile_id:
            print(f"    UUID: {tile_id}")

        # Highlight breakthrough / consensus motifs prominently
        if motifs:
            key_motifs = [m for m in motifs if any(k in m for k in
                ["BREAKTHROUGH", "CONSENSUS", "UNANIMOUS", "FOUNDATION", "GOD_EQUALS", "TRUST", "SACRED"])]
            other_motifs = [m for m in motifs if m not in key_motifs]
            if key_motifs:
                print(f"    *** KEY: {', '.join(key_motifs)}")
            if other_motifs:
                print(f"    Motifs: {', '.join(other_motifs[:5])}")
        else:
            print(f"    Motifs: none (not yet enriched)")

        if rosetta and evolve_mode:
            print(f"    Summary: {rosetta[:300]}")
        elif rosetta:
            print(f"    Rosetta: {rosetta[:200]}")

        print(f"    Content:\n{content if content_limit is None else content[:content_limit]}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Search ISMA memory")
    parser.add_argument("query", nargs="?", help="Semantic search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--raw", action="store_true", help="Output raw JSON")
    parser.add_argument("--full", action="store_true", help="Show full tile content (no truncation)")
    parser.add_argument("--chars", type=int, default=600, help="Content preview length (default: 600)")
    parser.add_argument("--file", metavar="PATH_FRAGMENT",
                        help="Filter by source_file path fragment (e.g. 'COFFEE_CACHE.md')")
    parser.add_argument("--hash", metavar="CONTENT_HASH",
                        help="Fetch all tiles by content_hash (exact, fast)")
    parser.add_argument("--uuid", metavar="TILE_ID",
                        help="Fetch a single tile by Weaviate UUID")
    parser.add_argument("--motif", metavar="MOTIF_NAME",
                        help="Find tiles tagged with HMM motif (e.g. GOD_EQUALS_MATH, BREAKTHROUGH_MOMENT)")
    parser.add_argument("--evolve", metavar="CONCEPT",
                        help="Show how a concept evolved over time (chronological semantic search)")
    args = parser.parse_args()

    content_limit = None if args.full else args.chars

    if args.uuid:
        tile = fetch_by_uuid(args.uuid)
        if args.raw:
            print(json.dumps(tile, indent=2))
        else:
            print(f"=== Tile: {args.uuid} ===")
            print_tiles([tile], content_limit)
        return

    if args.hash:
        results = search_by_hash(args.hash, args.limit)
        if args.raw:
            print(json.dumps(results, indent=2))
            return
        print(f"ISMA Hash: '{args.hash}' — {len(results)} tiles\n{'='*60}")
        print_tiles(results, content_limit)
        return

    if args.file:
        results = search_by_file(args.file, args.limit)
        if args.raw:
            print(json.dumps(results, indent=2))
            return
        print(f"ISMA File Filter: '{args.file}' — {len(results)} tiles\n{'='*60}")
        print_tiles(results, content_limit)
        return

    if args.motif:
        limit = args.limit if args.limit != 5 else 20
        results = search_by_motif(args.motif, limit)
        if args.raw:
            print(json.dumps(results, indent=2))
            return
        print(f"ISMA Motif: '{args.motif}' — {len(results)} tiles (chronological)\n{'='*60}")
        print_tiles(results, content_limit, evolve_mode=True)
        return

    if args.evolve:
        limit = args.limit if args.limit != 5 else 15
        results = search_evolve(args.evolve, limit)
        if args.raw:
            print(json.dumps(results, indent=2))
            return
        print(f"ISMA Evolution: '{args.evolve}' — {len(results)} tiles (oldest→newest)\n{'='*60}")
        print(f"Tracing how this concept developed over time...\n")
        print_tiles(results, content_limit, evolve_mode=True)
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    results = search(args.query, args.limit)
    if args.raw:
        print(json.dumps(results, indent=2))
        return
    print(f"ISMA Search: '{args.query}' — {len(results)} results\n{'='*60}")
    print_tiles(results, content_limit)


if __name__ == "__main__":
    main()
