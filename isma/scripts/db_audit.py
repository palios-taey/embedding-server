#!/usr/bin/env python3
"""
Definitive Database Audit - Single Source of Truth

Queries Weaviate, Neo4j, and Redis in one execution.
All numbers come from ONE run so they can't contradict each other.

Usage:
    python3 db_audit.py              # Print to stdout
    python3 db_audit.py --json       # JSON output
    python3 db_audit.py --save       # Save to /tmp/db_audit_latest.json

IMPORTANT: This is the ONLY way to get database stats.
Never run ad-hoc queries — always use this script.
"""

import json
import sys
import time
import redis
import requests
from datetime import datetime, timezone

# Connection constants (NCCL fabric IPs)
WEAVIATE_URL = "http://192.168.100.10:8088"
NEO4J_URI = "bolt://192.168.100.10:7689"
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379
WEAVIATE_CLASS = "ISMA_Quantum"

# Timeout for slow aggregate queries (enriched filter on 1M+ tiles)
WEAVIATE_TIMEOUT = 120  # seconds

_wv_session = requests.Session()


def _wv_graphql(query: str) -> dict:
    """Execute Weaviate GraphQL query, return data dict or raise."""
    r = _wv_session.post(
        f"{WEAVIATE_URL}/v1/graphql",
        json={"query": query},
        timeout=WEAVIATE_TIMEOUT,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise Exception(f"GraphQL errors: {body['errors']}")
    return body.get("data", {})


def _wv_rest_get(path: str) -> dict:
    """Execute Weaviate REST GET request."""
    r = _wv_session.get(f"{WEAVIATE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def audit_weaviate():
    """Get all Weaviate stats using raw GraphQL (no Python client)."""
    results = {
        "total_tiles": None,
        "hmm_enriched_true": None,
        "hmm_enriched_false": None,
        "scales": {},
        "with_rosetta_summary": None,
        "with_dominant_motifs": None,
        "with_motif_data_json": None,
        "sample_enriched": [],
        "sample_unenriched": [],
        "errors": [],
    }

    # 1. Total tile count (unfiltered — fast)
    try:
        data = _wv_graphql(f"""{{
            Aggregate {{ {WEAVIATE_CLASS} {{ meta {{ count }} }} }}
        }}""")
        results["total_tiles"] = data["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
        print(f"  Total tiles: {results['total_tiles']:,}")
    except Exception as e:
        results["errors"].append(f"total_count: {e}")

    # 2. Enriched count (hmm_enriched=true) — may be slow on 1M+ tiles
    try:
        data = _wv_graphql(f"""{{
            Aggregate {{
                {WEAVIATE_CLASS}(
                    where: {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
                ) {{ meta {{ count }} }}
            }}
        }}""")
        results["hmm_enriched_true"] = data["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
        print(f"  Enriched: {results['hmm_enriched_true']:,}")
    except Exception as e:
        results["errors"].append(f"enriched_count: {e}")

    # 3. Unenriched count (derive from total - enriched to avoid another slow query)
    if results["total_tiles"] is not None and results["hmm_enriched_true"] is not None:
        results["hmm_enriched_false"] = results["total_tiles"] - results["hmm_enriched_true"]

    # 4. Scale breakdown (search_512, rosetta, etc.)
    for scale in ["search_512", "rosetta", "document", "context_1024"]:
        try:
            data = _wv_graphql(f"""{{
                Aggregate {{
                    {WEAVIATE_CLASS}(
                        where: {{ path: ["scale"], operator: Equal, valueText: "{scale}" }}
                    ) {{ meta {{ count }} }}
                }}
            }}""")
            count = data["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
            results["scales"][scale] = count
            print(f"  Scale '{scale}': {count:,}")
        except Exception as e:
            results["errors"].append(f"scale_{scale}: {e}")

    # 5. Cursor-based sampling to measure rosetta_summary/motif presence
    # Sample 5000 tiles for statistical accuracy
    sample_size = 5000
    has_rosetta = 0
    has_motifs = 0
    has_motif_json = 0
    enriched_in_sample = 0
    sampled = 0
    cursor = ""
    batch_size = 200

    try:
        while sampled < sample_size:
            after_clause = f', after: "{cursor}"' if cursor else ""
            q = f"""{{
                Get {{
                    {WEAVIATE_CLASS}(
                        limit: {batch_size}{after_clause}
                    ) {{
                        rosetta_summary
                        dominant_motifs
                        motif_data_json
                        hmm_enriched
                        _additional {{ id }}
                    }}
                }}
            }}"""
            data = _wv_graphql(q)
            items = data.get("Get", {}).get(WEAVIATE_CLASS, [])
            if not items:
                break

            for item in items:
                sampled += 1
                rs = (item.get("rosetta_summary") or "").strip()
                dm = item.get("dominant_motifs") or []  # text[] array
                mj = (item.get("motif_data_json") or "").strip()
                enriched = item.get("hmm_enriched")

                if enriched:
                    enriched_in_sample += 1
                if rs and len(rs) > 5:
                    has_rosetta += 1
                if dm and len(dm) > 0:
                    has_motifs += 1
                if mj and len(mj) > 5:
                    has_motif_json += 1

                if sampled >= sample_size:
                    break

            cursor = items[-1]["_additional"]["id"]

        if sampled > 0:
            total = results["total_tiles"] or sampled
            results["with_rosetta_summary"] = {
                "sampled": sampled,
                "count_in_sample": has_rosetta,
                "pct": round(has_rosetta / sampled * 100, 1),
                "extrapolated": round(has_rosetta / sampled * total),
            }
            results["with_dominant_motifs"] = {
                "sampled": sampled,
                "count_in_sample": has_motifs,
                "pct": round(has_motifs / sampled * 100, 1),
                "extrapolated": round(has_motifs / sampled * total),
            }
            results["with_motif_data_json"] = {
                "sampled": sampled,
                "count_in_sample": has_motif_json,
                "pct": round(has_motif_json / sampled * 100, 1),
                "extrapolated": round(has_motif_json / sampled * total),
            }
            # Cross-check: enrichment rate in sample vs aggregate
            results["sample_enrichment_crosscheck"] = {
                "enriched_in_sample": enriched_in_sample,
                "sample_pct": round(enriched_in_sample / sampled * 100, 1),
                "aggregate_pct": round(results["hmm_enriched_true"] / total * 100, 1) if results["hmm_enriched_true"] else None,
            }
            print(f"  Sampled {sampled:,} tiles: {has_rosetta} with rosetta ({round(has_rosetta/sampled*100,1)}%)")
    except Exception as e:
        results["errors"].append(f"sampling: {e}")

    # 6. Sample 3 enriched tiles with rosetta content
    try:
        data = _wv_graphql(f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
                    limit: 3
                ) {{
                    rosetta_summary
                    dominant_motifs
                    platform
                    conversation_id
                    hmm_enriched
                }}
            }}
        }}""")
        items = data.get("Get", {}).get(WEAVIATE_CLASS, [])
        for item in items:
            results["sample_enriched"].append({
                "rosetta_summary": (item.get("rosetta_summary") or "")[:200],
                "dominant_motifs": (item.get("dominant_motifs") or "")[:200],
                "platform": item.get("platform"),
                "conversation_id": item.get("conversation_id"),
            })
    except Exception as e:
        results["errors"].append(f"sample_enriched: {e}")

    # 7. Sample 3 unenriched tiles
    try:
        data = _wv_graphql(f"""{{
            Get {{
                {WEAVIATE_CLASS}(
                    where: {{ path: ["hmm_enriched"], operator: NotEqual, valueBoolean: true }}
                    limit: 3
                ) {{
                    rosetta_summary
                    dominant_motifs
                    platform
                    conversation_id
                    hmm_enriched
                }}
            }}
        }}""")
        items = data.get("Get", {}).get(WEAVIATE_CLASS, [])
        for item in items:
            results["sample_unenriched"].append({
                "rosetta_summary": (item.get("rosetta_summary") or "")[:200],
                "dominant_motifs": (item.get("dominant_motifs") or "")[:200],
                "platform": item.get("platform"),
                "conversation_id": item.get("conversation_id"),
            })
    except Exception as e:
        results["errors"].append(f"sample_unenriched: {e}")

    return results


def audit_neo4j():
    """Get all Neo4j stats in one connection."""
    results = {
        "hmm_tiles": None,
        "hmm_tiles_with_rosetta": None,
        "hmm_motifs": None,
        "expresses_edges": None,
        "motif_names": [],
        "messages": None,
        "chat_sessions": None,
        "isma_exchanges": None,
        "errors": [],
    }

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI)

        with driver.session() as s:
            # HMMTile count
            try:
                r = s.run("MATCH (t:HMMTile) RETURN count(t) as c").single()
                results["hmm_tiles"] = r["c"]
            except Exception as e:
                results["errors"].append(f"hmm_tiles: {e}")

            # HMMTiles with non-empty rosetta_summary
            try:
                r = s.run("""
                    MATCH (t:HMMTile)
                    WHERE t.rosetta_summary IS NOT NULL AND t.rosetta_summary <> ''
                    RETURN count(t) as c
                """).single()
                results["hmm_tiles_with_rosetta"] = r["c"]
            except Exception as e:
                results["errors"].append(f"hmm_tiles_rosetta: {e}")

            # HMMMotif count
            try:
                r = s.run("MATCH (m:HMMMotif) RETURN count(m) as c").single()
                results["hmm_motifs"] = r["c"]
            except Exception as e:
                results["errors"].append(f"hmm_motifs: {e}")

            # EXPRESSES edges
            try:
                r = s.run("MATCH ()-[e:EXPRESSES]->() RETURN count(e) as c").single()
                results["expresses_edges"] = r["c"]
            except Exception as e:
                results["errors"].append(f"expresses: {e}")

            # Motif details
            try:
                records = s.run("""
                    MATCH (m:HMMMotif)
                    OPTIONAL MATCH (t:HMMTile)-[:EXPRESSES]->(m)
                    RETURN m.motif_id as id, m.name as name,
                           m.description as desc, count(t) as tile_count
                    ORDER BY tile_count DESC
                """).data()
                results["motif_names"] = [
                    {"id": r["id"], "name": r["name"],
                     "description": (r["desc"] or "")[:100],
                     "tile_count": r["tile_count"]}
                    for r in records
                ]
            except Exception as e:
                results["errors"].append(f"motif_names: {e}")

            # Messages
            try:
                r = s.run("MATCH (m:Message) RETURN count(m) as c").single()
                results["messages"] = r["c"]
            except Exception as e:
                results["errors"].append(f"messages: {e}")

            # ChatSessions
            try:
                r = s.run("MATCH (s:ChatSession) RETURN count(s) as c").single()
                results["chat_sessions"] = r["c"]
            except Exception as e:
                results["errors"].append(f"sessions: {e}")

            # ISMAExchange
            try:
                r = s.run("MATCH (e:ISMAExchange) RETURN count(e) as c").single()
                results["isma_exchanges"] = r["c"]
            except Exception as e:
                results["errors"].append(f"isma_exchanges: {e}")

        driver.close()
    except Exception as e:
        results["errors"].append(f"connection: {e}")

    return results


def audit_redis():
    """Get all Redis stats in one connection."""
    results = {
        "completed_set_size": None,
        "in_progress_count": None,
        "pipeline_stats": None,
        "motif_index_count": None,
        "current_packages": {},
        "errors": [],
    }

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Completed set
        try:
            results["completed_set_size"] = r.scard("hmm:pkg:completed")
        except Exception as e:
            results["errors"].append(f"completed: {e}")

        # In-progress
        try:
            keys = r.keys("hmm:pkg:in_progress:*")
            results["in_progress_count"] = len(keys)
        except Exception as e:
            results["errors"].append(f"in_progress: {e}")

        # Pipeline stats
        try:
            stats = r.hgetall("hmm:pkg:stats")
            results["pipeline_stats"] = stats
        except Exception as e:
            results["errors"].append(f"stats: {e}")

        # Motif index
        try:
            keys = r.keys("hmm:motif:*")
            results["motif_index_count"] = len(keys)
        except Exception as e:
            results["errors"].append(f"motif_index: {e}")

        # Current packages per platform
        try:
            for platform in ["chatgpt", "claude", "gemini", "grok", "perplexity"]:
                pkg = r.get(f"hmm:pkg:current:{platform}")
                if pkg:
                    try:
                        data = json.loads(pkg)
                        results["current_packages"][platform] = {
                            "package_file": data.get("package_file", ""),
                            "item_count": data.get("item_count", 0),
                        }
                    except json.JSONDecodeError:
                        results["current_packages"][platform] = pkg
        except Exception as e:
            results["errors"].append(f"current_packages: {e}")

    except Exception as e:
        results["errors"].append(f"connection: {e}")

    return results


def run_audit():
    """Run complete audit across all databases."""
    start = time.time()
    ts = datetime.now(timezone.utc).isoformat()

    print(f"Starting database audit at {ts}...")
    print(f"Weaviate: {WEAVIATE_URL}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print()

    print("Querying Weaviate (may take 30-60s for aggregate filters)...")
    weaviate_results = audit_weaviate()

    print("\nQuerying Neo4j...")
    neo4j_results = audit_neo4j()

    print("Querying Redis...")
    redis_results = audit_redis()

    elapsed = round(time.time() - start, 1)

    audit = {
        "timestamp": ts,
        "elapsed_seconds": elapsed,
        "weaviate": weaviate_results,
        "neo4j": neo4j_results,
        "redis": redis_results,
    }

    # Collect all errors
    all_errors = (
        weaviate_results.get("errors", []) +
        neo4j_results.get("errors", []) +
        redis_results.get("errors", [])
    )

    return audit, all_errors


def print_report(audit):
    """Print human-readable report."""
    w = audit["weaviate"]
    n = audit["neo4j"]
    r = audit["redis"]

    print("=" * 70)
    print(f"  DATABASE AUDIT - {audit['timestamp']}")
    print(f"  Completed in {audit['elapsed_seconds']}s")
    print("=" * 70)

    print("\n--- WEAVIATE (ISMA_Quantum) ---")
    print(f"  Total tiles:           {w['total_tiles']:,}" if w['total_tiles'] else "  Total tiles:           ERROR")
    print(f"  hmm_enriched=true:     {w['hmm_enriched_true']:,}" if w['hmm_enriched_true'] else "  hmm_enriched=true:     ERROR")
    print(f"  hmm_enriched=false:    {w['hmm_enriched_false']:,}" if w['hmm_enriched_false'] is not None else "  hmm_enriched=false:    ERROR")

    if w.get("scales"):
        print(f"\n  Scale breakdown:")
        for scale, count in sorted(w["scales"].items(), key=lambda x: -x[1]):
            print(f"    {scale}: {count:,}")

    if w.get("with_rosetta_summary"):
        rs = w["with_rosetta_summary"]
        print(f"\n  Content analysis ({rs['sampled']:,} tiles sampled):")
        print(f"    With rosetta_summary:  {rs['count_in_sample']:,} ({rs['pct']}%) -> ~{rs['extrapolated']:,} extrapolated")
    if w.get("with_dominant_motifs"):
        dm = w["with_dominant_motifs"]
        print(f"    With dominant_motifs:  {dm['count_in_sample']:,} ({dm['pct']}%) -> ~{dm['extrapolated']:,} extrapolated")
    if w.get("with_motif_data_json"):
        mj = w["with_motif_data_json"]
        print(f"    With motif_data_json:  {mj['count_in_sample']:,} ({mj['pct']}%) -> ~{mj['extrapolated']:,} extrapolated")

    if w.get("sample_enrichment_crosscheck"):
        cc = w["sample_enrichment_crosscheck"]
        print(f"\n  Enrichment cross-check:")
        print(f"    Sample enrichment rate: {cc['sample_pct']}%")
        print(f"    Aggregate enrichment rate: {cc['aggregate_pct']}%")
        if cc['aggregate_pct']:
            delta = abs(cc['sample_pct'] - cc['aggregate_pct'])
            status = "CONSISTENT" if delta < 5 else "DIVERGENT"
            print(f"    Delta: {delta:.1f}% -> {status}")

    enrichment_rate = None
    if w['total_tiles'] and w['hmm_enriched_true']:
        enrichment_rate = round(w['hmm_enriched_true'] / w['total_tiles'] * 100, 1)
        print(f"\n  Overall enrichment rate: {enrichment_rate}%")

    if w.get("sample_enriched"):
        print(f"\n  Sample enriched tile:")
        s = w["sample_enriched"][0]
        print(f"    Platform: {s.get('platform')}")
        print(f"    Rosetta:  {s.get('rosetta_summary', 'N/A')[:150]}...")
        print(f"    Motifs:   {s.get('dominant_motifs', 'N/A')[:150]}...")

    if w.get("sample_unenriched"):
        print(f"\n  Sample unenriched tile:")
        s = w["sample_unenriched"][0]
        print(f"    Platform: {s.get('platform')}")
        print(f"    Rosetta:  {(s.get('rosetta_summary') or '(empty)')[:150]}")

    print("\n--- NEO4J (Graph Layer) ---")
    print(f"  HMMTile nodes:         {n['hmm_tiles']:,}" if n['hmm_tiles'] is not None else "  HMMTile nodes:         ERROR")
    print(f"  HMMTiles w/rosetta:    {n['hmm_tiles_with_rosetta']:,}" if n['hmm_tiles_with_rosetta'] is not None else "  HMMTiles w/rosetta:    ERROR")
    print(f"  HMMMotif nodes:        {n['hmm_motifs']:,}" if n['hmm_motifs'] is not None else "  HMMMotif nodes:        ERROR")
    print(f"  EXPRESSES edges:       {n['expresses_edges']:,}" if n['expresses_edges'] is not None else "  EXPRESSES edges:       ERROR")
    print(f"  Messages:              {n['messages']:,}" if n['messages'] is not None else "  Messages:              ERROR")
    print(f"  ChatSessions:          {n['chat_sessions']:,}" if n['chat_sessions'] is not None else "  ChatSessions:          ERROR")
    if n.get('isma_exchanges') is not None:
        print(f"  ISMAExchanges:         {n['isma_exchanges']:,}")

    if n.get("motif_names"):
        print(f"\n  Top motifs by tile count:")
        for m in n["motif_names"][:10]:
            name = m.get("name") or m.get("id") or "unnamed"
            print(f"    {name}: {m['tile_count']:,} tiles")

    print("\n--- REDIS (Pipeline State) ---")
    print(f"  Completed set:         {r['completed_set_size']:,}" if r['completed_set_size'] is not None else "  Completed set:         ERROR")
    print(f"  In-progress:           {r['in_progress_count']:,}" if r['in_progress_count'] is not None else "  In-progress:           ERROR")
    print(f"  Motif index entries:   {r['motif_index_count']:,}" if r['motif_index_count'] is not None else "  Motif index entries:   ERROR")

    if r.get("pipeline_stats"):
        stats = r["pipeline_stats"]
        print(f"\n  Pipeline stats:")
        for k, v in sorted(stats.items()):
            print(f"    {k}: {v}")

    if r.get("current_packages"):
        print(f"\n  Current packages:")
        for platform, pkg in r["current_packages"].items():
            if isinstance(pkg, dict):
                print(f"    {platform}: {pkg.get('item_count', '?')} items")
            else:
                print(f"    {platform}: {pkg}")

    # Cross-database consistency checks
    print("\n--- CONSISTENCY CHECKS ---")
    if w['total_tiles'] and w['hmm_enriched_true']:
        gap = w['hmm_enriched_true'] - (n['hmm_tiles'] or 0)
        print(f"  Weaviate enriched vs Neo4j HMMTiles: {w['hmm_enriched_true']:,} vs {n['hmm_tiles'] or 0:,} (gap: {gap:,})")
        if gap > 0:
            print(f"    -> {gap:,} tiles enriched in Weaviate without Neo4j graph entry")
            print(f"    -> Tier1 batch classify writes Weaviate only (no Neo4j HMMTile)")
            print(f"    -> Tier2 AI enrichment writes both Weaviate + Neo4j")

    if r['completed_set_size'] and n['hmm_tiles']:
        print(f"  Redis completed hashes vs Neo4j HMMTiles: {r['completed_set_size']:,} vs {n['hmm_tiles']:,}")
        if r['completed_set_size'] > n['hmm_tiles']:
            gap = r['completed_set_size'] - n['hmm_tiles']
            print(f"    -> {gap:,} completed hashes without Neo4j node (old pipeline or tier1-only)")

    # Errors
    all_errors = w.get("errors", []) + n.get("errors", []) + r.get("errors", [])
    if all_errors:
        print(f"\n--- ERRORS ({len(all_errors)}) ---")
        for e in all_errors:
            print(f"  ! {e}")
    else:
        print(f"\n  No errors. All queries succeeded.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    audit, errors = run_audit()

    if "--json" in sys.argv:
        print(json.dumps(audit, indent=2, default=str))
    else:
        print_report(audit)

    if "--save" in sys.argv:
        path = "/tmp/db_audit_latest.json"
        with open(path, "w") as f:
            json.dump(audit, f, indent=2, default=str)
        print(f"\nSaved to {path}")

    sys.exit(1 if errors else 0)
