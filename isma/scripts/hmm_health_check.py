#!/usr/bin/env python3
"""HMM Pipeline Health Check — 6SIGMA fail-loud monitoring.

Checks pipeline integrity every 15 minutes via cron.
Any failure triggers tmux alert to taeys-hands session.

Usage:
    python3 hmm_health_check.py           # Run all checks
    python3 hmm_health_check.py --quiet   # Only report failures
"""

import argparse
import json
import subprocess
import sys
import time

import redis
import requests
from neo4j import GraphDatabase

REDIS_HOST = "192.168.100.10"
NEO4J_URI = "bolt://192.168.100.10:7689"
WEAVIATE_URL = "http://192.168.100.10:8088"
EMBEDDING_URL = "http://192.168.100.10:8091"
HEALTH_FILE = "/var/spark/isma/hmm_health.json"
ALERT_TMUX = "taeys-hands"
COMPLETED_KEY = "hmm:pkg:completed"
IN_PROGRESS_PREFIX = "hmm:pkg:in_progress:"


def check_all(quiet=False):
    issues = []
    info = {}

    # 1. Redis reachable + completed count
    try:
        r = redis.Redis(REDIS_HOST, decode_responses=True)
        r.ping()
        redis_completed = r.scard(COMPLETED_KEY)
        info["redis_completed"] = redis_completed
    except Exception as e:
        issues.append(f"Redis unreachable: {e}")
        redis_completed = -1

    # 2. Neo4j reachable + HMMTile count
    try:
        driver = GraphDatabase.driver(NEO4J_URI)
        with driver.session() as s:
            neo4j_tiles = s.run("MATCH (t:HMMTile) RETURN count(t) as c").single()["c"]
        driver.close()
        info["neo4j_tiles"] = neo4j_tiles
    except Exception as e:
        issues.append(f"Neo4j unreachable: {e}")
        neo4j_tiles = -1

    # 3. Redis vs Neo4j consistency
    if redis_completed >= 0 and neo4j_tiles >= 0:
        gap = redis_completed - neo4j_tiles
        info["gap"] = gap
        if gap > 100:
            issues.append(f"Redis/Neo4j gap: {gap} phantom completions (Redis={redis_completed}, Neo4j={neo4j_tiles})")

    # 4. Stale in-progress keys
    if redis_completed >= 0:
        try:
            stale = 0
            for key in r.scan_iter(f"{IN_PROGRESS_PREFIX}*"):
                ttl = r.ttl(key)
                if 0 < ttl < 3600:  # Less than 1hr remaining of 2hr TTL
                    stale += 1
            info["stale_in_progress"] = stale
            if stale > 20:
                issues.append(f"Stale in-progress: {stale} items stuck for >1hr")
        except Exception:
            pass

    # 5. Weaviate reachable
    try:
        resp = requests.get(f"{WEAVIATE_URL}/v1/meta", timeout=5)
        if resp.status_code != 200:
            issues.append(f"Weaviate unhealthy: HTTP {resp.status_code}")
        else:
            info["weaviate_version"] = resp.json().get("version", "unknown")
    except Exception as e:
        issues.append(f"Weaviate unreachable: {e}")

    # 6. Embedding server reachable
    try:
        resp = requests.get(f"{EMBEDDING_URL}/v1/models", timeout=5)
        if resp.status_code != 200:
            issues.append(f"Embedding server unhealthy: HTTP {resp.status_code}")
        else:
            info["embedding_ok"] = True
    except Exception as e:
        issues.append(f"Embedding server unreachable: {e}")

    # Write health file
    status = {
        "healthy": len(issues) == 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issues": issues,
        **info,
    }

    try:
        with open(HEALTH_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass

    if issues:
        alert_msg = f"ALERT: HMM pipeline — {len(issues)} issue(s): {'; '.join(issues[:3])}"
        # Truncate to avoid tmux overflow
        if len(alert_msg) > 300:
            alert_msg = alert_msg[:297] + "..."
        try:
            subprocess.run(
                ["tmux", "send-keys", "-t", ALERT_TMUX, alert_msg, "Enter"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        print(alert_msg, file=sys.stderr)
        sys.exit(1)
    else:
        if not quiet:
            print(f"OK: Redis={redis_completed}, Neo4j={neo4j_tiles}, gap={info.get('gap', 'N/A')}")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Pipeline Health Check")
    parser.add_argument("--quiet", action="store_true", help="Only report failures")
    args = parser.parse_args()
    check_all(quiet=args.quiet)
