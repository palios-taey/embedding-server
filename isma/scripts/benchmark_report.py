#!/usr/bin/env python3
"""
ISMA Benchmark Comparison Report

Compares two benchmark runs and generates a markdown report showing
improvements, regressions, and gate criteria status.

Usage:
    python3 benchmark_report.py baseline.json phase1.json
    python3 benchmark_report.py baseline.json phase1.json --output report.md
    python3 benchmark_report.py --latest   # Compare latest vs baseline
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def load_benchmark(path: str) -> Dict[str, Any]:
    """Load a benchmark result file."""
    with open(path) as f:
        return json.load(f)


def delta_str(old: float, new: float) -> str:
    """Format a delta with +/- sign and color hint."""
    if old < 0 or new < 0:
        return "N/A"
    d = new - old
    pct = (d / old * 100) if old > 0 else 0
    sign = "+" if d >= 0 else ""
    arrow = "^" if d > 0 else ("v" if d < 0 else "=")
    return f"{sign}{d:.4f} ({sign}{pct:.1f}%) {arrow}"


def compare_metrics(old: Dict, new: Dict, metrics: list) -> list:
    """Compare metrics between two benchmark summaries."""
    rows = []
    for metric in metrics:
        old_val = old.get(metric, -1)
        new_val = new.get(metric, -1)
        rows.append({
            "metric": metric,
            "old": old_val,
            "new": new_val,
            "delta": delta_str(old_val, new_val) if old_val >= 0 else "N/A",
            "improved": new_val > old_val if (old_val >= 0 and new_val >= 0) else None,
            "regressed": new_val < old_val - 0.02 if (old_val >= 0 and new_val >= 0) else False,
        })
    return rows


def generate_report(old_data: Dict, new_data: Dict,
                    old_label: str = None, new_label: str = None) -> str:
    """Generate a markdown comparison report."""
    old_label = old_label or old_data.get("label", "baseline")
    new_label = new_label or new_data.get("label", "current")

    lines = []
    lines.append(f"# ISMA Benchmark Comparison: {old_label} vs {new_label}")
    lines.append(f"")
    lines.append(f"**Generated**: {datetime.utcnow().isoformat()}")
    lines.append(f"**Baseline**: {old_data.get('timestamp', '?')} ({old_data['config']['total_queries']} queries)")
    lines.append(f"**Current**: {new_data.get('timestamp', '?')} ({new_data['config']['total_queries']} queries)")
    lines.append(f"")

    # Overall comparison
    old_ovr = old_data["summary"]["overall"]
    new_ovr = new_data["summary"]["overall"]

    metrics = [
        "recall_5_mean", "recall_10_mean", "mrr_mean",
        "precision_5_mean", "precision_10_mean",
        "dedup_5_mean", "dedup_10_mean",
        "latency_p50_ms", "latency_p95_ms",
        "enriched_in_top10_mean",
    ]

    lines.append("## Overall")
    lines.append("")
    lines.append(f"| Metric | {old_label} | {new_label} | Delta |")
    lines.append("|--------|---------|---------|-------|")

    comparisons = compare_metrics(old_ovr, new_ovr, metrics)
    for c in comparisons:
        name = c["metric"].replace("_mean", "").replace("_", " ").title()
        old_fmt = f"{c['old']:.4f}" if c['old'] >= 0 else "N/A"
        new_fmt = f"{c['new']:.4f}" if c['new'] >= 0 else "N/A"
        lines.append(f"| {name} | {old_fmt} | {new_fmt} | {c['delta']} |")

    lines.append("")

    # By category comparison
    lines.append("## By Category")
    lines.append("")

    categories = ["exact", "temporal", "conceptual", "relational", "motif"]
    key_metrics = ["recall_10_mean", "mrr_mean", "dedup_10_mean", "latency_p95_ms"]

    for cat in categories:
        old_cat = old_data["summary"]["by_category"].get(cat, {})
        new_cat = new_data["summary"]["by_category"].get(cat, {})

        if not old_cat.get("count") and not new_cat.get("count"):
            continue

        lines.append(f"### {cat.title()}")
        lines.append("")
        lines.append(f"| Metric | {old_label} | {new_label} | Delta |")
        lines.append("|--------|---------|---------|-------|")

        for m in key_metrics:
            old_val = old_cat.get(m, -1)
            new_val = new_cat.get(m, -1)
            name = m.replace("_mean", "").replace("_", " ").title()
            old_fmt = f"{old_val:.4f}" if old_val >= 0 else "N/A"
            new_fmt = f"{new_val:.4f}" if new_val >= 0 else "N/A"
            lines.append(f"| {name} | {old_fmt} | {new_fmt} | {delta_str(old_val, new_val)} |")

        lines.append("")

    # Regressions
    lines.append("## Regressions")
    lines.append("")

    regressions = []
    for cat in categories:
        old_cat = old_data["summary"]["by_category"].get(cat, {})
        new_cat = new_data["summary"]["by_category"].get(cat, {})
        old_r10 = old_cat.get("recall_10_mean", -1)
        new_r10 = new_cat.get("recall_10_mean", -1)
        if old_r10 >= 0 and new_r10 >= 0 and new_r10 < old_r10 - 0.02:
            regressions.append(f"- **{cat}**: Recall@10 dropped from {old_r10:.4f} to {new_r10:.4f} (delta: {new_r10 - old_r10:.4f})")

    if regressions:
        for r in regressions:
            lines.append(r)
    else:
        lines.append("No regressions detected (threshold: -0.02 on Recall@10)")

    lines.append("")

    # Per-query details (worst performers)
    lines.append("## Worst Performing Queries (by Recall@10)")
    lines.append("")

    new_details = sorted(
        [d for d in new_data["details"] if d.get("recall_10", -1) >= 0],
        key=lambda x: x["recall_10"]
    )
    lines.append("| ID | Category | Recall@10 | MRR | Dedup@10 | Query |")
    lines.append("|-----|----------|-----------|-----|----------|-------|")
    for d in new_details[:10]:
        q_short = d["query"][:40] + "..." if len(d["query"]) > 40 else d["query"]
        lines.append(
            f"| {d['query_id']} | {d['category']} | {d['recall_10']:.3f} | "
            f"{d['mrr']:.3f} | {d['dedup_10']:.3f} | {q_short} |"
        )

    lines.append("")

    # Gate criteria check
    lines.append("## Gate Criteria Status")
    lines.append("")

    # Phase 1 gates
    old_concept = old_data["summary"]["by_category"].get("conceptual", {})
    new_concept = new_data["summary"]["by_category"].get("conceptual", {})
    old_relat = old_data["summary"]["by_category"].get("relational", {})
    new_relat = new_data["summary"]["by_category"].get("relational", {})

    mrr_improvement = False
    if old_concept.get("mrr_mean", 0) > 0 and new_concept.get("mrr_mean", 0) > 0:
        concept_delta = (new_concept["mrr_mean"] - old_concept["mrr_mean"]) / old_concept["mrr_mean"]
        mrr_improvement = concept_delta >= 0.05
    if old_relat.get("mrr_mean", 0) > 0 and new_relat.get("mrr_mean", 0) > 0:
        relat_delta = (new_relat["mrr_mean"] - old_relat["mrr_mean"]) / old_relat["mrr_mean"]
        mrr_improvement = mrr_improvement or relat_delta >= 0.05

    no_regression = len(regressions) == 0
    latency_ok = new_ovr.get("latency_p95_ms", 9999) < 2000

    status = lambda ok: "PASS" if ok else "FAIL"
    lines.append(f"| Gate | Status |")
    lines.append(f"|------|--------|")
    lines.append(f"| MRR >= 5% improvement (conceptual+relational) | {status(mrr_improvement)} |")
    lines.append(f"| No regression (Recall@10 delta >= -0.02) | {status(no_regression)} |")
    lines.append(f"| Latency p95 < 2000ms | {status(latency_ok)} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ISMA Benchmark Comparison Report")
    parser.add_argument("baseline", nargs="?", help="Baseline benchmark JSON file")
    parser.add_argument("current", nargs="?", help="Current benchmark JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output markdown file")
    parser.add_argument("--latest", action="store_true",
                        help="Compare latest benchmark vs first (baseline)")
    args = parser.parse_args()

    if args.latest:
        # Find all benchmark files
        bench_dir = "/var/spark/isma"
        files = sorted([
            os.path.join(bench_dir, f)
            for f in os.listdir(bench_dir)
            if f.startswith("benchmark_") and f.endswith(".json")
            and f != "benchmark_latest.json"
        ])
        if len(files) < 2:
            print("Need at least 2 benchmark files for comparison")
            sys.exit(1)
        args.baseline = files[0]
        args.current = files[-1]

    if not args.baseline or not args.current:
        parser.error("Provide baseline and current benchmark files, or use --latest")

    old_data = load_benchmark(args.baseline)
    new_data = load_benchmark(args.current)

    report = generate_report(old_data, new_data)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
