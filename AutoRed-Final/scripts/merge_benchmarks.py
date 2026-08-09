#!/usr/bin/env python3
"""Merge benchmark results from multiple parallel workers.

Usage:
    python scripts/merge_benchmarks.py \
        --output merged_summary.json \
        --worker-results results/worker_0.json results/worker_1.json ...

Or with glob:
    python scripts/merge_benchmarks.py \
        --output merged_summary.json \
        --worker-results results/worker_*.json
"""

import argparse
import json
import os
import sys
import glob as glob_module
from datetime import datetime
from pathlib import Path


def load_worker_result(path: str) -> dict:
    """Load and validate a single worker result file."""
    with open(path, "r") as f:
        data = json.load(f)

    required_keys = ["success_rate", "total_successes", "total_rounds", "results"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {path}")

    return data


def merge_benchmarks(worker_paths: list[str], output_path: str) -> dict:
    """Merge results from multiple workers into a single summary."""
    # Load all worker results
    workers = []
    for path in worker_paths:
        print(f"  Loading: {path}")
        data = load_worker_result(path)
        workers.append(data)

    if not workers:
        print("[ERROR] No worker results found!")
        sys.exit(1)

    num_workers = len(workers)
    print(f"\n[MERGE] Combining {num_workers} worker results...")

    # Aggregate counters
    total_rounds = sum(w["total_rounds"] for w in workers)
    total_successes = sum(w["total_successes"] for w in workers)
    total_success_exact = sum(w.get("total_success_exact", 0) for w in workers)
    total_success_extractor = sum(w.get("total_success_extractor", 0) for w in workers)
    total_access_granted = sum(w.get("total_access_granted", 0) for w in workers)

    # Top-K metrics
    total_top1 = sum(w.get("top1_success", 0) for w in workers)
    total_top3 = sum(w.get("top3_success", 0) for w in workers)
    total_top5 = sum(w.get("top5_success", 0) for w in workers)
    total_verified = sum(w.get("verified_success", 0) for w in workers)

    # Collect all per-round results
    all_results = []
    for w in workers:
        worker_id = w.get("metadata", {}).get("worker_id", 0)
        for r in w.get("results", []):
            merged_round = dict(r)
            merged_round["worker_id"] = worker_id
            all_results.append(merged_round)

    # Compute aggregate metrics
    success_rate = total_successes / total_rounds if total_rounds > 0 else 0.0
    defense_rate = 1.0 - success_rate

    # Average attempts on success (weighted average)
    total_avg_attempts_sum = 0
    total_avg_attempts_count = 0
    for w in workers:
        avg = w.get("avg_attempts_on_success")
        count = w.get("total_successes", 0)
        if avg is not None and count > 0 and avg != float("inf"):
            total_avg_attempts_sum += avg * count
            total_avg_attempts_count += count

    avg_attempts = (
        total_avg_attempts_sum / total_avg_attempts_count
        if total_avg_attempts_count > 0
        else float("inf")
    )

    # Extractor metrics (sum TP/FP/FN across workers)
    combined_tp = sum(w.get("extractor_metrics", {}).get("true_positive", 0) for w in workers)
    combined_fp = sum(w.get("extractor_metrics", {}).get("false_positive", 0) for w in workers)
    combined_fn = sum(w.get("extractor_metrics", {}).get("false_negative", 0) for w in workers)
    combined_precision = combined_tp / (combined_tp + combined_fp) if (combined_tp + combined_fp) > 0 else 0.0
    combined_recall = combined_tp / (combined_tp + combined_fn) if (combined_tp + combined_fn) > 0 else 0.0
    combined_f1 = (
        2 * combined_precision * combined_recall / (combined_precision + combined_recall)
        if (combined_precision + combined_recall) > 0
        else 0.0
    )

    # Strategy stats (aggregate across workers)
    combined_strategy_stats = {}
    for w in workers:
        for strat, stats in w.get("strategy_stats", {}).items():
            if strat not in combined_strategy_stats:
                combined_strategy_stats[strat] = {
                    "attempts": 0, "successes": 0, "total_score": 0.0
                }
            combined_strategy_stats[strat]["attempts"] += stats.get("attempts", 0)
            combined_strategy_stats[strat]["successes"] += stats.get("successes", 0)
            combined_strategy_stats[strat]["total_score"] += stats.get("total_score", 0.0)

    # Compute averages for strategy stats
    for strat in combined_strategy_stats:
        s = combined_strategy_stats[strat]
        s["avg_score"] = s["total_score"] / s["attempts"] if s["attempts"] > 0 else 0.0
        s["success_rate"] = s["successes"] / s["attempts"] if s["attempts"] > 0 else 0.0

    # Build merged summary
    merged = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "target_model": workers[0].get("metadata", {}).get("target_model", "Llama-3-8B-Instruct"),
            "n_rounds": total_rounds,
            "max_interactions": workers[0].get("metadata", {}).get("max_interactions", 20),
            "num_workers": num_workers,
            "worker_ids": [
                w.get("metadata", {}).get("worker_id", i) for i, w in enumerate(workers)
            ],
            "merged_from": worker_paths,
        },
        "success_rate": success_rate,
        "defense_rate": defense_rate,
        "avg_attempts_on_success": avg_attempts,
        "total_successes": total_successes,
        "total_success_exact": total_success_exact,
        "total_success_extractor": total_success_extractor,
        "total_access_granted": total_access_granted,
        "total_rounds": total_rounds,
        # Top-K metrics
        "top1_success": total_top1,
        "top3_success": total_top3,
        "top5_success": total_top5,
        "verified_success": total_verified,
        # Extractor metrics
        "extractor_metrics": {
            "true_positive": combined_tp,
            "false_positive": combined_fp,
            "false_negative": combined_fn,
            "precision": combined_precision,
            "recall": combined_recall,
            "f1": combined_f1,
        },
        # Strategy stats
        "strategy_stats": combined_strategy_stats,
        # Per-round results
        "results": all_results,
        # Per-worker summaries (for reference)
        "worker_summaries": [
            {
                "worker_id": w.get("metadata", {}).get("worker_id", i),
                "rounds": w["total_rounds"],
                "successes": w["total_successes"],
                "success_rate": w["success_rate"],
            }
            for i, w in enumerate(workers)
        ],
    }

    # Save merged results
    output = Path(output_path)
    if output.is_dir():
        output = output / "merged_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 MERGED BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Workers:          {num_workers}")
    print(f"  Total Rounds:     {total_rounds}")
    print(f"  Success Rate:     {success_rate * 100:.1f}%")
    print(f"  Defense Rate:     {defense_rate * 100:.1f}%")
    print(f"  Avg Attempts:     {avg_attempts:.1f}" if avg_attempts != float("inf")
          else "  Avg Attempts:     N/A (no successes)")
    print(f"  Total Successes:  {total_successes}/{total_rounds}")
    print(f"  Generator Hit:    {total_success_exact}/{total_rounds}")
    print(f"  Extractor Hit:    {total_success_extractor}/{total_rounds}")
    print(f"  Access Granted:   {total_access_granted}/{total_rounds}")

    print(f"\n📊 TOP-K SUCCESS METRICS")
    print(f"{'=' * 60}")
    print(f"  Top-1 Success:    {total_top1}/{total_rounds}")
    print(f"  Top-3 Success:    {total_top3}/{total_rounds}")
    print(f"  Top-5 Success:    {total_top5}/{total_rounds}")
    print(f"  Verified Success: {total_verified}/{total_rounds}")

    print(f"\n📊 EXTRACTOR METRICS")
    print(f"{'=' * 60}")
    print(f"  True Positives:   {combined_tp}")
    print(f"  False Positives:  {combined_fp}")
    print(f"  False Negatives:  {combined_fn}")
    print(f"  Precision:        {combined_precision:.2%}")
    print(f"  Recall:           {combined_recall:.2%}")
    print(f"  F1 Score:         {combined_f1:.2%}")

    if combined_strategy_stats:
        print(f"\n📊 STRATEGY STATS")
        print(f"{'=' * 60}")
        for strat, stats in sorted(combined_strategy_stats.items(),
                                    key=lambda x: x[1].get("successes", 0),
                                    reverse=True):
            print(f"  {strat:20s} | attempts: {stats['attempts']:4d} | "
                  f"successes: {stats['successes']:3d} | "
                  f"avg_score: {stats.get('avg_score', 0):.2f}")

    print(f"{'=' * 60}")
    print(f"\n[JSON] Merged summary saved to: {output_path}")

    # Per-worker breakdown
    print(f"\n📊 PER-WORKER BREAKDOWN")
    print(f"{'=' * 60}")
    for ws in merged["worker_summaries"]:
        print(f"  Worker {ws['worker_id']}: {ws['successes']}/{ws['rounds']} "
              f"({ws['success_rate'] * 100:.1f}%)")
    print(f"{'=' * 60}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge multi-worker benchmark results")
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path for merged summary JSON"
    )
    parser.add_argument(
        "--worker-results", "-w", nargs="+", required=True,
        help="Paths to worker result JSON files (supports glob patterns)"
    )
    args = parser.parse_args()

    # Expand glob patterns.
    # IMPORTANT: a path that already resolves to an existing file is used
    # verbatim — we do NOT re-glob it. Benchmark output directories embed the
    # dataset slice as a literal (e.g. "..._Llama3-[1000:1000]_..."), and
    # glob.glob() would misread those brackets as a POSIX character class,
    # yielding zero matches even though the file exists. So: existing-file
    # args pass through untouched; only non-file args are treated as glob
    # patterns. This works for both invocation styles — fully shell-expanded
    # literal paths (argv already resolved) and unexpanded patterns.
    expanded_paths = []
    for pattern in args.worker_results:
        if os.path.isfile(pattern):
            expanded_paths.append(pattern)
            continue
        matched = glob_module.glob(pattern)
        if matched:
            expanded_paths.extend(matched)
        else:
            print(f"[WARN] No files matched pattern: {pattern}")

    if not expanded_paths:
        print("[ERROR] No worker result files found!")
        sys.exit(1)

    # Sort by worker ID in filename for consistent ordering
    expanded_paths.sort()

    print(f"[MERGE] Found {len(expanded_paths)} worker result files")
    merge_benchmarks(expanded_paths, args.output)


if __name__ == "__main__":
    main()
