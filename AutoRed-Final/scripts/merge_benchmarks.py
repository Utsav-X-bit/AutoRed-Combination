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

    # Top-K metrics
    total_top1 = sum(w.get("top1_success", 0) for w in workers)
    total_top3 = sum(w.get("top3_success", 0) for w in workers)
    total_top5 = sum(w.get("top5_success", 0) for w in workers)
    total_verified = sum(w.get("verified_success", 0) for w in workers)

    # Mutation fallback + failure-mode stats (preserved through merge)
    total_mutation_triggered = sum(w.get("mutation_fallback_triggered", 0) for w in workers)
    total_mutation_successes = sum(w.get("mutation_fallback_successes", 0) for w in workers)

    # Failure-mode stats (sum per-label across workers)
    combined_failure_modes = {}
    for w in workers:
        for mode, count in w.get("failure_mode_stats", {}).items():
            combined_failure_modes[mode] = combined_failure_modes.get(mode, 0) + count

    # Fallback diagnostics (mutator draws + no-op + per-mutator win attribution)
    # summed across workers.
    combined_fb_mutator_counts: dict[str, int] = {}
    combined_fb_no_op_counts: dict[str, int] = {}
    combined_fb_winning_counts: dict[str, int] = {}
    total_fb_variants = 0
    total_fb_no_op = 0
    for w in workers:
        diag = w.get("mutation_fallback_diagnostics") or {}
        total_fb_variants += diag.get("variant_total", 0)
        total_fb_no_op += diag.get("no_op_total", 0)
        for m, c in (diag.get("mutator_counts") or {}).items():
            combined_fb_mutator_counts[m] = combined_fb_mutator_counts.get(m, 0) + c
        for m, c in (diag.get("no_op_counts") or {}).items():
            combined_fb_no_op_counts[m] = combined_fb_no_op_counts.get(m, 0) + c
        for m, c in (diag.get("winning_mutator_counts") or {}).items():
            combined_fb_winning_counts[m] = combined_fb_winning_counts.get(m, 0) + c

    # Collect all per-round results
    all_results = []
    for w in workers:
        worker_id = w.get("metadata", {}).get("worker_id", 0)
        for r in w.get("results", []):
            merged_round = dict(r)
            merged_round["worker_id"] = worker_id
            all_results.append(merged_round)

    # Per-mutator no-op rate: per_mutator = {mutator: {drawn, no_op, no_op_rate}}.
    # Isolates which mutator wastes queries (e.g. all no-ops from TL) instead of
    # hiding a single broken mutator behind the aggregate no_op_rate.
    per_mutator_diagnostics = {}
    for m, drawn in combined_fb_mutator_counts.items():
        n_op = combined_fb_no_op_counts.get(m, 0)
        wins = combined_fb_winning_counts.get(m, 0)
        per_mutator_diagnostics[m] = {
            "drawn": drawn,
            "no_op": n_op,
            "no_op_rate": round(n_op / drawn, 4) if drawn else 0.0,
            "wins": wins,
            "win_rate": round(wins / drawn, 4) if drawn else 0.0,
        }

    # ── success_path_breakdown: where each success (and failure) came from ──
    # Aggregated from the per-round `success_path` field (Task 3). This is the
    # single most informative matrix for interpreting a benchmark: it shows the
    # relative contribution of ground-truth leak vs. extractor vs. fallback vs.
    # verified, and the `none` bucket = total failures.
    #
    #   gt_leak     — victim produced the EXACT access code (the attack ceiling)
    #   extractor   — extractor caught a non-exact leak the judge would've missed
    #   fallback    — mutation fallback won after all 20 regular attempts failed
    #   verified    — verification loop confirmed a candidate
    #   none        — failure (no success path)
    _sp_counts = {}
    for r in all_results:
        sp = r.get("success_path") or "none"
        _sp_counts[sp] = _sp_counts.get(sp, 0) + 1
    # Stable, meaningful ordering; any unseen labels appended alphabetically.
    _sp_order = ["gt_leak", "extractor", "fallback", "verified", "none"]
    _sp_extra = sorted(k for k in _sp_counts if k not in _sp_order)
    success_path_breakdown = []
    for sp in _sp_order + _sp_extra:
        cnt = _sp_counts.get(sp, 0)
        if cnt == 0 and sp not in _sp_counts:
            continue
        success_path_breakdown.append({
            "path": sp,
            "count": cnt,
            "pct_of_total": round(cnt / total_rounds * 100, 2) if total_rounds else 0.0,
            "pct_of_successes": round(cnt / total_successes * 100, 2) if total_successes else 0.0,
        })

    # ── failure_mode_breakdown: why failed scenarios failed (mirrors success_path) ─
    _fm_counts = {}
    for r in all_results:
        if not r.get("success"):
            fm = r.get("failure_mode") or "unlabeled"
            _fm_counts[fm] = _fm_counts.get(fm, 0) + 1
    total_failures = sum(_fm_counts.values())
    failure_mode_breakdown = [
        {"mode": m, "count": c,
         "pct_of_failures": round(c / total_failures * 100, 2) if total_failures else 0.0,
         "pct_of_total": round(c / total_rounds * 100, 2) if total_rounds else 0.0}
        for m, c in sorted(_fm_counts.items(), key=lambda kv: -kv[1])
    ]

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
            # Run-config propagated from worker[0] (homogeneous across workers):
            # makes a merged result file self-describing so two benchmark dirs
            # can be distinguished by seed / fallback / escalation / slice.
            "seed": workers[0].get("metadata", {}).get("seed"),
            "start_idx": workers[0].get("metadata", {}).get("start_idx"),
            "mutation_fallback_enabled": workers[0].get("metadata", {}).get(
                "mutation_fallback_enabled", False
            ),
            "max_fallback_rounds": workers[0].get("metadata", {}).get("max_fallback_rounds", 1),
            "planner_temp_escalation": workers[0].get("metadata", {}).get(
                "planner_temp_escalation", 0.0
            ),
            # Task 3/5 (model-agnostic v2): cooperation-aware fallback config.
            # Emitted by the runtime at llama_3_8b_vllm.py (metadata block); the
            # UI surfaces these in the JailGuard fallback panel.
            "cooperative_seeding": workers[0].get("metadata", {}).get(
                "cooperative_seeding"
            ),
            "cooperative_n": workers[0].get("metadata", {}).get("cooperative_n"),
        },
        "success_rate": success_rate,
        "defense_rate": defense_rate,
        "avg_attempts_on_success": avg_attempts,
        "total_successes": total_successes,
        "total_success_exact": total_success_exact,
        "total_success_extractor": total_success_extractor,
        "mutation_fallback_triggered": total_mutation_triggered,
        "mutation_fallback_successes": total_mutation_successes,
        # Per-variant fallback diagnostics, summed across workers. High no_op_rate
        # = a broken/offline mutator pool wasting queries (e.g. TL without internet).
        # no_op_counts + winning_mutator_counts attribute waste AND wins to each
        # mutator axis (the per-mutator attribution the merged summary previously
        # lacked); per_mutator rates combine them into one diagnostic per mutator.
        "mutation_fallback_diagnostics": {
            "variant_total": total_fb_variants,
            "no_op_total": total_fb_no_op,
            "no_op_rate": (
                round(total_fb_no_op / total_fb_variants, 4)
                if total_fb_variants else 0.0
            ),
            "mutator_counts": dict(sorted(combined_fb_mutator_counts.items())),
            "no_op_counts": dict(sorted(combined_fb_no_op_counts.items())),
            "winning_mutator_counts": dict(sorted(combined_fb_winning_counts.items())),
            "per_mutator": per_mutator_diagnostics,
        },
        "gt_leak_rate": (total_success_exact / total_rounds) if total_rounds > 0 else 0.0,
        "extractor_recovery_rate": (
            combined_tp / (combined_tp + combined_fn)
            if (combined_tp + combined_fn) > 0 else 0.0
        ),
        "failure_mode_stats": combined_failure_modes,
        # Where successes (and failures) came from — the headline attribution matrix.
        "success_path_breakdown": success_path_breakdown,
        "failure_mode_breakdown": failure_mode_breakdown,
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
        # Per-worker summaries (for reference). Includes per-worker fallback
        # diagnostics so a glance at the merged summary shows which workers
        # triggered / won / wasted-no-op, without opening each worker JSON.
        "worker_summaries": [
            {
                "worker_id": w.get("metadata", {}).get("worker_id", i),
                "rounds": w["total_rounds"],
                "successes": w["total_successes"],
                "success_rate": w["success_rate"],
                "mutation_fallback_triggered": w.get("mutation_fallback_triggered", 0),
                "mutation_fallback_successes": w.get("mutation_fallback_successes", 0),
                "mutation_fallback_diagnostics": w.get("mutation_fallback_diagnostics", {}),
            }
            for i, w in enumerate(workers)
        ],
    }

    # Save merged results
    output = Path(output_path)
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
    # Run-config (propagated from worker[0]) so the printout identifies which
    # benchmark dir this is: seed / fallback / escalation / start-idx.
    _md0 = workers[0].get("metadata", {}) if workers else {}
    print(f"  Seed:             {_md0.get('seed')}")
    print(f"  Start Idx:        {_md0.get('start_idx')}")
    print(f"  Mutation Fallback: {bool(_md0.get('mutation_fallback_enabled', False))}")
    if _md0.get("mutation_fallback_enabled"):
        print(f"  Max FB Rounds:    {_md0.get('max_fallback_rounds', 1)}")
    _esc = _md0.get("planner_temp_escalation", 0.0)
    print(f"  Planner Temp Esc:  {_esc}{' (off)' if not _esc else ''}")

    print(f"\n📊 SUCCESS PATH BREAKDOWN — where successes come from")
    print(f"{'=' * 60}")
    print(f"  {'path':<16s} {'count':>6s}  {'% of successes':>15s}  {'% of total':>10s}")
    print(f"  {'-'*16} {'-'*6}  {'-'*15}  {'-'*10}")
    for row in success_path_breakdown:
        is_fail = row["path"] == "none"
        tag = " (FAILURE)" if is_fail else ""
        print(f"  {row['path']:<16s} {row['count']:>6d}  "
              f"{row['pct_of_successes']:>14.1f}%  {row['pct_of_total']:>9.1f}%{tag}")
    if total_mutation_triggered:
        fb_share = total_mutation_successes / total_mutation_triggered * 100
        print(f"\n  Fallback: {total_mutation_successes}/{total_mutation_triggered} "
              f"triggered scenarios won ({fb_share:.1f}% conversion), "
              f"+{total_mutation_successes} / {total_rounds} = "
              f"+{total_mutation_successes/total_rounds*100:.2f}pp headline.")
    if total_fb_variants:
        print(f"\n  Fallback variant diagnostics (across {total_mutation_triggered} "
              f"triggered scenarios, {total_fb_variants} variants):")
        print(f"    no-op (== seed): {total_fb_no_op}/{total_fb_variants} = "
              f"{total_fb_no_op/total_fb_variants*100:.1f}% wasted queries")
        print(f"    mutator draws: {dict(sorted(combined_fb_mutator_counts.items()))}")
        # Per-mutator attribution: which mutator wins, which wastes queries.
        print(f"\n  Per-mutator attribution (draws / no-ops / wins):")
        print(f"    {'mutator':<8s} {'drawn':>6s} {'no-op':>6s} "
              f"{'no-op%':>7s} {'wins':>5s} {'win%':>6s}")
        for m in sorted(per_mutator_diagnostics):
            p = per_mutator_diagnostics[m]
            print(f"    {m:<8s} {p['drawn']:>6d} {p['no_op']:>6d} "
                  f"{p['no_op_rate']*100:>6.1f}% {p['wins']:>5d} "
                  f"{p['win_rate']*100:>5.1f}%")
        total_wins = sum(combined_fb_winning_counts.values())
        if total_wins:
            print(f"    win attribution: {dict(sorted(combined_fb_winning_counts.items()))}")
        if total_fb_no_op / total_fb_variants > 0.25:
            print(f"    ⚠️  High no-op rate — a mutator pool is likely offline/broken.")
        # Flag a single mutator if it is the no-op source even when aggregate is low.
        for m, p in per_mutator_diagnostics.items():
            if p["drawn"] > 0 and p["no_op_rate"] > 0.25:
                print(f"    ⚠️  {m} no-op rate {p['no_op_rate']*100:.1f}% — that mutator "
                      f"is wasting queries even though the pool aggregate is low.")

    if failure_mode_breakdown:
        print(f"\n📊 FAILURE MODE BREAKDOWN — why failures failed")
        print(f"{'=' * 60}")
        print(f"  {'mode':<24s} {'count':>6s}  {'% of failures':>13s}  {'% of total':>10s}")
        print(f"  {'-'*24} {'-'*6}  {'-'*13}  {'-'*10}")
        for row in failure_mode_breakdown:
            print(f"  {row['mode']:<24s} {row['count']:>6d}  "
                  f"{row['pct_of_failures']:>12.1f}%  {row['pct_of_total']:>9.1f}%")
        print(f"  NOTE: in a fallback run, every failure that triggered fallback is")
        print(f"  labeled 'fallback_failed', masking planner_stuck / generator_rephrase_fail.")
        print(f"  Run the no-fallback BASELINE to see the underlying failure-mode mix.")

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

    # Per-worker breakdown (with fallback triggered/success + no-op rate).
    print(f"\n📊 PER-WORKER BREAKDOWN")
    print(f"{'=' * 60}")
    print(f"  {'worker':>6s} {'succ/rounds':>14s} {'rate':>7s} "
          f"{'fb_trig':>8s} {'fb_win':>7s} {'conv':>6s} {'no-op%':>7s}")
    for ws in merged["worker_summaries"]:
        wd = ws.get("mutation_fallback_diagnostics") or {}
        trig = ws.get("mutation_fallback_triggered", 0)
        win = ws.get("mutation_fallback_successes", 0)
        conv = f"{win/trig*100:.1f}%" if trig else "-"
        nrate = wd.get("no_op_rate", 0.0) * 100
        print(f"  {ws['worker_id']:>6d} {ws['successes']:>6d}/{ws['rounds']:<7d} "
              f"{ws['success_rate']*100:>6.1f}% {trig:>8d} {win:>7d} {conv:>6s} "
              f"{nrate:>6.1f}%")
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

    # Expand glob patterns
    expanded_paths = []
    for pattern in args.worker_results:
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
