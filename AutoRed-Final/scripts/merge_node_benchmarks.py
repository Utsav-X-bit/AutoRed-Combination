#!/usr/bin/env python3
"""Merge benchmark results from multiple NODES into one combined summary.

Why this exists (distinct from ``scripts/merge_benchmarks.py``)
---------------------------------------------------------------
``merge_benchmarks.py`` merges the 4 *worker*-level JSONs produced by a single
node (``logs/worker_{0..3}.json`` → ``logs/merged_summary.json``) — it operates
*within* one node's 4-GPU run. It also DROPS several fields the per-worker
summaries carry (``mutation_fallback_triggered`` / ``mutation_fallback_successes``
/ ``mutation_fallback_diagnostics`` / ``per_type_stats`` / ``failure_mode_stats``
/ ``models`` / and the run-config metadata fields ``seed`` / ``start_idx`` /
``mutation_fallback_enabled`` / ``max_fallback_rounds`` / ``cooperative_seeding``
/ ``cooperative_n`` / ``planner_temp_escalation`` / ``planner_model`` /
``generator_model``).

This script merges across NODES — two 4-GPU nodes that each ran a disjoint
``--start-idx`` slice of the deduped scenario pool — into one final benchmark.
It does three things ``merge_benchmarks.py`` does not:

1. **Worker-id re-keying across nodes.** Both nodes emit ``worker_0..3.json``.
   ``analyze_benchmark_comparison.load_worker_summaries`` keys worker JSONs by
   the filename-derived id (``worker_(\\d+)\\.json$``) and requires all-digit
   ids. Left as-is, the two nodes' worker files would collide in a combined
   ``logs/`` dir. So this script re-keys them to unique global ids:
   node *k* (0-indexed) worker *w* → ``worker_{k*workers_per_node + w}.json``
   (node 1 → 0..3, node 2 → 4..7). The original node-local id is preserved as
   ``metadata.node_worker_id`` and the node index as ``metadata.node_id``.

2. **Full-schema aggregation.`` The combined ``merged_summary.json`` reuses
   ``merge_benchmarks.py``'s aggregation for the shared counters (total_rounds,
   successes, top-K, extractor TP/FP/FN, strategy_stats, results list,
   worker_summaries) AND additionally sums the fields that script drops:
   ``mutation_fallback_triggered``, ``mutation_fallback_successes``,
   ``mutation_fallback_diagnostics`` (variant_total / no_op_total / mutator_counts
   / no_op_counts / winning_mutator_counts, with a re-derived no_op_rate),
   ``per_type_stats`` (per defense_type), ``failure_mode_stats`` (per mode),
   ``avg_verified_rank`` (weighted by verified_success), and ``models`` (kept
   from the first worker — identical across a run). It also preserves the
   run-config metadata fields so the combined summary is self-describing.

3. **Optional KB/RAG rebuild across both nodes' run trees.`` With
   ``--rebuild-kb``, it invokes ``scripts.rebuild_kb_rag.rebuild`` once, pointing
   ``--results-dir`` at a common parent that contains both nodes'
   ``runs/{success,failed}/run_*.json`` subtrees so the oracle miner sees every
   run in one pass.

Usage
-----
    # nodes passed as their <chars> run-dirs (the parent of logs/ and runs/):
    python3 scripts/merge_node_benchmarks.py \
        --node-dir results/benchmark/Llama3-8B/<chars_node1> \
        --node-dir results/benchmark/Llama3-8B/<chars_node2> \
        --output results/benchmark/Llama3-8B/combined/merged_summary.json

    # or pass the per-node logs/ dirs directly:
    python3 scripts/merge_node_benchmarks.py \
        --node-logs <chars_node1>/logs \
        --node-logs <chars_node2>/logs \
        --output combined/merged_summary.json \
        --rebuild-kb --data-dir data

The combined, re-keyed worker JSONs are written alongside the output as
``worker_{0..7}.json`` (or into ``--combined-logs-dir``) so
``analyze_benchmark_comparison.py`` can be pointed at that dir to aggregate
all 8 workers directly.
"""
from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path


# Required keys every worker summary must carry (same contract as
# merge_benchmarks.load_worker_result).
REQUIRED_WORKER_KEYS = ["success_rate", "total_successes", "total_rounds", "results"]

# Run-config metadata fields that merge_benchmarks.py drops but we preserve so
# the combined summary is self-describing.
CONFIG_META_KEYS = [
    "seed",
    "start_idx",
    "mutation_fallback_enabled",
    "max_fallback_rounds",
    "planner_temp_escalation",
    "cooperative_seeding",
    "cooperative_n",
    "planner_model",
    "generator_model",
    "target_model",
    "max_interactions",
]


def load_worker_result(path: str) -> dict:
    """Load and validate a single worker result file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in REQUIRED_WORKER_KEYS:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {path}")
    return data


def _expand_arg_paths(patterns: list[str], exist_desc: str) -> list[str]:
    """Bracket-safe glob expansion — identical contract to merge_benchmarks.py.

    A path that already resolves to an existing file/dir is used verbatim — we
    do NOT re-glob it. Benchmark output directories embed the dataset slice as a
    literal (e.g. "..._Llama3-[0:13024]_..."), and glob.glob() would misread
    those brackets as a POSIX character class, yielding zero matches even though
    the file exists. So: existing-path args pass through untouched; only
    non-existing args are treated as glob patterns.
    """
    expanded: list[str] = []
    for pattern in patterns:
        if os.path.exists(pattern):
            expanded.append(pattern)
            continue
        matched = glob_module.glob(pattern)
        if matched:
            expanded.extend(matched)
        else:
            print(f"[WARN] No {exist_desc} matched pattern: {pattern}")
    return expanded


def _collect_node_worker_files(node_dirs: list[str], node_logs: list[str]) -> list[tuple[int, Path]]:
    """Resolve each node's logs/ dir and return (node_index, logs_dir) pairs.

    Accepts either the <chars> run-dir (parent of logs/) via --node-dir, or the
    logs/ dir directly via --node-logs. Each entry is one node.
    """
    entries: list[tuple[int, Path]] = []
    idx = 0
    for d in node_dirs:
        run_dir = Path(d)
        logs_dir = run_dir / "logs"
        if not logs_dir.is_dir():
            # Allow the user to point straight at a dir whose worker_*.json live
            # alongside (older flat layout).
            if any(run_dir.glob("worker_*.json")):
                logs_dir = run_dir
            else:
                print(f"[WARN] --node-dir {d} has no logs/ subdir and no worker_*.json; skipping.")
                continue
        entries.append((idx, logs_dir))
        idx += 1
    for d in node_logs:
        logs_dir = Path(d)
        if not logs_dir.is_dir():
            print(f"[WARN] --node-logs {d} is not a directory; skipping.")
            continue
        entries.append((idx, logs_dir))
        idx += 1
    return entries


def _gather_and_rekey(
    node_entries: list[tuple[int, Path]],
    combined_logs_dir: Path,
) -> list[dict]:
    """Copy every node's worker_*.json into combined_logs_dir with unique global ids.

    Returns the list of loaded worker dicts (with patched metadata) in global-id
    order. Node k worker w → global id = k * workers_per_node + w. We infer
    workers_per_node from the first node's worker count; every node must match.
    """
    combined_logs_dir.mkdir(parents=True, exist_ok=True)

    # First pass: determine workers_per_node from the first node that has files.
    workers_per_node = 0
    for _, logs_dir in node_entries:
        n = len(list(logs_dir.glob("worker_*.json")))
        if n > 0:
            workers_per_node = n
            break
    if workers_per_node == 0:
        print("[ERROR] No worker_*.json files found in any node dir.")
        sys.exit(1)

    workers: list[dict] = []
    for node_idx, logs_dir in node_entries:
        node_files = sorted(logs_dir.glob("worker_*.json"))
        if len(node_files) != workers_per_node and node_files:
            print(f"[WARN] node {node_idx} has {len(node_files)} worker files; "
                  f"expected {workers_per_node} (from first node). Re-keying by "
                  f"file index within this node.")
        for w_idx, src in enumerate(node_files):
            global_id = node_idx * workers_per_node + w_idx
            data = load_worker_result(str(src))
            meta = data.setdefault("metadata", {})
            # Preserve original node-local id, then stamp the global id so the
            # combined summary's worker_summaries and the analysis script agree.
            meta["node_id"] = node_idx
            meta["node_worker_id"] = meta.get("worker_id", w_idx)
            meta["worker_id"] = global_id
            dst = combined_logs_dir / f"worker_{global_id}.json"
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            workers.append(data)
            print(f"  node {node_idx} worker {w_idx} -> {dst.name}")
    return workers


def _merge_counters(dict_a: dict, dict_b: dict) -> dict:
    """Sum two {key: number} dicts."""
    out = dict(dict_a)
    for k, v in dict_b.items():
        out[k] = out.get(k, 0) + v
    return out


def _merge_nested_counter(parent: dict, key: str, addition: dict) -> None:
    """parent[key] += addition (summing numeric values)."""
    parent.setdefault(key, {})
    for k, v in addition.items():
        parent[key][k] = parent[key].get(k, 0) + v


def _merge_per_type_stats(all_stats: list[dict]) -> dict:
    """Merge per_type_stats lists across workers.

    Each worker's per_type_stats is {defense_type: {total, leaks, extracts,
    verifys, access_granted, ...}}. Sum the numeric counters per defense_type.
    """
    merged: dict = {}
    for stats in all_stats:
        if not isinstance(stats, dict):
            continue
        for dtype, s in stats.items():
            if not isinstance(s, dict):
                continue
            tgt = merged.setdefault(dtype, {})
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    tgt[k] = tgt.get(k, 0) + v
                # else: skip non-numeric (none expected in per_type_stats)
    return merged


def _merge_failure_mode_stats(all_stats: list[dict]) -> dict:
    """Merge failure_mode_stats across workers ({mode: count})."""
    merged: dict = {}
    for stats in all_stats:
        if not isinstance(stats, dict):
            continue
        for mode, v in stats.items():
            if isinstance(v, (int, float)):
                merged[mode] = merged.get(mode, 0) + v
    return merged


def merge_node_workers(workers: list[dict], node_paths: list[str], output_path: str) -> dict:
    """Aggregate the (already re-keyed) worker dicts into one combined summary.

    Extends merge_benchmarks.merge_benchmarks with the full-schema fields that
    script drops: mutation_fallback_*, mutation_fallback_diagnostics,
    per_type_stats, failure_mode_stats, models, avg_verified_rank, and the
    run-config metadata.
    """
    if not workers:
        print("[ERROR] No worker results to merge.")
        sys.exit(1)

    num_workers = len(workers)
    print(f"\n[MERGE] Combining {num_workers} worker results across nodes...")

    # --- shared counters (same as merge_benchmarks.py) ---
    total_rounds = sum(w["total_rounds"] for w in workers)
    total_successes = sum(w["total_successes"] for w in workers)
    total_success_exact = sum(w.get("total_success_exact", 0) for w in workers)
    total_success_extractor = sum(w.get("total_success_extractor", 0) for w in workers)
    total_access_granted = sum(w.get("total_access_granted", 0) for w in workers)
    total_top1 = sum(w.get("top1_success", 0) for w in workers)
    total_top3 = sum(w.get("top3_success", 0) for w in workers)
    total_top5 = sum(w.get("top5_success", 0) for w in workers)
    total_verified = sum(w.get("verified_success", 0) for w in workers)

    # --- full-schema counters merge_benchmarks drops ---
    total_fb_triggered = sum(w.get("mutation_fallback_triggered", 0) for w in workers)
    total_fb_successes = sum(w.get("mutation_fallback_successes", 0) for w in workers)

    fb_variant_total = 0
    fb_no_op_total = 0
    fb_mutator_counts: dict = {}
    fb_no_op_counts: dict = {}
    fb_winning_mutator_counts: dict = {}
    for w in workers:
        diag = w.get("mutation_fallback_diagnostics", {}) or {}
        fb_variant_total += diag.get("variant_total", 0)
        fb_no_op_total += diag.get("no_op_total", 0)
        fb_mutator_counts = _merge_counters(fb_mutator_counts, diag.get("mutator_counts", {}))
        fb_no_op_counts = _merge_counters(fb_no_op_counts, diag.get("no_op_counts", {}))
        fb_winning_mutator_counts = _merge_counters(fb_winning_mutator_counts, diag.get("winning_mutator_counts", {}))

    # weighted avg_verified_rank
    vrank_sum = 0.0
    vrank_count = 0
    for w in workers:
        n = w.get("verified_success", 0)
        r = w.get("avg_verified_rank", 0)
        if n and r is not None:
            vrank_sum += r * n
            vrank_count += n
    avg_verified_rank = vrank_sum / vrank_count if vrank_count else 0.0

    per_type_stats = _merge_per_type_stats([w.get("per_type_stats", {}) for w in workers])
    failure_mode_stats = _merge_failure_mode_stats([w.get("failure_mode_stats", {}) for w in workers])

    # --- per-round results (append global worker_id) ---
    all_results = []
    for w in workers:
        worker_id = w.get("metadata", {}).get("worker_id", 0)
        for r in w.get("results", []):
            merged_round = dict(r)
            merged_round["worker_id"] = worker_id
            all_results.append(merged_round)

    # --- aggregate rates ---
    success_rate = total_successes / total_rounds if total_rounds > 0 else 0.0
    defense_rate = 1.0 - success_rate

    # weighted avg attempts on success
    att_sum = 0.0
    att_count = 0
    for w in workers:
        avg = w.get("avg_attempts_on_success")
        c = w.get("total_successes", 0)
        if avg is not None and c > 0 and avg != float("inf"):
            att_sum += avg * c
            att_count += c
    avg_attempts = att_sum / att_count if att_count > 0 else float("inf")

    # extractor metrics
    tp = sum(w.get("extractor_metrics", {}).get("true_positive", 0) for w in workers)
    fp = sum(w.get("extractor_metrics", {}).get("false_positive", 0) for w in workers)
    fn = sum(w.get("extractor_metrics", {}).get("false_negative", 0) for w in workers)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # strategy stats
    combined_strategy_stats: dict = {}
    for w in workers:
        for strat, stats in (w.get("strategy_stats", {}) or {}).items():
            tgt = combined_strategy_stats.setdefault(strat, {"attempts": 0, "successes": 0, "total_score": 0.0})
            tgt["attempts"] += stats.get("attempts", 0)
            tgt["successes"] += stats.get("successes", 0)
            tgt["total_score"] += stats.get("total_score", 0.0)
    for s in combined_strategy_stats.values():
        s["avg_score"] = s["total_score"] / s["attempts"] if s["attempts"] > 0 else 0.0
        s["success_rate"] = s["successes"] / s["attempts"] if s["attempts"] > 0 else 0.0

    # --- metadata: preserve config fields + node provenance ---
    first_meta = workers[0].get("metadata", {}) or {}
    combined_meta: dict = {
        "timestamp": datetime.now().isoformat(),
        "target_model": first_meta.get("target_model", "Llama-3-8B-Instruct"),
        "planner_model": first_meta.get("planner_model"),
        "generator_model": first_meta.get("generator_model"),
        "n_rounds": total_rounds,
        "max_interactions": first_meta.get("max_interactions", 20),
        "num_workers": num_workers,
        "worker_ids": [w.get("metadata", {}).get("worker_id", i) for i, w in enumerate(workers)],
        "merged_from": node_paths,
        "merge_type": "cross_node",
    }
    # Preserve the run-config fields so the combined summary is self-describing.
    for k in CONFIG_META_KEYS:
        if k in first_meta and k not in combined_meta:
            combined_meta[k] = first_meta[k]

    # models block (identical across workers in a run)
    models = workers[0].get("models")

    merged: dict = {
        "metadata": combined_meta,
        "success_rate": success_rate,
        "defense_rate": defense_rate,
        "avg_attempts_on_success": avg_attempts,
        "total_successes": total_successes,
        # full-schema: mutation fallback (merge_benchmarks.py drops these)
        "mutation_fallback_triggered": total_fb_triggered,
        "mutation_fallback_successes": total_fb_successes,
        "mutation_fallback_diagnostics": {
            "variant_total": fb_variant_total,
            "no_op_total": fb_no_op_total,
            "no_op_rate": round(fb_no_op_total / fb_variant_total, 4) if fb_variant_total else 0.0,
            "mutator_counts": dict(sorted(fb_mutator_counts.items())),
            "no_op_counts": dict(sorted(fb_no_op_counts.items())),
            "winning_mutator_counts": dict(sorted(fb_winning_mutator_counts.items())),
        },
        "total_success_exact": total_success_exact,
        "total_success_extractor": total_success_extractor,
        "total_access_granted": total_access_granted,
        "total_rounds": total_rounds,
        "top1_success": total_top1,
        "top3_success": total_top3,
        "top5_success": total_top5,
        "verified_success": total_verified,
        "avg_verified_rank": avg_verified_rank,
        # full-schema: per-type + failure-mode (merge_benchmarks.py drops these)
        "per_type_stats": per_type_stats,
        "failure_mode_stats": failure_mode_stats,
        "extractor_metrics": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "strategy_stats": combined_strategy_stats,
        "results": all_results,
        "worker_summaries": [
            {
                "worker_id": w.get("metadata", {}).get("worker_id", i),
                "node_id": w.get("metadata", {}).get("node_id"),
                "node_worker_id": w.get("metadata", {}).get("node_worker_id"),
                "rounds": w["total_rounds"],
                "successes": w["total_successes"],
                "success_rate": w["success_rate"],
            }
            for i, w in enumerate(workers)
        ],
    }
    if models is not None:
        merged["models"] = models

    # write
    out = Path(output_path)
    if out.is_dir():
        out = out / "merged_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    # --- print ---
    print(f"\n{'=' * 60}")
    print(f"📊 MERGED CROSS-NODE BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Workers (total):  {num_workers}")
    print(f"  Total Rounds:     {total_rounds}")
    print(f"  Success Rate:      {success_rate * 100:.1f}%")
    print(f"  Defense Rate:      {defense_rate * 100:.1f}%")
    print(f"  Avg Attempts:      {avg_attempts:.1f}" if avg_attempts != float("inf")
          else "  Avg Attempts:      N/A (no successes)")
    print(f"  Total Successes:   {total_successes}/{total_rounds}")
    print(f"  Generator Hit:     {total_success_exact}/{total_rounds}")
    print(f"  Extractor Hit:      {total_success_extractor}/{total_rounds}")
    print(f"  Access Granted:     {total_access_granted}/{total_rounds}")
    print(f"\n  Top-1 / Top-3 / Top-5 / Verified: "
          f"{total_top1} / {total_top3} / {total_top5} / {total_verified}")
    print(f"\n  Mutation Fallback: triggered={total_fb_triggered} "
          f"successes={total_fb_successes}")
    if fb_variant_total:
        print(f"  Fallback variants: {fb_variant_total} "
              f"(no-op {fb_no_op_total} = {fb_no_op_total/fb_variant_total*100:.1f}%)")
    print(f"\n  Extractor P/R/F1:   {precision:.2%} / {recall:.2%} / {f1:.2%}")
    print(f"{'=' * 60}")
    print(f"\n[JSON] Combined summary saved to: {out}")
    print(f"[JSON] Re-keyed worker JSONs in:   {out.parent}")
    print(f"       (point analyze_benchmark_comparison.py at this dir)")
    return merged


def _maybe_rebuild_kb(
    node_dirs: list[str],
    data_dir: str,
    rebuild_results_dir: str | None,
    verbose: bool,
) -> None:
    """Invoke scripts.rebuild_kb_rag.rebuild across both nodes' run trees.

    The oracle miner rglobs run_*.json under results_dir, so pointing at a
    common parent that contains both nodes' runs/ subtrees mines everything in
    one pass. If the node dirs share a parent (the usual case — both are
    <chars> siblings under results/benchmark/<model>/), we use that parent.
    Otherwise the user must pass --rebuild-results-dir explicitly.
    """
    try:
        # Late import so a missing dependency doesn't break the plain merge.
        from scripts.rebuild_kb_rag import rebuild as _rebuild
    except Exception as exc:
        print(f"[REBUILD] could not import scripts.rebuild_kb_rag: {exc}", file=sys.stderr)
        return

    results_dir = rebuild_results_dir
    if results_dir is None:
        # Derive the common parent of the node <chars> dirs.
        parents = [Path(d).resolve().parent for d in node_dirs]
        common = os.path.commonpath([str(p) for p in parents]) if parents else ""
        if common and Path(common).is_dir():
            results_dir = common
        else:
            print("[REBUILD] could not infer a common results dir for the nodes; "
                  "pass --rebuild-results-dir explicitly. Skipping KB rebuild.",
                  file=sys.stderr)
            return

    print(f"\n[REBUILD] rebuilding KB/oracle/RAG with --results-dir {results_dir}")
    report = _rebuild(data_dir=data_dir, results_dir=results_dir, verbose=verbose)
    print("[REBUILD] summary:")
    print(json.dumps(report, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge multi-NODE benchmark results into one combined summary. "
                    "Each node is a 4-GPU run that produced logs/worker_{0..3}.json; "
                    "this script re-keys them to unique global ids, aggregates the "
                    "full worker schema, and optionally rebuilds the KB/RAG across "
                    "both nodes' run trees."
    )
    ap.add_argument("--node-dir", action="append", default=[],
                    help="A node's <chars> run-dir (parent of logs/ and runs/). "
                         "Repeat once per node: --node-dir <node1> --node-dir <node2>.")
    ap.add_argument("--node-logs", action="append", default=[],
                    help="A node's logs/ dir directly (contains worker_*.json). "
                         "Repeat once per node.")
    ap.add_argument("--output", "-o", required=True,
                    help="Output path for the combined merged_summary.json "
                         "(a dir is resolved to merged_summary.json inside it).")
    ap.add_argument("--combined-logs-dir", default=None,
                    help="Where to write the re-keyed worker_*.json. Defaults to "
                         "the output file's parent dir so analyze_benchmark_"
                         "comparison.py can be pointed at it.")
    ap.add_argument("--rebuild-kb", action="store_true",
                    help="After merging, run scripts.rebuild_kb_rag across both "
                         "nodes' run trees (oracle miner sees all run_*.json).")
    ap.add_argument("--data-dir", default="data",
                    help="data dir for rebuild (default: data).")
    ap.add_argument("--rebuild-results-dir", default=None,
                    help="Explicit results dir for the KB rebuild (default: "
                         "common parent of the --node-dir values).")
    args = ap.parse_args()

    # bracket-safe expansion (output dirs embed [start:rounds] brackets)
    node_dirs = _expand_arg_paths(args.node_dir, "node-dir")
    node_logs = _expand_arg_paths(args.node_logs, "node-logs")
    if not node_dirs and not node_logs:
        print("[ERROR] supply at least one --node-dir or --node-logs.")
        return 2

    node_entries = _collect_node_worker_files(node_dirs, node_logs)
    if not node_entries:
        print("[ERROR] no resolvable node directories found.")
        return 2

    out = Path(args.output)
    combined_logs_dir = Path(args.combined_logs_dir) if args.combined_logs_dir else out.parent
    combined_logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MERGE] nodes: {len(node_entries)}; combined logs -> {combined_logs_dir}")
    workers = _gather_and_rekey(node_entries, combined_logs_dir)

    # node_paths for provenance (use the resolved node dirs when available).
    provenance = node_dirs if node_dirs else node_logs
    merge_node_workers(workers, provenance, args.output)

    if args.rebuild_kb:
        # rebuild needs the <chars> run-dirs (parents of runs/); if only
        # --node-logs was given, fall back to each logs dir's parent.
        rb_nodes = node_dirs if node_dirs else [str(Path(d).parent) for d in node_logs]
        _maybe_rebuild_kb(rb_nodes, args.data_dir, args.rebuild_results_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
