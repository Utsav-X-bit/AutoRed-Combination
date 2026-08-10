#!/usr/bin/env python3
"""
AutoRed benchmark analyzer.

Produces a single-run analysis report (markdown) from a benchmark run's
worker_*.json summaries + worker_*.log runtime traces. NO OPT-vs-NONOPT
comparison — one column of the current run only.

It is designed to run *itself* once a benchmark finishes: in watch mode it
polls a run directory until either ``merged_summary.json`` appears (success)
or an OOM / fatal traceback is seen in a worker log (failure), then writes
the report.

Usage
-----
  # Analyze a completed run immediately:
  python3 scripts/analyze_benchmark_comparison.py \\
      --run-dir results/benchmark/<model_id>/<run_dir>

  # Watch a still-running run and auto-run the moment it completes (or OOMs):
  python3 scripts/analyze_benchmark_comparison.py \\
      --run-dir results/benchmark/<model_id>/<run_dir> --watch

Layout (results_layout v3):
    results/benchmark/<model_id>/<chars>/logs/worker_{0..3}.json   (summaries)
    results/benchmark/<model_id>/<chars>/logs/worker_{0..3}.log    (vLLM traces)
    results/benchmark/<model_id>/<chars>/logs/merged_summary.json  (4-worker merge)
    results/benchmark/<model_id>/<chars>/logs/analysis.md         <-- written here

The report is written to ``<run-dir>/logs/analysis.md`` regardless of how the
run ended. On a crash it records the OOM / failure diagnostics so the run is
still analyzable.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# --- constants ---------------------------------------------------------------

MERGED_NAME = "merged_summary.json"
# Fatal markers in worker logs (only consulted when worker JSONs are absent).
FAILURE_MARKERS = [
    "torch.OutOfMemoryError",
    "OutOfMemoryError: CUDA out of memory",
    "RuntimeError: CUDA error",
    "FATAL",
    "Traceback (most recent call last)",
]
# NCCL benign-exit line we must NOT mistake for a failure.
NCCL_WARN_RE = re.compile(r"destroy_process_group\(\) was not called")

WORKER_JSON_RE = re.compile(r"worker_(\d+)\.json$")
WORKER_LOG_RE = re.compile(r"worker_(\d+)\.log$")

# vLLM log-line patterns for runtime extraction.
VLLM_TS_RE = re.compile(r"INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\b")
MODEL_LOAD_RE = re.compile(r"Model loading took\s+([\d.]+)\s+GiB and\s+([\d.]+)\s+seconds")
INIT_ENGINE_RE = re.compile(r"init engine .* took\s+([\d.]+)\s+seconds")
GRAPH_CAPTURE_RE = re.compile(r"Graph capturing finished in\s+(\d+)\s+secs")
KV_BLOCKS_RE = re.compile(r"# cuda blocks:\s+(\d+),\s*# CPU blocks:\s+(\d+)")
GPU_MEM_LINE_RE = re.compile(
    r"the current vLLM instance can use total_gpu_memory \(([\d.]+)GiB\)\s*"
    r"x gpu_memory_utilization \(([\d.]+)\)\s*=\s*([\d.]+)GiB"
)
KV_DETAIL_RE = re.compile(
    r"model weights take\s+([\d.]+)GiB;.*reserved for KV Cache is\s+([\d.]+)GiB"
)
WEIGHTS_TOOK_RE = re.compile(r"Loading weights took\s+([\d.]+)\s+seconds")
BENCH_BANNER_RE = re.compile(r"🏁 BENCHMARK.*?(\d+)\s+Rounds.*?(\d+)\s+Max Interactions")
RESULTS_BANNER_RE = re.compile(r"📊 BENCHMARK RESULTS")
SUMMARY_SAVED_RE = re.compile(r"\[JSON\] Worker summary saved to")

MUTATORS = ["EN", "PI", "SR", "TL"]
SUCCESS_PATHS = ["gt_leak", "access_granted", "extractor", "verified", "fallback", "none"]


# --- result loading ---------------------------------------------------------


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_worker_summaries(run_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(run_dir.glob("worker_*.json")):
        m = WORKER_JSON_RE.search(p.name)
        if not m:
            continue
        d = _load_json(p)
        if d is not None:
            out[int(m.group(1))] = d
    return out


def load_merged(run_dir: Path):
    # Results-layout v3 keeps merged_summary.json under <run_dir>/logs/; the
    # older flat-layout launchers (autored_benchmark_4gpu.sh,
    # benchmark_multigpu.sh) write it directly under <run_dir>. Support both so
    # auto-analysis works regardless of which launcher produced the run.
    nested = run_dir / "logs" / MERGED_NAME
    if nested.exists():
        return _load_json(nested)
    return _load_json(run_dir / MERGED_NAME)


# --- worker log parsing -----------------------------------------------------


def _parse_vllm_ts(line: str):
    m = VLLM_TS_RE.search(line)
    if not m:
        return None
    year = datetime.now().year
    try:
        return datetime.strptime(f"{year} {m.group(1)}", "%Y %m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_worker_log(log_path: Path) -> dict:
    info: dict = {
        "log_path": str(log_path),
        "first_ts": None,
        "last_ts": None,
        "n_lines": 0,
        "victim_load_s": None,
        "shared_load_s": None,
        "init_engine_s": None,
        "graph_capture_s": None,
        "weights_took_s": None,
        "gpu_total_gib": None,
        "gpu_util": None,
        "gpu_budget_gib": None,
        "weights_gib": None,
        "kv_reserved_gib": None,
        "cuda_blocks": None,
        "cpu_blocks": None,
        "saw_results_banner": False,
        "saw_summary_saved": False,
        "saw_banner": False,
        "banner_rounds": None,
        "banner_max_inter": None,
        "failure_lines": [],
        "oom": False,
        "oom_alloc_mib": None,
        "oom_free_mib": None,
        "oom_total_gib": None,
    }
    if not log_path.exists():
        return info
    oom_alloc_re = re.compile(r"Tried to allocate\s+([\d.]+)\s*MiB")
    oom_free_re = re.compile(r"of which\s+([\d.]+)\s*MiB\s+is free")
    oom_total_re = re.compile(r"total capacity of\s+([\d.]+)\s*GiB")
    model_loads: list[float] = []
    init_engines: list[float] = []
    weights_took: list[float] = []
    graph_caps: list[int] = []
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                info["n_lines"] += 1
                ts = _parse_vllm_ts(line)
                if ts is not None:
                    if info["first_ts"] is None:
                        info["first_ts"] = ts
                    info["last_ts"] = ts
                m = MODEL_LOAD_RE.search(line)
                if m:
                    model_loads.append(float(m.group(2)))
                m = INIT_ENGINE_RE.search(line)
                if m:
                    init_engines.append(float(m.group(1)))
                m = GRAPH_CAPTURE_RE.search(line)
                if m:
                    graph_caps.append(int(m.group(1)))
                m = WEIGHTS_TOOK_RE.search(line)
                if m:
                    weights_took.append(float(m.group(1)))
                m = KV_BLOCKS_RE.search(line)
                if m:
                    info["cuda_blocks"] = int(m.group(1))
                    info["cpu_blocks"] = int(m.group(2))
                m = GPU_MEM_LINE_RE.search(line)
                if m:
                    info["gpu_total_gib"] = float(m.group(1))
                    info["gpu_util"] = float(m.group(2))
                    info["gpu_budget_gib"] = float(m.group(3))
                m = KV_DETAIL_RE.search(line)
                if m:
                    info["weights_gib"] = float(m.group(1))
                    info["kv_reserved_gib"] = float(m.group(2))
                if BENCH_BANNER_RE.search(line):
                    info["saw_banner"] = True
                    bm = BENCH_BANNER_RE.search(line)
                    if bm:
                        info["banner_rounds"] = int(bm.group(1))
                        info["banner_max_inter"] = int(bm.group(2))
                if RESULTS_BANNER_RE.search(line):
                    info["saw_results_banner"] = True
                if SUMMARY_SAVED_RE.search(line):
                    info["saw_summary_saved"] = True
                if "torch.OutOfMemoryError" in line or "OutOfMemoryError" in line:
                    info["oom"] = True
                    a = oom_alloc_re.search(line)
                    fr = oom_free_re.search(line)
                    tot = oom_total_re.search(line)
                    if a:
                        info["oom_alloc_mib"] = float(a.group(1))
                    if fr:
                        info["oom_free_mib"] = float(fr.group(1))
                    if tot:
                        info["oom_total_gib"] = float(tot.group(1))
                    info["failure_lines"].append(line.strip()[:300])
                elif any(mark in line for mark in FAILURE_MARKERS) and not NCCL_WARN_RE.search(line):
                    if not info["oom"]:
                        info["failure_lines"].append(line.strip()[:300])
    except OSError:
        pass
    if model_loads:
        info["victim_load_s"] = model_loads[0]
        if len(model_loads) > 1:
            info["shared_load_s"] = model_loads[1]
    if init_engines:
        info["init_engine_s"] = init_engines[0]
    if graph_caps:
        info["graph_capture_s"] = graph_caps[0]
    if weights_took:
        info["weights_took_s"] = weights_took[0]
    return info


def worker_span_seconds(info: dict):
    if info["first_ts"] and info["last_ts"]:
        return (info["last_ts"] - info["first_ts"]).total_seconds()
    return None


# --- aggregation ------------------------------------------------------------


def aggregate_results(summaries: dict[int, dict]) -> dict:
    agg: dict = {
        "n_workers": len(summaries),
        "worker_ids": sorted(summaries),
        "total_rounds": 0,
        "total_successes": 0,
        "total_success_exact": 0,
        "total_success_extractor": 0,
        "total_access_granted": 0,
        "top1_success": 0,
        "top3_success": 0,
        "top5_success": 0,
        "verified_success": 0,
        "fallback_triggered": 0,
        "fallback_successes": 0,
        "success_path_counts": collections.Counter(),
        "winning_mutator_counts": collections.Counter(),
        "mutator_counts": collections.Counter(),
        "strategy_counts": collections.Counter(),
        "attempts_on_success": [],
        "per_worker": [],
        "extractor_tp": 0,
        "extractor_fp": 0,
        "extractor_fn": 0,
        "metadata": None,
    }
    seen_meta = None
    for wid in sorted(summaries):
        d = summaries[wid]
        if seen_meta is None:
            seen_meta = d.get("metadata")
        agg["total_rounds"] += d.get("total_rounds", 0)
        agg["total_successes"] += d.get("total_successes", 0)
        agg["total_success_exact"] += d.get("total_success_exact", 0)
        agg["total_success_extractor"] += d.get("total_success_extractor", 0)
        agg["total_access_granted"] += d.get("total_access_granted", 0)
        agg["top1_success"] += d.get("top1_success", 0)
        agg["top3_success"] += d.get("top3_success", 0)
        agg["top5_success"] += d.get("top5_success", 0)
        agg["verified_success"] += d.get("verified_success", 0)
        agg["fallback_triggered"] += d.get("mutation_fallback_triggered", 0)
        agg["fallback_successes"] += d.get("mutation_fallback_successes", 0)
        em = d.get("extractor_metrics", {})
        agg["extractor_tp"] += em.get("true_positive", 0)
        agg["extractor_fp"] += em.get("false_positive", 0)
        agg["extractor_fn"] += em.get("false_negative", 0)
        diag = d.get("mutation_fallback_diagnostics", {})
        agg["winning_mutator_counts"].update(diag.get("winning_mutator_counts", {}))
        agg["mutator_counts"].update(diag.get("mutator_counts", {}))
        for r in d.get("results", []):
            sp = r.get("success_path", "none")
            agg["success_path_counts"][sp] += 1
            wm = r.get("winning_mutator")
            if wm is not None:
                agg["winning_mutator_counts"][wm] += 1
            agg["strategy_counts"][r.get("best_strategy", "unknown")] += 1
            if r.get("success"):
                agg["attempts_on_success"].append(r.get("attempts", 0))
        agg["per_worker"].append({
            "worker_id": wid,
            "rounds": d.get("total_rounds", 0),
            "successes": d.get("total_successes", 0),
            "success_rate": d.get("success_rate", 0.0),
            "defense_rate": d.get("defense_rate", 0.0),
            "fallback_triggered": d.get("mutation_fallback_triggered", 0),
            "fallback_successes": d.get("mutation_fallback_successes", 0),
        })
    agg["metadata"] = seen_meta
    n = agg["total_rounds"] or 1
    agg["success_rate"] = agg["total_successes"] / n
    agg["defense_rate"] = 1.0 - agg["success_rate"]
    if agg["attempts_on_success"]:
        agg["avg_attempts_on_success"] = sum(agg["attempts_on_success"]) / len(
            agg["attempts_on_success"]
        )
    else:
        agg["avg_attempts_on_success"] = 0.0
    tp, fp, fn = agg["extractor_tp"], agg["extractor_fp"], agg["extractor_fn"]
    agg["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    agg["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    agg["f1"] = (
        2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])
        if (agg["precision"] + agg["recall"]) else 0.0
    )
    return agg


# --- helpers ----------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_seconds(s):
    if s is None:
        return "—"
    if s >= 3600:
        return f"{s/3600:.2f}h"
    if s >= 60:
        return f"{s/60:.2f}m"
    return f"{s:.1f}s"


# --- the report -------------------------------------------------------------


def build_report(run_dir: Path, summaries, logs, merged, run_status: str) -> str:
    lines: list[str] = []
    agg = aggregate_results(summaries)

    def section(title: str):
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")

    # ---- header -----------------------------------------------------------
    meta = agg["metadata"] or {}
    target = meta.get("target_model", run_dir.parent.name)
    lines.append(f"# Benchmark Analysis — {target}")
    lines.append("")
    lines.append(f"- **Run dir:** `{run_dir}`")
    lines.append(f"- **Status:** {run_status}")
    lines.append(f"- **Workers:** {agg['n_workers']} (ids {agg['worker_ids']})")
    lines.append(f"- **Total rounds:** {agg['total_rounds']}")
    lines.append(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    if meta:
        lines.append("**Config:**")
        for k in [
            "seed", "start_idx", "mutation_fallback_enabled",
            "max_fallback_rounds", "cooperative_seeding", "cooperative_n",
            "planner_temp_escalation", "max_interactions", "n_rounds",
        ]:
            if k in meta:
                lines.append(f"- {k}: `{meta[k]}`")
        pm = meta.get("planner_model", "")
        if pm:
            lines.append(f"- planner_model: `{os.path.basename(pm)}`")

    # ---- run status / crash diagnostics (FIRST) ---------------------------
    if run_status != "complete":
        section("Run Status & Crash Diagnostics")
        lines.append(f"The run ended with status **{run_status}**.")
        any_oom = any(l.get("oom") for l in logs.values())
        if any_oom:
            lines.append("")
            lines.append("### OOM details (per worker)")
            lines.append("")
            lines.append("| Worker | OOM | Alloc (MiB) | Free (MiB) | Total (GiB) | Last vLLM ts |")
            lines.append("|---|---|---|---|---|---|")
            for wid in sorted(logs):
                l = logs[wid]
                if not l["oom"]:
                    continue
                lines.append(
                    f"| w{wid} | yes | {l['oom_alloc_mib'] or '—'} | "
                    f"{l['oom_free_mib'] or '—'} | {l['oom_total_gib'] or '—'} | "
                    f"{l['last_ts'].isoformat() if l['last_ts'] else '—'} |"
                )
            lines.append("")
            lines.append(
                "> The run crashed before all workers wrote summaries, so the "
                "metric tables below are computed from whatever worker_*.json "
                "did get written (partial). Treat numbers as incomplete."
            )
        if any(l["failure_lines"] for l in logs.values()):
            lines.append("")
            lines.append("### First failure line per worker")
            lines.append("")
            for wid in sorted(logs):
                if logs[wid]["failure_lines"]:
                    lines.append(f"- **w{wid}:** `{logs[wid]['failure_lines'][0]}`")

    # ---- Table 1: Runtime --------------------------------------------------
    section("1. Runtime")
    lines.append("Per-worker wall-clock span (first→last vLLM log timestamp), "
                 "model-load times, and GPU memory profile.")
    lines.append("")
    lines.append("| Worker | Span | Victim load | Shared load | init engine | Graph capture | GPU util | KV blocks | KV reserved (GiB) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    spans = []
    for wid in sorted(logs):
        l = logs[wid]
        sp = worker_span_seconds(l)
        if sp is not None:
            spans.append(sp)
        lines.append(
            f"| w{wid} | {_fmt_seconds(sp)} | "
            f"{_fmt_seconds(l['victim_load_s'])} | "
            f"{_fmt_seconds(l['shared_load_s'])} | "
            f"{_fmt_seconds(l['init_engine_s'])} | "
            f"{_fmt_seconds(l['graph_capture_s'])} | "
            f"{l['gpu_util'] or '—'} | {l['cuda_blocks'] or '—'} | "
            f"{l['kv_reserved_gib'] or '—'} |"
        )
    if spans:
        lines.append(
            f"| **max (wall-clock)** | **{_fmt_seconds(max(spans))}** | | | | | | | |"
        )
        lines.append(
            f"| **sum** | **{_fmt_seconds(sum(spans))}** | | | | | | | |"
        )

    # ---- Table 2: Core metrics --------------------------------------------
    section("2. Core Metrics")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    rows_pct = ["success_rate", "defense_rate"]
    rows_int = ["total_successes", "avg_attempts_on_success", "top1_success",
                "top3_success", "top5_success", "verified_success",
                "total_success_exact", "total_success_extractor",
                "total_access_granted", "total_rounds"]
    for name in rows_pct:
        lines.append(f"| {name} | {_fmt_pct(agg[name])} |")
    for name in rows_int:
        v = agg[name]
        lines.append(f"| {name} | {v:.2f} |" if isinstance(v, float) else f"| {name} | {v} |")

    # ---- Table 3: Success path --------------------------------------------
    section("3. Success Path")
    lines.append(
        "How each scenario resolved. `none` = failed (no success path). "
        "Counts across all workers."
    )
    lines.append("")
    lines.append("| Path | Count |")
    lines.append("|---|---|")
    # include any observed path beyond the canonical list
    all_paths = SUCCESS_PATHS + [p for p in agg["success_path_counts"] if p not in SUCCESS_PATHS]
    for sp in all_paths:
        lines.append(f"| {sp} | {agg['success_path_counts'].get(sp, 0)} |")

    # ---- Table 4: Winning mutator -----------------------------------------
    section("4. Winning Mutator")
    lines.append(
        "Which mutator produced the variant that broke a fallback-resistant "
        "scenario. `None` = won without mutation fallback. "
        "Wins (winning_mutator) and total draws (mutator_counts)."
    )
    lines.append("")
    lines.append("| Mutator | Wins | Draws |")
    lines.append("|---|---|---|")
    all_muts = sorted(set(MUTATORS) | set(agg["winning_mutator_counts"]) |
                      set(agg["mutator_counts"]))
    for m in all_muts:
        lines.append(
            f"| {m} | {agg['winning_mutator_counts'].get(m, 0)} | "
            f"{agg['mutator_counts'].get(m, 0)} |"
        )
    lines.append(f"| **fallback triggered** | {agg['fallback_triggered']} | |")
    lines.append(f"| **fallback successes** | {agg['fallback_successes']} | |")
    if agg["fallback_triggered"]:
        rate = agg["fallback_successes"] / agg["fallback_triggered"] * 100
        lines.append(f"| **fallback success rate** | {rate:.1f}% | |")

    # ---- Table 5: Precision / Recall / F1 ---------------------------------
    section("5. Extractor Precision / Recall / F1")
    lines.append("Extractor (LM judge) confusion-based metrics, run-level.")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| precision | {_fmt_pct(agg['precision'])} |")
    lines.append(f"| recall | {_fmt_pct(agg['recall'])} |")
    lines.append(f"| f1 | {_fmt_pct(agg['f1'])} |")
    lines.append(f"| true_positive | {agg['extractor_tp']} |")
    lines.append(f"| false_positive | {agg['extractor_fp']} |")
    lines.append(f"| false_negative | {agg['extractor_fn']} |")

    # ---- Table 6: Per-worker success --------------------------------------
    section("6. Per-Worker Success")
    lines.append("| Worker | Rounds | Successes | succ% | fb triggered | fb successes |")
    lines.append("|---|---|---|---|---|---|")
    for pw in agg["per_worker"]:
        lines.append(
            f"| w{pw['worker_id']} | {pw['rounds']} | {pw['successes']} | "
            f"{_fmt_pct(pw['success_rate'])} | {pw['fallback_triggered']} | "
            f"{pw['fallback_successes']} |"
        )

    # ---- Extra: strategy distribution -------------------------------------
    section("7. Winning Strategy Distribution")
    lines.append("`best_strategy` across all scenarios (which attack template won).")
    lines.append("")
    all_strats = sorted(agg["strategy_counts"])
    if all_strats:
        lines.append("| Strategy | Count |")
        lines.append("|---|---|")
        for s in all_strats:
            lines.append(f"| {s} | {agg['strategy_counts'][s]} |")
    else:
        lines.append("_No strategy data._")

    # ---- Extra: fallback diagnostics --------------------------------------
    section("8. Mutation Fallback Diagnostics")
    diag_lines = []
    for wid in sorted(summaries):
        d = summaries[wid]
        diag = d.get("mutation_fallback_diagnostics", {})
        diag_lines.append(
            f"- w{wid}: triggered={d.get('mutation_fallback_triggered',0)}, "
            f"successes={d.get('mutation_fallback_successes',0)}, "
            f"variants={diag.get('variant_total',0)}, "
            f"no-op_rate={diag.get('no_op_rate',0):.2%}, "
            f"mutators={diag.get('mutator_counts',{})}, "
            f"winners={diag.get('winning_mutator_counts',{})}"
        )
    if diag_lines:
        lines.extend(diag_lines)
    else:
        lines.append("_No fallback diagnostics available._")

    # ---- Verdict ----------------------------------------------------------
    section("9. Verdict")
    if agg["total_rounds"]:
        lines.append(
            f"- **success_rate**: {_fmt_pct(agg['success_rate'])} "
            f"({agg['total_successes']}/{agg['total_rounds']})."
        )
        lines.append(f"- **defense_rate**: {_fmt_pct(agg['defense_rate'])}.")
        lines.append(
            f"- **top1_success**: {agg['top1_success']} | "
            f"**verified_success**: {agg['verified_success']} | "
            f"**extractor_success**: {agg['total_success_extractor']} | "
            f"**access_granted**: {agg['total_access_granted']}."
        )
        lines.append(
            f"- **extractor F1**: {_fmt_pct(agg['f1'])} "
            f"(P={_fmt_pct(agg['precision'])}, R={_fmt_pct(agg['recall'])})."
        )
        if agg["fallback_triggered"]:
            rate = agg["fallback_successes"] / agg["fallback_triggered"] * 100
            lines.append(
                f"- **fallback**: {agg['fallback_successes']}/"
                f"{agg['fallback_triggered']} triggered succeeded ({rate:.1f}%)."
            )
        else:
            lines.append("- **fallback**: never triggered.")
        if run_status != "complete":
            lines.append(
                f"- ⚠️ Run status is **{run_status}** — figures above are from "
                "partial worker summaries and are not final."
            )
    else:
        lines.append("_No completed rounds to summarize (run crashed before data)._")

    lines.append("")
    return "\n".join(lines)


# --- completion detection ---------------------------------------------------


def run_status(run_dir: Path, summaries, logs) -> str:
    # merged_summary.json lives under logs/ (results-layout v3) or directly
    # under run_dir (older flat layout) — accept either.
    merged = (run_dir / "logs" / MERGED_NAME).exists() or (run_dir / MERGED_NAME).exists()
    n_jsons = len(summaries)
    if merged and n_jsons >= 1:
        return "complete"
    any_oom = any(l.get("oom") for l in logs.values())
    any_failure = any(l["failure_lines"] for l in logs.values())
    if any_oom:
        return "crashed (OOM)"
    if any_failure and n_jsons == 0:
        return "crashed"
    if n_jsons >= 1:
        return "partial"
    return "running"


# --- main -------------------------------------------------------------------


def analyze(run_dir: Path, write: bool = True) -> str:
    run_dir = run_dir.resolve()
    # Data dir: results-layout v3 keeps worker_*.json + merged_summary.json under
    # <run_dir>/logs/; older flat-layout launchers keep worker_*.json +
    # merged_summary.json directly under <run_dir>. Resolve which holds the data.
    logs_subdir = run_dir / "logs"
    has_logs_subdir = logs_subdir.is_dir() and any(logs_subdir.glob("worker_*.json"))
    data_dir = logs_subdir if has_logs_subdir else run_dir

    summaries = load_worker_summaries(data_dir)
    logs = {}
    for p in sorted(data_dir.glob("worker_*.log")):
        m = WORKER_LOG_RE.search(p.name)
        if m:
            logs[int(m.group(1))] = parse_worker_log(p)
    merged = load_merged(run_dir)
    status = run_status(run_dir, summaries, logs)
    report = build_report(run_dir, summaries, logs, merged, status)
    if write:
        # Write analysis.md next to the data (logs/ for v3 layout, run_dir for flat).
        out = data_dir / "analysis.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"[analysis] wrote {out}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True, help="Benchmark run dir to analyze.")
    ap.add_argument("--watch", action="store_true",
                    help="Poll until the run completes/crashes, then analyze.")
    ap.add_argument("--poll-interval", type=float, default=30.0,
                    help="Watch poll interval (seconds). Default 30.")
    ap.add_argument("--timeout", type=float, default=4 * 3600,
                    help="Watch timeout (seconds). Default 4h.")
    ap.add_argument("--no-write", action="store_true",
                    help="Print report to stdout instead of writing analysis.md.")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    if args.watch:
        print(f"[analysis] watching {run_dir} (poll {args.poll_interval}s, "
              f"timeout {_fmt_seconds(args.timeout)})")
        start = time.monotonic()
        last_status = None
        while True:
            # Resolve the data dir each poll (v3 logs/ subdir vs flat run_dir)
            # — the layout may only become apparent once worker JSONs land.
            cand = run_dir / "logs"
            data_dir = cand if cand.is_dir() and any(cand.glob("worker_*.json")) else run_dir
            summaries = load_worker_summaries(data_dir)
            logs = {}
            for p in sorted(data_dir.glob("worker_*.log")):
                m = WORKER_LOG_RE.search(p.name)
                if m:
                    logs[int(m.group(1))] = parse_worker_log(p)
            status = run_status(run_dir, summaries, logs)
            if status != last_status:
                print(f"[analysis] status: {status} "
                      f"({len(summaries)} worker jsons, {len(logs)} logs)")
                last_status = status
            if status in ("complete", "crashed", "crashed (OOM)"):
                break
            if status == "partial" and logs and all(
                logs[w].get("oom") or logs[w].get("saw_summary_saved") for w in logs
            ):
                break
            if time.monotonic() - start > args.timeout:
                print("[analysis] watch timeout — analyzing whatever we have.")
                break
            time.sleep(args.poll_interval)

    write = not args.no_write
    report = analyze(run_dir, write=write)
    if args.no_write:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
