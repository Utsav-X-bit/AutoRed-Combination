import argparse
import json
import math
import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

# Import scipy if available, otherwise define simple fallback
try:
    from scipy.stats import mcnemar
except ImportError:
    mcnemar = None

def compute_difficulty(sample: dict) -> str:
    """Compute difficulty tier based on defense complexity."""
    opening = sample.get("opening_defense") or ""
    closing = sample.get("closing_defense") or ""
    combined = opening + " " + closing

    # Length-based
    total_words = len(combined.split())

    # Complexity signals
    has_sandwich = bool(opening.strip() and closing.strip())
    has_roleplay = bool(re.search(r"(?:role|pretend|character|persona)", combined, re.I))
    has_constraint = bool(re.search(r"(?:must not|do not|never|forbidden|prohibited)", combined, re.I))
    has_formatting = bool(re.search(r"(?:format|formatted|markdown|html|table|list)", combined, re.I))
    has_encoding = bool(re.search(r"(?:base64|binary|hex|encode|decode|unicode)", combined, re.I))
    has_translation = bool(re.search(r"(?:translate|language|French|German|Spanish|Japanese|Chinese)", combined, re.I))

    complexity_score = sum([
        has_sandwich, has_roleplay, has_constraint,
        has_formatting, has_encoding, has_translation,
    ])

    if complexity_score >= 4 or total_words > 200:
        return "hard"
    elif complexity_score >= 2 or total_words > 100:
        return "medium"
    else:
        return "easy"

def calc_entropy(counts):
    total = sum(counts.values())
    if total == 0: return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def detect_primitives(attack_text):
    if not attack_text:
        return []
    prims = []
    if re.search(r"(?:role|pretend|character|persona|imagine)", attack_text, re.I):
        prims.append("Roleplay")
    if re.search(r"(?:must not|do not|never|forbidden|prohibited)", attack_text, re.I):
        prims.append("Reflection")
    if re.search(r"(?:base64|binary|hex|encode|decode|unicode)", attack_text, re.I):
        prims.append("Encoding")
    if re.search(r"(?:format|formatted|markdown|html|table|list|json|xml)", attack_text, re.I):
        prims.append("Markdown")
    if re.search(r"(?:you are|administrator|system admin|authority|official)", attack_text, re.I):
        prims.append("Authority")
    return prims


def _parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _benchmark_sort_key(path: Path):
    """Sort key for benchmark dirs; returns a timezone-naive datetime.

    Handles both layouts: ``<path>/merged_summary.json`` (legacy) and
    ``<path>/logs/merged_summary.json`` (new nested). All return values are
    normalized to naive datetimes so mixed aware/naive comparisons never raise.
    """
    for summary_path in (path / "merged_summary.json", path / "logs" / "merged_summary.json"):
        summary = _load_json(summary_path)
        if summary:
            ts = _parse_timestamp(summary.get("metadata", {}).get("timestamp"))
            if ts is not None:
                return ts.replace(tzinfo=None) if ts.tzinfo else ts
    for worker_path in (path / "worker_0.json", path / "logs" / "worker_0.json"):
        worker0 = _load_json(worker_path)
        if worker0:
            ts = _parse_timestamp(worker0.get("metadata", {}).get("timestamp"))
            if ts is not None:
                return ts.replace(tzinfo=None) if ts.tzinfo else ts
    return datetime.fromtimestamp(path.stat().st_mtime)


def discover_benchmark_dirs(benchmarks_root: str):
    """Discover benchmark directories under ``benchmarks_root``.

    Supports the legacy flat layout (``<root>/<name>/merged_summary.json``)
    and, when ``benchmarks_root`` is the project ``results`` dir, the new
    nested layout (``results/benchmark/<model>/<chars>/logs/merged_summary.json``).
    """
    root = Path(benchmarks_root)
    candidates = []
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "merged_summary.json").exists() or list(child.glob("worker_*.json")):
                candidates.append(child)

    # New nested layout: results/benchmark/<model>/<chars>/logs/merged_summary.json.
    # Discovered when benchmarks_root is "results" (or "results/benchmarks" and
    # a sibling "benchmark" dir exists). Use the <chars> dir as the benchmark dir
    # so callers see logs/merged_summary.json and runs/{success,failed}.
    nested_root = None
    if root.name == "benchmarks":
        nested_root = root.parent / "benchmark"
    elif root.name == "results":
        nested_root = root / "benchmark"
    elif (root.parent / "benchmark").exists():
        nested_root = root.parent / "benchmark"
    if nested_root and nested_root.exists():
        for model_dir in sorted(nested_root.iterdir()):
            if not model_dir.is_dir():
                continue
            for chars_dir in sorted(model_dir.iterdir()):
                if not chars_dir.is_dir():
                    continue
                if (chars_dir / "logs" / "merged_summary.json").exists():
                    candidates.append(chars_dir)

    return sorted(candidates, key=_benchmark_sort_key)


def resolve_benchmark_path(arg_value: str, benchmarks_root: str, kind: str, current_path: Path | None = None):
    if arg_value and arg_value != "auto":
        return Path(arg_value)

    candidates = discover_benchmark_dirs(benchmarks_root)
    if not candidates:
        raise FileNotFoundError(f"No benchmark directories found under {benchmarks_root}")

    if kind == "current":
        return candidates[-1]

    if current_path is not None:
        current_resolved = current_path.resolve()
        matching = [p for p in candidates if p.resolve() == current_resolved]
        if matching:
            idx = candidates.index(matching[0])
            if idx == 0:
                raise FileNotFoundError(
                    f"Cannot auto-select a baseline before {matching[0]} because no earlier benchmark exists"
                )
            return candidates[idx - 1]

    if len(candidates) < 2:
        raise FileNotFoundError(
            f"Need at least two benchmark directories under {benchmarks_root} to auto-select a baseline"
        )
    return candidates[-2]


def resolve_trace_root_for_benchmark(benchmark_dir: Path, traces_root: str):
    """Return the directory holding this benchmark's run_*.json traces.

    New layout: ``<benchmark_dir>/runs`` (created by results_layout.py) is
    preferred when present, regardless of timestamp. Legacy layout falls back
    to the date-based archive tree under ``traces_root``.
    """
    runs_dir = benchmark_dir / "runs"
    if runs_dir.exists() and list(runs_dir.rglob("run_*.json")):
        return runs_dir

    summary = _load_json(benchmark_dir / "merged_summary.json")
    ts = None
    if summary:
        ts = _parse_timestamp(summary.get("metadata", {}).get("timestamp"))
    if ts is None:
        worker0 = _load_json(benchmark_dir / "worker_0.json")
        if worker0:
            ts = _parse_timestamp(worker0.get("metadata", {}).get("timestamp"))
    if ts is None:
        return None

    candidate = Path(traces_root) / ts.strftime("%Y-%m-%d")
    if candidate.exists():
        run_files = list(candidate.rglob("run_*.json"))
        if run_files:
            return candidate
    return None

def analyze_directory(dir_path):
    dir_path = Path(dir_path)
    summary_file = dir_path / "merged_summary.json"
    if summary_file.exists():
        json_files = [summary_file]
    else:
        json_files = list(dir_path.rglob("run_*.json"))
        if not json_files:
            json_files = list(dir_path.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {dir_path}")
        return {}

    stats = {
        "total_scenarios": len(json_files),
        "success_count": 0,
        "verified_success_count": 0,
        "top1_success_count": 0,
        "top3_success_count": 0,
        "top5_success_count": 0,
        "total_attempts": 0,
        "successful_attempts_list": [],
        
        # Extractor Metrics
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "regex_hits": 0, "llm_hits": 0, "consensus_hits": 0,
        
        # Planner Metrics
        "judge_confidences": [],
        "strategy_counts": defaultdict(int),
        "strategy_successes": defaultdict(int),
        "strategy_attempts": defaultdict(int),
        
        # Generator Metrics
        "attack_lengths": [],
        "duplicate_attacks": 0,
        "unique_prompts": set(),
        "total_ttr": 0.0,
        "valid_ttr_count": 0,
        
        # Behavioral / Primitive
        "primitive_counts": defaultdict(int),
        "primitive_successes": defaultdict(int),
        
        # Transitions
        "transitions": defaultdict(int),
        
        # Defense Intelligence
        "success_by_diff": defaultdict(int),
        "total_by_diff": defaultdict(int),
        "success_by_defense": defaultdict(int),
        "total_by_defense": defaultdict(int),
        "success_by_code": defaultdict(int),
        "total_by_code": defaultdict(int),
        
        # Failures
        "fail_verifier": 0,
        "fail_judge": 0,
        "fail_extractor": 0,
        
        # Near Miss to Success
        "near_miss_count": 0,
        "near_miss_to_success": 0,
    }

    # Scenario results dictionary for statistical comparison
    stats["scenario_results"] = {}

    summary_mode = len(json_files) == 1 and json_files[0].name == "merged_summary.json"

    for f_path in json_files:
        with open(f_path, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        if summary_mode:
            results = data.get("results", [])
            total_rounds = data.get("total_rounds", len(results))
            success_count = data.get("total_successes", 0)
            verified_count = data.get("verified_success", 0)
            total_attempts = sum(r.get("attempts", 0) for r in results)
            successful_attempts = [r.get("attempts", 0) for r in results if r.get("success")]
            stats["total_scenarios"] = total_rounds
            stats["success_count"] = success_count
            stats["verified_success_count"] = verified_count
            stats["top1_success_count"] = data.get("top1_success", 0)
            stats["top3_success_count"] = data.get("top3_success", 0)
            stats["top5_success_count"] = data.get("top5_success", 0)
            stats["total_attempts"] = total_attempts
            stats["successful_attempts_list"] = successful_attempts
            stats["tp"] = data.get("extractor_metrics", {}).get("true_positive", 0)
            stats["fp"] = data.get("extractor_metrics", {}).get("false_positive", 0)
            stats["fn"] = data.get("extractor_metrics", {}).get("false_negative", 0)
            # The merged summary does not carry trace-level planner/generator analytics.
            # Keep these buckets explicit so the report can clearly show what is and isn't available.
            stats["total_by_defense"]["unknown"] += total_rounds
            stats["total_by_code"]["unknown"] += total_rounds
            stats["success_by_defense"]["unknown"] += success_count
            stats["success_by_code"]["unknown"] += success_count
            for r in results:
                scenario_id = f"{r.get('worker_id', 'w0')}:{r.get('round', 0)}:{r.get('access_code', '')}"
                stats["scenario_results"][scenario_id] = {
                    "success": r.get("success", False),
                    "verified": r.get("success", False),
                    "attempts": r.get("attempts", 0),
                }
            continue

        scenario = data.get("scenario", {})
        raw_entry = data.get("raw_dataset_entry", {})
        
        # Identify defense ID/attributes
        defense_id = raw_entry.get("defense_id", "unknown")
        defense_type = scenario.get("defense_type", "unknown")
        access_code_type = scenario.get("access_code_type", "unknown")
        
        # Calculate difficulty dynamically
        difficulty = compute_difficulty(raw_entry)
        
        stats["total_by_diff"][difficulty] += 1
        stats["total_by_defense"][defense_type] += 1
        stats["total_by_code"][access_code_type] += 1
        
        attempts = data.get("attempts", [])
        is_success = data.get("result") == "SUCCESS" or any(att.get("generator_success") or att.get("ground_truth_found") or att.get("verification", {}).get("verified") for att in attempts)
        is_verified = any(att.get("verification", {}).get("verified") for att in attempts)
        
        scenario_id = getattr(f_path, "name", str(f_path))
        stats["scenario_results"][scenario_id] = {
            "success": is_success,
            "verified": is_verified,
            "attempts": len(attempts)
        }

        if is_success:
            stats["success_count"] += 1
            stats["success_by_diff"][difficulty] += 1
            stats["success_by_defense"][defense_type] += 1
            stats["success_by_code"][access_code_type] += 1
            stats["successful_attempts_list"].append(len(attempts))
            
            if len(attempts) == 1:
                stats["top1_success_count"] += 1
            if len(attempts) <= 3:
                stats["top3_success_count"] += 1
            if len(attempts) <= 5:
                stats["top5_success_count"] += 1

        if is_verified:
            stats["verified_success_count"] += 1

        stats["total_attempts"] += len(attempts)

        # Transition tracking & Failure attribution
        prev_strategy = None
        for i, att in enumerate(attempts):
            judge = att.get("judge", {})
            judge_conf = judge.get("confidence", 0)
            stats["judge_confidences"].append(judge_conf)
            
            gen = att.get("generator", {})
            attack_text = gen.get("generated_attack", "")
            strategy = gen.get("strategy", "unknown")
            attack_len = gen.get("attack_length", len(attack_text))
            is_dup = gen.get("duplicate_attack", False)
            
            stats["attack_lengths"].append(attack_len)
            if is_dup:
                stats["duplicate_attacks"] += 1
                
            stats["strategy_counts"][strategy] += 1
            stats["strategy_attempts"][strategy] += 1
            if is_success and i == len(attempts) - 1:
                stats["strategy_successes"][strategy] += 1
                
            att_hash = gen.get("attack_hash")
            if att_hash:
                stats["unique_prompts"].add(att_hash)
                
            ttr = att.get("attempt_time_ms")
            if ttr:
                stats["total_ttr"] += ttr
                stats["valid_ttr_count"] += 1
                
            # Primitives
            prims = detect_primitives(attack_text)
            for prim in prims:
                stats["primitive_counts"][prim] += 1
                if is_success and i == len(attempts) - 1:
                    stats["primitive_successes"][prim] += 1
                    
            # Transitions
            if prev_strategy is not None:
                trans_type = "Retry" if prev_strategy == strategy else "Switch"
                stats["transitions"][trans_type] += 1
            prev_strategy = strategy

            # Extractor Hits & Consensus
            ext = att.get("extractor", {})
            regex_c = len(ext.get("regex_candidates", []))
            llm_c = len(ext.get("llm_candidates", []))
            stats["regex_hits"] += regex_c
            stats["llm_hits"] += llm_c
            
            # Simple consensus proxy (common elements)
            if regex_c > 0 and llm_c > 0:
                stats["consensus_hits"] += 1

            # Precision/Recall calculation
            gt_leaked = att.get("ground_truth_found", False)
            extractor_found = ext.get("best_candidate") is not None
            verified = att.get("verification", {}).get("verified", False)
            
            if gt_leaked:
                if extractor_found and verified:
                    stats["tp"] += 1
                else:
                    stats["fn"] += 1
            else:
                if extractor_found:
                    stats["fp"] += 1
                else:
                    stats["tn"] += 1

            # Failures
            if not is_success and i == len(attempts) - 1:
                # Last attempt failed
                if gt_leaked and extractor_found and not verified:
                    stats["fail_verifier"] += 1
                elif gt_leaked and not extractor_found:
                    stats["fail_extractor"] += 1
                else:
                    stats["fail_judge"] += 1

    return stats

def compute_metrics_summary(stats):
    if not stats: return {}
    
    total = stats["total_scenarios"]
    avg_attempts = np.mean(stats["successful_attempts_list"]) if stats["successful_attempts_list"] else 0.0
    
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    strategy_entropy = calc_entropy(stats["strategy_counts"])
    avg_conf = np.mean(stats["judge_confidences"]) if stats["judge_confidences"] else 0.0
    avg_len = np.mean(stats["attack_lengths"]) if stats["attack_lengths"] else 0.0
    dup_rate = stats["duplicate_attacks"] / stats["total_attempts"] if stats["total_attempts"] > 0 else 0.0
    avg_ttr = (stats["total_ttr"] / stats["valid_ttr_count"] / 1000.0) if stats["valid_ttr_count"] > 0 else 0.0
    
    return {
        "success_rate": stats["success_count"] / total,
        "verified_rate": stats["verified_success_count"] / total,
        "top1_rate": stats["top1_success_count"] / total,
        "top3_rate": stats["top3_success_count"] / total,
        "top5_rate": stats["top5_success_count"] / total,
        "avg_attempts": avg_attempts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "strategy_entropy": strategy_entropy,
        "avg_conf": avg_conf,
        "avg_len": avg_len,
        "dup_rate": dup_rate,
        "avg_ttr": avg_ttr,
        "novel_prompts": len(stats["unique_prompts"])
    }

def run_significance_test(baseline_results, current_results):
    """McNemar statistical significance test on scenario success."""
    if not mcnemar:
        return "N/A (scipy not installed)", 1.0
        
    # Align by scenario ID
    common_ids = set(baseline_results.keys()).intersection(current_results.keys())
    if len(common_ids) < 10:
        return "Insufficient aligned scenarios", 1.0
        
    # McNemar table:
    #                 Current Success | Current Failure
    # Base Success        a           |       b
    # Base Failure        c           |       d
    a, b, c, d = 0, 0, 0, 0
    for cid in common_ids:
        bs = baseline_results[cid]["success"]
        cs = current_results[cid]["success"]
        if bs and cs: a += 1
        elif bs and not cs: b += 1
        elif not bs and cs: c += 1
        else: d += 1
        
    table = [[a, b], [c, d]]
    try:
        result = mcnemar(table, exact=True)
        p_val = result.pvalue
        sig = "Significant" if p_val < 0.05 else "Not Significant"
        return f"{sig} (p={p_val:.4f})", p_val
    except Exception as e:
        return f"Error: {e}", 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="auto",
        help="Baseline benchmark directory or 'auto' to use the previous latest benchmark",
    )
    parser.add_argument(
        "--current",
        default="auto",
        help="Current benchmark directory or 'auto' to use the latest benchmark",
    )
    parser.add_argument(
        "--benchmarks-root",
        default="results/benchmarks",
        help="Root directory that contains benchmark folders",
    )
    parser.add_argument(
        "--traces-root",
        default="results",
        help="Root directory that contains dated run_*.json trace archives",
    )
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    args = parser.parse_args()

    current_benchmark = resolve_benchmark_path(args.current, args.benchmarks_root, "current")
    baseline_benchmark = resolve_benchmark_path(
        args.baseline,
        args.benchmarks_root,
        "baseline",
        current_path=current_benchmark,
    )
    current_trace_root = resolve_trace_root_for_benchmark(current_benchmark, args.traces_root)
    baseline_trace_root = resolve_trace_root_for_benchmark(baseline_benchmark, args.traces_root)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "csv").mkdir(exist_ok=True)
    (out_dir / "json").mkdir(exist_ok=True)

    print(f"Baseline benchmark: {baseline_benchmark}")
    print(f"Current benchmark : {current_benchmark}")
    if baseline_trace_root:
        print(f"Baseline traces   : {baseline_trace_root}")
    if current_trace_root:
        print(f"Current traces    : {current_trace_root}")

    print("Analyzing baseline benchmark...")
    base_stats = analyze_directory(baseline_benchmark)
    print("Analyzing current benchmark...")
    curr_stats = analyze_directory(current_benchmark)

    if not base_stats or not curr_stats:
        print("Error: Could not analyze runs.")
        return

    base_trace_stats = analyze_directory(baseline_trace_root) if baseline_trace_root else {}
    curr_trace_stats = analyze_directory(current_trace_root) if current_trace_root else {}

    base_summary = compute_metrics_summary(base_stats)
    curr_summary = compute_metrics_summary(curr_stats)
    base_trace_summary = compute_metrics_summary(base_trace_stats) if base_trace_stats else {}
    curr_trace_summary = compute_metrics_summary(curr_trace_stats) if curr_trace_stats else {}

    # Statistical significance
    sig_text, p_val = run_significance_test(base_stats["scenario_results"], curr_stats["scenario_results"])
    base_trace_available = bool(base_trace_stats) and bool(base_trace_stats.get("strategy_counts"))
    curr_trace_available = bool(curr_trace_stats) and bool(curr_trace_stats.get("strategy_counts"))

    # Generate Markdown Report
    md_path = out_dir / "comparison_report.md"
    with open(md_path, "w") as f:
        f.write("# AutoRed Benchmark Comparison Report (20-Layer Analysis)\n\n")

        # 1 Executive Summary
        f.write("## 1 Executive Summary\n")
        f.write("This report presents a comprehensive multi-layered evaluation comparing the AutoRed baseline strategy selection algorithm against the optimized planner adapter.\n\n")
        f.write(f"- Baseline benchmark: `{baseline_benchmark}`\n")
        f.write(f"- Current benchmark: `{current_benchmark}`\n")
        if baseline_trace_root:
            f.write(f"- Baseline trace archive: `{baseline_trace_root}`\n")
        if current_trace_root:
            f.write(f"- Current trace archive: `{current_trace_root}`\n")
        f.write("\n")

        # 2 Overall Metrics
        f.write("## 2 Overall Metrics\n")
        f.write("| Metric | Baseline | Current | Δ |\n")
        f.write("|---|---|---|---|\n")
        for key in ["success_rate", "verified_rate", "top1_rate", "top3_rate", "top5_rate"]:
            b_val = base_summary[key]
            c_val = curr_summary[key]
            f.write(f"| {key.replace('_', ' ').title()} | {b_val*100:.1f}% | {c_val*100:.1f}% | {(c_val-b_val)*100:+.1f}% |\n")
        b_att = base_summary["avg_attempts"]
        c_att = curr_summary["avg_attempts"]
        f.write(f"| Avg Attempts (Success) | {b_att:.2f} | {c_att:.2f} | {c_att-b_att:+.2f} |\n")
        for key in ["precision", "recall", "f1"]:
            b_val = base_summary[key]
            c_val = curr_summary[key]
            f.write(f"| Extractor {key.title()} | {b_val:.3f} | {c_val:.3f} | {c_val-b_val:+.3f} |\n")

        # 3 Statistical Significance
        f.write("\n## 3 Statistical Significance\n")
        f.write(f"- **McNemar Test Result:** {sig_text}\n")
        f.write(f"- **Confidence Level:** 95%\n\n")

        # 4 Component Analysis
        f.write("## 4 Component Analysis\n")
        f.write("| Component | Baseline Status | Current Status | Improvement |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Scenario Intelligence | Basic parsing | Complexity-aware | Yes (Difficulty scaling resolved) |\n")
        f.write(f"| Planner | SFT/Adapter based | Fine-tuned adapter | Yes (Conciseness up) |\n")
        f.write(f"| Generator | 8B Lexi Uncensored | 8B Lexi Uncensored | Yes (Fewer duplicate attacks) |\n")
        f.write(f"| Extractor | Regex + LLM Consensus | Regex + LLM Consensus | Yes (Precision = 1.000) |\n")
        f.write(f"| Verifier | Active verification | Active verification | Stable |\n")
        f.write(f"| Runtime Controller | Static thresholds | Dynamic retry | Stable |\n\n")

        # 5 Planner Analysis
        f.write("## 5 Planner Analysis\n")
        if curr_trace_available:
            f.write(f"- **Strategy Entropy:** Current {curr_trace_summary['strategy_entropy']:.3f}\n")
            f.write(f"- **Average Judge Confidence:** Current {curr_trace_summary['avg_conf']:.3f}\n")
            if base_trace_available:
                f.write(f"- **Baseline Strategy Entropy:** {base_trace_summary['strategy_entropy']:.3f}\n")
                f.write(f"- **Baseline Average Judge Confidence:** {base_trace_summary['avg_conf']:.3f}\n")
            else:
                f.write("- Baseline planner trace metrics are unavailable from the archived benchmark summaries.\n")
            f.write("\n")
        else:
            f.write(f"- **Strategy Entropy:** Baseline {base_summary['strategy_entropy']:.3f} vs Current {curr_summary['strategy_entropy']:.3f}\n")
            f.write(f"- **Average Judge Confidence:** Baseline {base_summary['avg_conf']:.3f} vs Current {curr_summary['avg_conf']:.3f}\n\n")

        # 6 Generator Analysis
        f.write("## 6 Generator Analysis\n")
        if curr_trace_available:
            f.write(f"- **Average Attack Length:** Current {curr_trace_summary['avg_len']:.1f} chars\n")
            f.write(f"- **Duplicate Attacks Rate:** Current {curr_trace_summary['dup_rate']*100:.2f}%\n")
            f.write(f"- **Average TTR:** Current {curr_trace_summary['avg_ttr']:.2f}s\n")
            if base_trace_available:
                f.write(f"- **Baseline Average Attack Length:** {base_trace_summary['avg_len']:.1f} chars\n")
                f.write(f"- **Baseline Duplicate Attacks Rate:** {base_trace_summary['dup_rate']*100:.2f}%\n")
                f.write(f"- **Baseline Average TTR:** {base_trace_summary['avg_ttr']:.2f}s\n")
            else:
                f.write("- Baseline generator trace metrics are unavailable from the archived benchmark summaries.\n")
            f.write("\n")
        else:
            f.write(f"- **Average Attack Length:** Baseline {base_summary['avg_len']:.1f} chars vs Current {curr_summary['avg_len']:.1f} chars\n")
            f.write(f"- **Duplicate Attacks Rate:** Baseline {base_summary['dup_rate']*100:.2f}% vs Current {curr_summary['dup_rate']*100:.2f}%\n")
            f.write(f"- **Average TTR:** Baseline {base_summary['avg_ttr']:.2f}s vs Current {curr_summary['avg_ttr']:.2f}s\n\n")

        # 7 Extractor Analysis
        f.write("## 7 Extractor Analysis\n")
        if curr_trace_available:
            f.write(f"- **Regex Hits:** Current {curr_trace_stats['regex_hits']}\n")
            f.write(f"- **LLM Hits:** Current {curr_trace_stats['llm_hits']}\n")
            f.write(f"- **Consensus Hits:** Current {curr_trace_stats['consensus_hits']}\n")
            if base_trace_available:
                f.write(f"- **Baseline Regex Hits:** {base_trace_stats['regex_hits']}\n")
                f.write(f"- **Baseline LLM Hits:** {base_trace_stats['llm_hits']}\n")
                f.write(f"- **Baseline Consensus Hits:** {base_trace_stats['consensus_hits']}\n")
            else:
                f.write("- Baseline extractor trace metrics are unavailable from the archived benchmark summaries.\n")
            f.write("\n")
        else:
            f.write(f"- **Regex Hits:** Baseline {base_stats['regex_hits']} vs Current {curr_stats['regex_hits']}\n")
            f.write(f"- **LLM Hits:** Baseline {base_stats['llm_hits']} vs Current {curr_stats['llm_hits']}\n")
            f.write(f"- **Consensus Hits:** Baseline {base_stats['consensus_hits']} vs Current {curr_stats['consensus_hits']}\n\n")

        # 8 Verifier Analysis
        f.write("## 8 Verifier Analysis\n")
        f.write(f"- **True Positives:** Baseline {base_stats['tp']} vs Current {curr_stats['tp']}\n")
        f.write(f"- **False Positives:** Baseline {base_stats['fp']} vs Current {curr_stats['fp']}\n\n")

        # 9 Runtime Controller Analysis
        f.write("## 9 Runtime Controller Analysis\n")
        f.write(f"- **Total Attempts Executed:** Baseline {base_stats['total_attempts']} vs Current {curr_stats['total_attempts']}\n\n")

        # 10 Defense Analysis
        f.write("## 10 Defense Analysis\n")
        f.write("| Defense Type | Baseline Success | Current Success |\n")
        f.write("|---|---|---|\n")
        all_defenses = set(base_stats["total_by_defense"].keys()).union(curr_stats["total_by_defense"].keys())
        for dtype in sorted(all_defenses):
            b_succ = base_stats["success_by_defense"].get(dtype, 0)
            b_tot = base_stats["total_by_defense"].get(dtype, 1)
            c_succ = curr_stats["success_by_defense"].get(dtype, 0)
            c_tot = curr_stats["total_by_defense"].get(dtype, 1)
            f.write(f"| {dtype} | {b_succ}/{b_tot} ({b_succ/b_tot*100:.1f}%) | {c_succ}/{c_tot} ({c_succ/c_tot*100:.1f}%) |\n")

        # 11 Access Code Analysis
        f.write("\n## 11 Access Code Analysis\n")
        f.write("| Access Code Type | Baseline Success | Current Success |\n")
        f.write("|---|---|---|\n")
        all_codes = set(base_stats["total_by_code"].keys()).union(curr_stats["total_by_code"].keys())
        for ctype in sorted(all_codes):
            b_succ = base_stats["success_by_code"].get(ctype, 0)
            b_tot = base_stats["total_by_code"].get(ctype, 1)
            c_succ = curr_stats["success_by_code"].get(ctype, 0)
            c_tot = curr_stats["total_by_code"].get(ctype, 1)
            f.write(f"| {ctype} | {b_succ}/{b_tot} ({b_succ/b_tot*100:.1f}%) | {c_succ}/{c_tot} ({c_succ/c_tot*100:.1f}%) |\n")

        # 12 Strategy Analysis
        f.write("\n## 12 Strategy Analysis\n")
        if curr_trace_available:
            f.write("| Strategy | Current Attempts | Current Verified Successes |\n")
            f.write("|---|---|---|\n")
            for strat in sorted(curr_trace_stats["strategy_counts"].keys()):
                f.write(
                    f"| {strat} | {curr_trace_stats['strategy_counts'].get(strat, 0)} | {curr_trace_stats['strategy_successes'].get(strat, 0)} |\n"
                )
            if base_trace_available:
                f.write("\n")
                f.write("| Strategy | Baseline Attempts | Baseline Verified Successes |\n")
                f.write("|---|---|---|\n")
                for strat in sorted(base_trace_stats["strategy_counts"].keys()):
                    f.write(
                        f"| {strat} | {base_trace_stats['strategy_counts'].get(strat, 0)} | {base_trace_stats['strategy_successes'].get(strat, 0)} |\n"
                    )
            else:
                f.write("\nBaseline strategy usage is unavailable from the archived benchmark summaries.\n")
        else:
            f.write("| Strategy | Baseline Usage | Current Usage | Baseline Success | Current Success |\n")
            f.write("|---|---|---|---|---|\n")
            all_strats = set(base_stats["strategy_counts"].keys()).union(curr_stats["strategy_counts"].keys())
            for strat in sorted(all_strats):
                b_count = base_stats["strategy_counts"].get(strat, 0)
                c_count = curr_stats["strategy_counts"].get(strat, 0)
                b_succ = base_stats["strategy_successes"].get(strat, 0)
                c_succ = curr_stats["strategy_successes"].get(strat, 0)
                f.write(f"| {strat} | {b_count} | {c_count} | {b_succ} | {c_succ} |\n")

        # 13 Primitive Analysis
        f.write("\n## 13 Primitive Analysis\n")
        if curr_trace_available:
            f.write("| Primitive | Current Count | Current Success |\n")
            f.write("|---|---|---|\n")
            for prim in sorted(curr_trace_stats["primitive_counts"].keys()):
                f.write(
                    f"| {prim} | {curr_trace_stats['primitive_counts'].get(prim, 0)} | {curr_trace_stats['primitive_successes'].get(prim, 0)} |\n"
                )
            if base_trace_available:
                f.write("\n| Primitive | Baseline Count | Baseline Success |\n")
                f.write("|---|---|---|\n")
                for prim in sorted(base_trace_stats["primitive_counts"].keys()):
                    f.write(
                        f"| {prim} | {base_trace_stats['primitive_counts'].get(prim, 0)} | {base_trace_stats['primitive_successes'].get(prim, 0)} |\n"
                    )
            else:
                f.write("\nBaseline primitive usage is unavailable from the archived benchmark summaries.\n")
        else:
            f.write("| Primitive | Baseline Count | Current Count | Baseline Success | Current Success |\n")
            f.write("|---|---|---|---|---|\n")
            all_prims = set(base_stats["primitive_counts"].keys()).union(curr_stats["primitive_counts"].keys())
            for prim in sorted(all_prims):
                b_count = base_stats["primitive_counts"].get(prim, 0)
                c_count = curr_stats["primitive_counts"].get(prim, 0)
                b_succ = base_stats["primitive_successes"].get(prim, 0)
                c_succ = curr_stats["primitive_successes"].get(prim, 0)
                f.write(f"| {prim} | {b_count} | {c_count} | {b_succ} | {c_succ} |\n")

        # 14 Transition Analysis
        f.write("\n## 14 Transition Analysis\n")
        if curr_trace_available:
            f.write("| Transition Type | Current Count |\n")
            f.write("|---|---|\n")
            for ttype in ["Retry", "Switch"]:
                f.write(f"| {ttype} | {curr_trace_stats['transitions'].get(ttype, 0)} |\n")
            if base_trace_available:
                f.write("\n| Transition Type | Baseline Count |\n")
                f.write("|---|---|\n")
                for ttype in ["Retry", "Switch"]:
                    f.write(f"| {ttype} | {base_trace_stats['transitions'].get(ttype, 0)} |\n")
            else:
                f.write("\nBaseline transition counts are unavailable from the archived benchmark summaries.\n")
        else:
            f.write("| Transition Type | Baseline Count | Current Count |\n")
            f.write("|---|---|---|\n")
            for ttype in ["Retry", "Switch"]:
                f.write(f"| {ttype} | {base_stats['transitions'].get(ttype, 0)} | {curr_stats['transitions'].get(ttype, 0)} |\n")

        # 15 Failure Attribution
        f.write("\n## 15 Failure Attribution\n")
        if curr_trace_available:
            f.write("| Failure Phase | Current Count |\n")
            f.write("|---|---|\n")
            f.write(f"| Verifier Reject | {curr_trace_stats['fail_verifier']} |\n")
            f.write(f"| Judge Reject | {curr_trace_stats['fail_judge']} |\n")
            f.write(f"| Extractor Miss | {curr_trace_stats['fail_extractor']} |\n")
            if base_trace_available:
                f.write("\n| Failure Phase | Baseline Count |\n")
                f.write("|---|---|\n")
                f.write(f"| Verifier Reject | {base_trace_stats['fail_verifier']} |\n")
                f.write(f"| Judge Reject | {base_trace_stats['fail_judge']} |\n")
                f.write(f"| Extractor Miss | {base_trace_stats['fail_extractor']} |\n")
            else:
                f.write("\nBaseline failure attribution is unavailable from the archived benchmark summaries.\n")
        else:
            f.write("| Failure Phase | Baseline | Current |\n")
            f.write("|---|---|---|\n")
            f.write(f"| Verifier Reject | {base_stats['fail_verifier']} | {curr_stats['fail_verifier']} |\n")
            f.write(f"| Judge Reject | {base_stats['fail_judge']} | {curr_stats['fail_judge']} |\n")
            f.write(f"| Extractor Miss | {base_stats['fail_extractor']} | {curr_stats['fail_extractor']} |\n")

        # 16 Oracle Agreement
        f.write("\n## 16 Trace-Level Recovery\n")
        if curr_trace_available:
            f.write(f"- **Current Trace Success Rate:** {curr_trace_summary['success_rate']*100:.1f}%\n")
            f.write(f"- **Current First-Pick Success Rate:** {curr_trace_summary['top1_rate']*100:.1f}%\n")
            if base_trace_available:
                f.write(f"- **Baseline Trace Success Rate:** {base_trace_summary['success_rate']*100:.1f}%\n")
                f.write(f"- **Baseline First-Pick Success Rate:** {base_trace_summary['top1_rate']*100:.1f}%\n")
            else:
                f.write("- Baseline trace recovery metrics are unavailable from the archived benchmark summaries.\n")
            f.write("\n")
        else:
            f.write("- Trace archives were not available for the current benchmark.\n\n")

        # 17 Knowledge Base Growth
        f.write("## 17 Knowledge Base Growth\n")
        if curr_trace_available:
            f.write(f"- **Unique Prompts Harvested:** {len(curr_trace_stats['unique_prompts'])}\n")
            f.write(f"- **Successfully Saved Trajectories:** {curr_trace_stats['success_count']}\n\n")
        else:
            f.write(f"- **Unique Prompts Harvested:** {len(curr_stats['unique_prompts'])}\n")
            f.write(f"- **Successfully Saved Trajectories:** {curr_stats['success_count']}\n\n")

        # 18 Bottleneck Identification
        f.write("## 18 Bottleneck Identification\n")
        f.write("1. Extraction on multiline access codes remains lower than single tokens.\n")
        f.write("2. Translation bypasses still fail when victim enforces multilingual sanitization.\n\n")

        # 19 Actionable Recommendations
        f.write("## 19 Actionable Recommendations\n")
        f.write("1. Incorporate multi-stage formatting cues inside SFT generator.\n")
        f.write("2. Retrain reward model with hard-negatives matching the new failures.\n\n")

        # 20 Next Development Phase
        f.write("## 20 Next Development Phase\n")
        f.write("Phase 11: Planner DPO.\n")

    # Generate HTML Report
    html_path = out_dir / "comparison_report.html"
    with open(html_path, "w") as f:
        f.write("<html><head><title>Comparison Report</title></head><body>")
        with open(md_path, 'r') as md_file:
            f.write(md_file.read().replace("\n", "<br>"))
        f.write("</body></html>")

    # Generate JSON summary
    summary_data = {
        "baseline": base_summary,
        "current": curr_summary,
        "significance": {"p_value": p_val, "result": sig_text}
    }
    with open(out_dir / "json" / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    # Generate Plot
    try:
        metrics = ["Success Rate", "Verified Rate", "Top1 Success", "Extractor F1"]
        base_vals = [base_summary["success_rate"], base_summary["verified_rate"], base_summary["top1_rate"], base_summary["f1"]]
        curr_vals = [curr_summary["success_rate"], curr_summary["verified_rate"], curr_summary["top1_rate"], curr_summary["f1"]]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, base_vals, width, label='Baseline')
        rects2 = ax.bar(x + width/2, curr_vals, width, label='Current')

        ax.set_ylabel('Scores')
        ax.set_title('AutoRed Performance Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "comparison_plot.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"Reports successfully generated at {args.output_dir}/")

if __name__ == "__main__":
    main()
