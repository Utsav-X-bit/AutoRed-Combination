"""
JailGuard Reimplementation — Results Analyser
=============================================
Reads saved results JSON/CSV and produces:
  • Overall accuracy / precision / recall / F1
  • Per-attack-type breakdown table
  • Divergence distribution plots
  • Comparison across mutators (if multiple result files present)

Usage:
    # Analyse latest results file
    python analyze_results.py

    # Analyse a specific file
    python analyze_results.py --results results/results_20240120_153000.json

    # Compare multiple result files (different mutators/settings)
    python analyze_results.py --compare results/results_PL.json results/results_TL.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import List

sys.path.insert(0, os.path.dirname(__file__))


# ─────────────────────────────────────────────────────────────────────────────
#  Load results
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def latest_results_file(results_dir: str = "./results") -> str:
    """Find the most recently written results JSON."""
    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.startswith("results_") and f.endswith(".json")
    ]
    if not files:
        raise FileNotFoundError(f"No results JSON found in {results_dir}")
    return max(files, key=os.path.getmtime)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

_BENIGN_LABELS = {"Benign", "benign", "BENIGN"}


def compute_metrics(results: List[dict]) -> dict:
    TP = FP = TN = FN = 0
    per_type = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0,
                                    "total": 0, "max_divs": []})
    for r in results:
        label     = str(r.get("label", "Unknown"))
        is_attack = r.get("is_attack")
        if is_attack is None:
            continue
        is_benign = label in _BENIGN_LABELS
        per_type[label]["total"] += 1
        if r.get("max_div") is not None:
            per_type[label]["max_divs"].append(r["max_div"])

        if not is_benign and is_attack:
            TP += 1; per_type[label]["TP"] += 1
        elif is_benign and is_attack:
            FP += 1; per_type[label]["FP"] += 1
        elif is_benign and not is_attack:
            TN += 1; per_type[label]["TN"] += 1
        else:
            FN += 1; per_type[label]["FN"] += 1

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP)    if (TP + FP) else 0
    recall    = TP / (TP + FN)    if (TP + FN) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return dict(
        total=total, TP=TP, FP=FP, TN=TN, FN=FN,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        per_type=dict(per_type),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_full_report(results: List[dict], title: str = "Results"):
    metrics = compute_metrics(results)

    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")

    # Overall
    print(f"  Total evaluated : {metrics['total']}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  TN={metrics['TN']}  FN={metrics['FN']}")
    print()
    print(f"  {'Metric':<15} {'Value':>10}")
    print(f"  {'─'*28}")
    print(f"  {'Accuracy':<15} {metrics['accuracy']:>9.2%}")
    print(f"  {'Precision':<15} {metrics['precision']:>9.2%}")
    print(f"  {'Recall':<15} {metrics['recall']:>9.2%}")
    print(f"  {'F1-Score':<15} {metrics['f1']:>9.2%}")

    # Per-type
    print(f"\n  {'─'*65}")
    print(f"  {'Attack Type':<22} {'Total':>6} {'Detected':>9} {'DR%':>6} {'Avg maxDiv':>11}")
    print(f"  {'─'*65}")

    # Sort: benign first, then attacks by detection rate descending
    types = sorted(
        metrics['per_type'].items(),
        key=lambda x: (x[0] not in _BENIGN_LABELS,
                       -(x[1]['TP'] / max(x[1]['total'], 1)))
    )
    for atype, m in types:
        t  = m['total']
        if atype in _BENIGN_LABELS:
            detected = m['TN']
            label_tag = "(benign → pass)"
        else:
            detected = m['TP']
            label_tag = ""
        dr = detected / t if t else 0
        avg_div = (sum(m['max_divs']) / len(m['max_divs'])
                   if m['max_divs'] else 0.0)
        print(f"  {atype:<22} {t:>6} {detected:>9}  {dr:>5.0%}  {avg_div:>11.5f}  {label_tag}")

    print(f"{'═'*65}\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Divergence distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_divergence_distribution(results: List[dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        attacks = [r for r in results
                   if r.get("max_div") is not None and r.get("label") not in _BENIGN_LABELS]
        benigns = [r for r in results
                   if r.get("max_div") is not None and r.get("label") in _BENIGN_LABELS]

        attack_divs = [r["max_div"] for r in attacks]
        benign_divs = [r["max_div"] for r in benigns]

        threshold = results[0].get("threshold", 0.02) if results else 0.02

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ── Distribution histogram ──────────────────────────────────────
        ax1.set_title("Max KL Divergence Distribution", fontsize=12, fontweight='bold')
        bins = np.linspace(0, max(attack_divs + benign_divs + [0.1]), 40)
        if attack_divs:
            ax1.hist(attack_divs, bins=bins, alpha=0.6, color='#e74c3c', label='Attack')
        if benign_divs:
            ax1.hist(benign_divs, bins=bins, alpha=0.6, color='#2ecc71', label='Benign')
        ax1.axvline(x=threshold, color='navy', linestyle='--', linewidth=2,
                    label=f'Threshold = {threshold}')
        ax1.set_xlabel("Max KL Divergence")
        ax1.set_ylabel("Count")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ── Per-type bar chart ──────────────────────────────────────────
        metrics = compute_metrics(results)
        types = [(t, m) for t, m in metrics['per_type'].items()
                 if t not in _BENIGN_LABELS and m['total'] > 0]
        types.sort(key=lambda x: x[1]['TP'] / max(x[1]['total'], 1), reverse=True)

        labels_ = [t for t, _ in types]
        drs     = [m['TP'] / max(m['total'], 1) for _, m in types]
        colors_ = ['#e74c3c' if dr >= 0.7 else '#f39c12' if dr >= 0.4 else '#95a5a6'
                   for dr in drs]

        ax2.set_title("Detection Rate by Attack Type", fontsize=12, fontweight='bold')
        bars = ax2.barh(labels_, drs, color=colors_, edgecolor='white', linewidth=0.5)
        ax2.axvline(x=0.7, color='navy', linestyle='--', alpha=0.5, label='70% line')
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Detection Rate")
        for bar, dr in zip(bars, drs):
            ax2.text(min(dr + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                     f'{dr:.0%}', va='center', fontsize=9)
        ax2.grid(alpha=0.3, axis='x')
        ax2.legend()

        plt.suptitle(
            f"JailGuard Results  |  mutator={results[0].get('mutator','?')}  "
            f"N={results[0].get('n_variants','?')}  "
            f"acc={metrics['accuracy']:.1%}",
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved → {output_path}")
    except ImportError as e:
        print(f"  [plot] matplotlib not available: {e}")
    except Exception as e:
        print(f"  [plot] Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Compare multiple runs
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(paths: List[str]):
    print(f"\n{'═'*70}")
    print(f"  COMPARISON: {len(paths)} result files")
    print(f"{'═'*70}")
    print(f"  {'File':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'─'*65}")

    for p in paths:
        results  = load_results(p)
        metrics  = compute_metrics(results)
        basename = os.path.basename(p)
        print(
            f"  {basename:<35} "
            f"{metrics['accuracy']:>5.1%} "
            f"{metrics['precision']:>5.1%} "
            f"{metrics['recall']:>5.1%} "
            f"{metrics['f1']:>5.1%}"
        )
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse JailGuard batch evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results",     default=None, nargs="?",
                        help="Path to results JSON (default: latest in ./results/)")
    parser.add_argument("--results_dir", default="./results",
                        help="Directory to scan for results")
    parser.add_argument("--compare",     nargs="+", default=None,
                        help="Paths to multiple results files for comparison")
    parser.add_argument("--no_plot",     action="store_true",
                        help="Skip generating the distribution plot")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return

    # Single file analysis
    path = args.results
    if path is None:
        try:
            path = latest_results_file(args.results_dir)
            print(f"Auto-selected: {path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    results = load_results(path)
    print_full_report(results, title=f"JailGuard Results: {os.path.basename(path)}")

    if not args.no_plot:
        plot_path = path.replace(".json", "_plot.png")
        plot_divergence_distribution(results, plot_path)


if __name__ == "__main__":
    main()
