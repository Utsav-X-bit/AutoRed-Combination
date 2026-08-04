"""
JailGuard Reimplementation — Batch Evaluation
=============================================
Run JailGuard on a stratified sample of the dataset (covering all attack types),
compute detection metrics, and save results to CSV + JSON.

Usage:
    # Evaluate with default settings (~112 items, all attack types)
    python run_batch.py

    # Override mutator and number of variants
    python run_batch.py --mutator TI --n 4

    # Custom number of samples per attack type
    python run_batch.py --samples_per_type 5

    # Use tfidf similarity (faster, no model needed)
    python run_batch.py --sim tfidf

    # Resume from checkpoint (skips already-evaluated items)
    python run_batch.py --resume
"""

import argparse
import sys
import os
import json
import csv
import time
import random
import pickle
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from llm_interface import build_llm
from detector      import JailGuardDetector, load_dataset, get_label, get_params


# ─────────────────────────────────────────────────────────────────────────────
#  Stratified sampling
# ─────────────────────────────────────────────────────────────────────────────

def build_evaluation_set(
    dataset: list, dataset_key: dict,
    attack_types: List[str],
    samples_per_type: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a balanced evaluation set with `samples_per_type` items per attack type.

    Returns:
        List of dicts with keys: serial_num, label, input_data
    """
    random.seed(seed)

    # Group indices by attack type
    type_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        lbl = get_label(dataset_key, idx)
        # normalise label (strip trailing whitespace, None → Unknown)
        lbl = lbl.strip() if lbl else "Unknown"
        type_to_indices[lbl].append(idx)

    print("\n─── Dataset Distribution ─────────────────────────────────")
    for t, idxs in sorted(type_to_indices.items()):
        marker = "✓" if t in attack_types else "✗"
        print(f"  [{marker}] {t:20s}: {len(idxs):5d} samples")
    print()

    eval_set = []
    for atype in attack_types:
        idxs = type_to_indices.get(atype, [])
        if not idxs:
            print(f"  ⚠️  Attack type '{atype}' not found in dataset, skipping.")
            continue
        chosen = random.sample(idxs, min(samples_per_type, len(idxs)))
        for idx in chosen:
            eval_set.append({
                "serial_num": idx,
                "label":      atype,
                "input_data": dataset[idx],
            })

    random.shuffle(eval_set)
    return eval_set


# ─────────────────────────────────────────────────────────────────────────────
#  Results saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: List[dict], output_dir: str):
    """Save results to both JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = os.path.join(output_dir, f"results_{ts}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # CSV
    csv_path = os.path.join(output_dir, f"results_{ts}.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n  Results saved:")
    print(f"    JSON → {json_path}")
    print(f"    CSV  → {csv_path}")
    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

_BENIGN_LABELS = {"Benign", "benign", "BENIGN"}

def compute_metrics(results: List[dict]) -> dict:
    """
    Compute detection metrics.

    Positive class = ATTACK.
    """
    TP = FP = TN = FN = 0
    per_type = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

    for r in results:
        label      = r["label"]
        is_attack  = r["is_attack"]
        is_benign  = label in _BENIGN_LABELS

        if not is_benign and is_attack:     # correctly detected attack
            TP += 1
            per_type[label]["TP"] += 1
        elif is_benign and is_attack:       # falsely flagged benign as attack
            FP += 1
            per_type[label]["FP"] += 1
        elif is_benign and not is_attack:   # correctly identified benign
            TN += 1
            per_type[label]["TN"] += 1
        else:                               # missed attack
            FN += 1
            per_type[label]["FN"] += 1

    total     = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP)    if (TP + FP) else 0
    recall    = TP / (TP + FN)    if (TP + FN) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)

    return {
        "total":     total,
        "TP":        TP,  "FP": FP,  "TN": TN,  "FN": FN,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "per_type":  dict(per_type),
    }


def print_metrics(metrics: dict, mutator: str, n: int, threshold: float):
    lines = [
        f"\n{'═'*60}",
        f"  EVALUATION SUMMARY",
        f"  Mutator: {mutator}  |  N={n}  |  Threshold={threshold}",
        f"{'═'*60}",
        f"  Total evaluated : {metrics['total']}",
        f"  True Positives  : {metrics['TP']}  (attacks correctly detected)",
        f"  False Positives : {metrics['FP']}  (benign wrongly flagged)",
        f"  True Negatives  : {metrics['TN']}  (benign correctly passed)",
        f"  False Negatives : {metrics['FN']}  (attacks missed)",
        f"",
        f"  Accuracy  : {metrics['accuracy']:.2%}",
        f"  Precision : {metrics['precision']:.2%}",
        f"  Recall    : {metrics['recall']:.2%}",
        f"  F1-Score  : {metrics['f1']:.2%}",
        f"{'─'*60}",
        f"  Per-type breakdown:",
    ]
    for atype, counts in sorted(metrics['per_type'].items()):
        t = counts['TP'] + counts['FN']
        d = counts['TP'] + counts['FP']
        type_recall = counts['TP'] / t if t else 0
        lines.append(
            f"    {atype:20s}  detected {counts['TP']}/{t}  ({type_recall:.0%})"
        )
    lines.append(f"{'═'*60}\n")
    print('\n'.join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="JailGuard — batch evaluation over the dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset",          default=cfg.DATASET_PATH,     help="Path to dataset.pkl")
    parser.add_argument("--keys",             default=cfg.DATASET_KEY_PATH,  help="Path to dataset-key.pkl")
    parser.add_argument("--mutator",          default=cfg.MUTATOR,           help="Mutator abbreviation")
    parser.add_argument("--n",                default=cfg.N_VARIANTS, type=int, help="Variants per input")
    parser.add_argument("--threshold",        default=cfg.THRESHOLD,  type=float, help="Detection threshold")
    parser.add_argument("--sim",              default=cfg.SIMILARITY,         help="Similarity: spacy / tfidf")
    parser.add_argument("--backend",          default=cfg.LLM_BACKEND,        help="LLM backend: ollama / huggingface / vllm / openai")
    parser.add_argument("--victim-model",     default=None, dest="victim_model",
                        help=("HuggingFace model ID to use as the victim LLM "
                              "(overrides VLLM_MODEL_ID / HF_MODEL_ID in config). "
                              "Must be cached locally when OFFLINE=True. "
                              "e.g. 'mistralai/Mistral-7B-Instruct-v0.3'"))
    parser.add_argument("--gpus",             default=None, type=int, dest="gpus",
                        help=("Number of GPUs for vLLM tensor parallelism "
                              "(overrides VLLM_TENSOR_PARALLEL in config). "
                              "e.g. --gpus 4"))
    parser.add_argument("--samples_per_type", default=cfg.SAMPLES_PER_TYPE, type=int,
                                                                              help="Items per attack type")
    parser.add_argument("--output_dir",       default=cfg.RESULTS_DIR,       help="Where to save results")
    parser.add_argument("--save_items",       action="store_true",            help="Save per-item variants/responses")
    parser.add_argument("--resume",           action="store_true",            help="Skip already-evaluated indices")
    parser.add_argument("--seed",             default=42, type=int,           help="Random seed for sampling")
    parser.add_argument("--quiet",            action="store_true",            help="Less verbose output")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  JailGuard Batch Evaluation — Reimplementation")
    print(f"{'='*60}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    dataset, dataset_key = load_dataset(args.dataset, args.keys)
    print(f"  {len(dataset):,} items loaded.")

    # ── Build evaluation set ──────────────────────────────────────────────
    eval_set = build_evaluation_set(
        dataset, dataset_key,
        attack_types     = cfg.ATTACK_TYPES_TO_EVAL,
        samples_per_type = args.samples_per_type,
        seed             = args.seed,
    )
    print(f"\nEvaluation set: {len(eval_set)} items")

    # ── Load checkpoint (resume) ──────────────────────────────────────────
    done_indices = set()
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")
    all_results = []
    if args.resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        done_indices = {r['serial_num'] for r in all_results}
        print(f"  Resuming: {len(done_indices)} items already done.")

    # ── Build LLM + Detector ──────────────────────────────────────────────
    if args.victim_model:
        print(f"  Victim model : {args.victim_model}")
    if args.gpus:
        print(f"  GPUs         : {args.gpus}")
    llm   = build_llm(
        backend         = args.backend,
        model_id        = args.victim_model,
        tensor_parallel = args.gpus,
    )
    guard = JailGuardDetector(
        llm        = llm,
        mutator    = args.mutator,
        n_variants = args.n,
        threshold  = args.threshold,
        similarity = args.sim,
        save_dir   = args.output_dir if args.save_items else None,
        verbose    = not args.quiet,
    )

    # ── Run evaluation ────────────────────────────────────────────────────
    t_start = time.time()
    remaining = [e for e in eval_set if e['serial_num'] not in done_indices]
    total_to_run = len(remaining)

    print(f"\nRunning {total_to_run} evaluations ...\n")

    for i, item in enumerate(remaining, 1):
        sn    = item['serial_num']
        label = item['label']
        data  = item['input_data']

        print(f"[{i:3d}/{total_to_run}] serial={sn}  label={label}")

        try:
            result = guard.detect(data, label=label, serial_num=sn)
            row = result.to_dict()
        except KeyboardInterrupt:
            print("\n  Interrupted by user. Saving checkpoint...")
            break
        except Exception as e:
            print(f"  ERROR on item {sn}: {e}")
            row = {
                "serial_num": sn, "label": label,
                "is_attack": None, "verdict": "ERROR",
                "max_div": None, "mean_div": None,
                "all_blocked": None, "threshold": args.threshold,
                "n_variants": args.n, "mutator": args.mutator,
                "blocked_counts": [],
            }

        all_results.append(row)

        # Save checkpoint after every item
        os.makedirs(args.output_dir, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print running accuracy
        valid = [r for r in all_results if r.get('is_attack') is not None]
        if valid:
            metrics = compute_metrics(valid)
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (total_to_run - i) if i < total_to_run else 0
            print(f"  → {result.verdict}  max_div={result.max_div:.5f}  "
                  f"running_acc={metrics['accuracy']:.1%}  "
                  f"ETA {eta/60:.1f}min")

    # ── Final metrics ─────────────────────────────────────────────────────
    valid_results = [r for r in all_results if r.get('is_attack') is not None]
    if valid_results:
        metrics = compute_metrics(valid_results)
        print_metrics(metrics, args.mutator, args.n, args.threshold)

        # Save final results
        save_results(valid_results, args.output_dir)

        # Also save metrics summary
        metrics_path = os.path.join(
            args.output_dir,
            f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metrics_path, 'w') as f:
            # Convert defaultdict for JSON serialisation
            metrics['per_type'] = dict(metrics['per_type'])
            json.dump({
                "config": {
                    "mutator": args.mutator, "n_variants": args.n,
                    "threshold": args.threshold, "backend": args.backend,
                    "similarity": args.sim,
                },
                "metrics": metrics,
            }, f, indent=2)
        print(f"  Metrics saved → {metrics_path}")
    else:
        print("No valid results to evaluate.")

    # Explicit cleanup of LLM resources to prevent exit warnings
    if hasattr(llm, 'close'):
        llm.close()


if __name__ == "__main__":
    main()
