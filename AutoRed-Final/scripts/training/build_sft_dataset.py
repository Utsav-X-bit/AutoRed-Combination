#!/usr/bin/env python3
"""
Build SFT training dataset from AutoRed verified/positive results.

Joins verified/positive JSONL with run files to extract defense text,
then constructs 3 training variants:
  A: Strategy -> Attack
  B: Defense + Strategy -> Attack  
  C: Defense + Response + Strategy -> Attack (best, matches AutoRed workflow)

Usage:
    python scripts/training/build_sft_dataset.py --mode all
    python scripts/training/build_sft_dataset.py --mode variant_c --dataset verified
"""

import argparse
import json
import glob
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "training" / "sft_data"


def load_jsonl(path):
    """Load JSONL file, return list of dicts."""
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {i+1} in {path.name}: {e}")
    return samples


def build_run_index():
    """Build scenario_id -> run_data mapping from results/."""
    print("Building run file index...")
    index = {}
    files = sorted(glob.glob(str(RESULTS_DIR / "run_*.json")))
    loaded = 0
    for fpath in files:
        try:
            with open(fpath) as fh:
                d = json.load(fh)
            sid = str(d.get("experiment", {}).get("scenario_id", ""))
            if sid:
                index[sid] = d
                loaded += 1
        except Exception:
            continue
    print(f"  Indexed {loaded} unique scenarios from {len(files)} files")
    return index


def extract_defense_text(run_data):
    """Extract combined defense text from a run file."""
    scenario = run_data.get("scenario", {})
    pre = scenario.get("pre_defense", "").strip()
    post = scenario.get("post_defense", "").strip()
    parts = [p for p in [pre, post] if p]
    return "\n\n".join(parts) if parts else "[No defense text]"


def build_variant_a(sample, _run_data):
    """Strategy -> Attack (simplest)."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Strategy:\n{sample['strategy']}\n\nGenerate the next attack."
            },
            {
                "role": "assistant",
                "content": sample["attack"]
            }
        ],
        "metadata": {
            "strategy": sample["strategy"],
            "variant": "A",
            "scenario_id": sample.get("scenario_id"),
            "defense_complexity": sample.get("defense_complexity"),
            "access_code_type": sample.get("access_code_type"),
            "verification_success": sample.get("verification_success"),
            "ground_truth_leaked": sample.get("ground_truth_leaked")
        }
    }


def build_variant_b(sample, run_data):
    """Defense + Strategy -> Attack."""
    defense = extract_defense_text(run_data)
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Defense:\n{defense}\n\nStrategy:\n{sample['strategy']}\n\nGenerate the next attack."
            },
            {
                "role": "assistant",
                "content": sample["attack"]
            }
        ],
        "metadata": {
            "strategy": sample["strategy"],
            "variant": "B",
            "scenario_id": sample.get("scenario_id"),
            "defense_complexity": sample.get("defense_complexity"),
            "access_code_type": sample.get("access_code_type"),
            "verification_success": sample.get("verification_success"),
            "ground_truth_leaked": sample.get("ground_truth_leaked")
        }
    }


def build_variant_c(sample, run_data):
    """Defense + Previous Response + Strategy -> Attack (best, matches AutoRed)."""
    defense = extract_defense_text(run_data)
    response = sample.get("victim_response", "").strip()
    if not response:
        response = "[No previous response — first attempt]"

    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Defense:\n{defense}\n\n"
                    f"Previous Response:\n{response}\n\n"
                    f"Strategy:\n{sample['strategy']}\n\n"
                    f"Generate the next attack."
                )
            },
            {
                "role": "assistant",
                "content": sample["attack"]
            }
        ],
        "metadata": {
            "strategy": sample["strategy"],
            "variant": "C",
            "scenario_id": sample.get("scenario_id"),
            "defense_complexity": sample.get("defense_complexity"),
            "access_code_type": sample.get("access_code_type"),
            "verification_success": sample.get("verification_success"),
            "ground_truth_leaked": sample.get("ground_truth_leaked")
        }
    }


VARIANT_BUILDERS = {
    "a": build_variant_a,
    "b": build_variant_b,
    "c": build_variant_c,
}


def split_train_val(data, val_ratio=0.2, seed=42):
    """Split data into train/validation, stratified by strategy."""
    random.seed(seed)

    # Group by strategy
    by_strategy = {}
    for entry in data:
        s = entry["metadata"]["strategy"]
        by_strategy.setdefault(s, []).append(entry)

    train, val = [], []
    for strategy, entries in by_strategy.items():
        random.shuffle(entries)
        n_val = max(1, int(len(entries) * val_ratio))
        val.extend(entries[:n_val])
        train.extend(entries[n_val:])

    return train, val


def process_dataset(dataset_name, variant, run_index, val_ratio):
    """Process one dataset + variant combination."""
    source_file = DATA_DIR / f"autored_{dataset_name}_v1.jsonl"
    if not source_file.exists():
        print(f"  Error: {source_file} not found")
        return None

    samples = load_jsonl(source_file)
    print(f"  Loaded {len(samples)} samples from {source_file.name}")

    build_fn = VARIANT_BUILDERS[variant]

    # Join with run index and build entries
    training_data = []
    joined = 0
    missed = 0

    for sample in samples:
        sid = str(sample.get("scenario_id", ""))
        if sid in run_index:
            try:
                entry = build_fn(sample, run_index[sid])
                training_data.append(entry)
                joined += 1
            except Exception as e:
                print(f"  Error processing scenario {sid}: {e}")
                missed += 1
        else:
            missed += 1

    print(f"  Joined: {joined}, Missed (no run file): {missed}")

    if not training_data:
        return None

    # Split
    train, val = split_train_val(training_data, val_ratio)

    # Save
    prefix = f"variant{variant}_{dataset_name}"
    train_path = OUTPUT_DIR / f"{prefix}_train.jsonl"
    val_path = OUTPUT_DIR / f"{prefix}_val.jsonl"

    with open(train_path, "w") as f:
        for entry in train:
            f.write(json.dumps(entry) + "\n")

    with open(val_path, "w") as f:
        for entry in val:
            f.write(json.dumps(entry) + "\n")

    # Stats
    strat_counts = {}
    for entry in train:
        s = entry["metadata"]["strategy"]
        strat_counts[s] = strat_counts.get(s, 0) + 1

    print(f"  Saved: {train_path.name} ({len(train)}), {val_path.name} ({len(val)})")
    print(f"  Strategy distribution (train):")
    for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    return {
        "train": len(train),
        "val": len(val),
        "strategies": strat_counts
    }


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from AutoRed results")
    parser.add_argument("--mode", choices=["variant_a", "variant_b", "variant_c", "all"],
                        default="all", help="Which variant(s) to build")
    parser.add_argument("--dataset", choices=["verified", "positive", "both"],
                        default="verified", help="Source dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build run index
    run_index = build_run_index()

    # Determine what to build
    if args.mode == "all":
        variants = ["a", "b", "c"]
    else:
        variants = [args.mode.replace("variant_", "")]

    if args.dataset == "both":
        datasets = ["verified", "positive"]
    else:
        datasets = [args.dataset]

    summary = {}

    for dataset_name in datasets:
        for variant in variants:
            print(f"\n{'='*60}")
            print(f"Building Variant {variant.upper()} from {dataset_name}")
            print(f"{'='*60}")

            result = process_dataset(dataset_name, variant, run_index, args.val_ratio)
            if result:
                summary[f"variant{variant}_{dataset_name}"] = result

    # Save summary
    if summary:
        summary_path = OUTPUT_DIR / "dataset_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
