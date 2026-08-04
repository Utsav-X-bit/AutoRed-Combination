#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def build_confusion_matrix(results_dir, output_file):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist.")
        return

    print(f"Scanning result files in {results_dir}...")
    json_files = list(results_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON result files.")

    confusion_records = []
    
    # Statistics counters
    confusion_pairs = Counter()
    runs_by_defense = defaultdict(int)
    correct_first_picks_by_defense = defaultdict(int)
    total_wasted_attempts = 0
    successful_runs_count = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                run = json.load(f)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        attempts = run.get("attempts", [])
        if not attempts:
            continue

        # Get first pick strategy
        first_strategy = attempts[0].get("generator", {}).get("strategy")
        if not first_strategy:
            continue

        # Find oracle strategy (strategy that led to first success)
        oracle_strategy = None
        oracle_attempt_number = None
        winning_attack = None
        oracle_attempt = None

        for attempt in attempts:
            extractor_verified = attempt.get("extractor", {}).get("verified", False)
            ver_success = attempt.get("verification", {}).get("success", False)
            gt_leaked = attempt.get("ground_truth_found", False) or attempt.get("extractor", {}).get("ground_truth_leaked", False)
            
            if ver_success or extractor_verified or gt_leaked:
                oracle_strategy = attempt.get("generator", {}).get("strategy")
                oracle_attempt_number = attempt.get("attempt_number")
                winning_attack = attempt.get("generator", {}).get("generated_attack")
                oracle_attempt = attempt
                break

        if oracle_strategy:
            successful_runs_count += 1
            d_type = run.get("scenario", {}).get("defense_type", "unknown")
            runs_by_defense[d_type] += 1
            
            if first_strategy == oracle_strategy:
                correct_first_picks_by_defense[d_type] += 1
            else:
                # First pick differed from oracle strategy
                confusion_pairs[(first_strategy, oracle_strategy)] += 1
                wasted = oracle_attempt_number - 1
                total_wasted_attempts += wasted

                record = {
                    "scenario_id": run.get("experiment", {}).get("scenario_id", "unknown"),
                    "defense_type": d_type,
                    "access_code_type": run.get("scenario", {}).get("access_code_type", "unknown"),
                    "planner_choice": first_strategy,
                    "oracle_strategy": oracle_strategy,
                    "attempts_to_win": oracle_attempt_number,
                    "pre_defense": run.get("scenario", {}).get("pre_defense", ""),
                    "post_defense": run.get("scenario", {}).get("post_defense", ""),
                    "winning_attack": winning_attack,
                    "total_attempts": len(attempts)
                }
                confusion_records.append(record)

    # Save to JSONL
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in confusion_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved {len(confusion_records)} confusion records to {output_file}\n")

    # Output statistics
    print("==============================================================")
    print("PLANNER CONFUSION STATISTICS")
    print("==============================================================")
    print(f"Total Successful Runs Evaluated: {successful_runs_count}")
    print(f"Total Wasted Attempts: {total_wasted_attempts} (attempts that could be saved by perfect 1st-pick)")
    print(f"Average Wasted Attempts per incorrect 1st-pick: {total_wasted_attempts / max(len(confusion_records), 1):.2f}")
    print("--------------------------------------------------------------")
    print("Accuracy by Defense Type (Correct 1st-pick / Total Successful Runs):")
    for d_type in sorted(runs_by_defense.keys()):
        total_runs = runs_by_defense[d_type]
        correct = correct_first_picks_by_defense[d_type]
        acc = (correct / total_runs) * 100 if total_runs > 0 else 0
        print(f"  - {d_type:20s} : {acc:5.1f}% ({correct}/{total_runs})")
    
    print("--------------------------------------------------------------")
    print("Top-20 (Planner Choice -> Oracle Strategy) Confusion Pairs:")
    for (planner, oracle), count in confusion_pairs.most_common(20):
        pct = (count / len(confusion_records)) * 100 if confusion_records else 0
        print(f"  - {planner:25s} -> {oracle:25s} : {count:3d} ({pct:5.1f}%)")
    print("==============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build planner confusion dataset")
    parser.add_argument("--results-dir", type=str, default="results/", help="Path to run results folder")
    parser.add_argument("--output", type=str, default="data/planner_confusion_v1.jsonl", help="Path to output JSONL file")
    args = parser.parse_args()

    build_confusion_matrix(args.results_dir, args.output)
