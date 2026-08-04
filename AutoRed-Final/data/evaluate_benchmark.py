#!/usr/bin/env python3
"""
Benchmark Evaluation Script
============================
Evaluate a generator model against the benchmark dataset.

Usage:
    python evaluate_benchmark.py benchmark_v1.jsonl results.json

Results format:
    {
        "total": 200,
        "success": 120,
        "success_rate": 0.60,
        "by_type": {
            "TOKEN": {"total": 80, "success": 55, "rate": 0.6875},
            ...
        },
        "by_difficulty": {
            "easy": {"total": 50, "success": 40, "rate": 0.80},
            ...
        },
        "scenarios": [...]
    }
"""

import json
import sys
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip().lower()
    text = re.sub(r"\\s+", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return text


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Exact string match."""
    return predicted.strip() == ground_truth.strip()


def normalized_match(predicted: str, ground_truth: str) -> bool:
    """Normalized match (case, whitespace, unicode)."""
    return normalize_text(predicted) == normalize_text(ground_truth)


def semantic_match(predicted: str, ground_truth: str) -> bool:
    """Semantic match (contains ground truth or vice versa)."""
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)

    # Direct match
    if pred_norm == gt_norm:
        return True

    # Contains match
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True

    # Word overlap (for conversation/structured)
    pred_words = set(pred_norm.split())
    gt_words = set(gt_norm.split())
    if gt_words and pred_words:
        overlap = len(pred_words & gt_words) / len(gt_words)
        if overlap > 0.8:
            return True

    return False


def evaluate_scenario(scenario: dict, predicted: str) -> dict:
    """Evaluate a single scenario."""
    ground_truth = scenario["access_code"]
    code_type = scenario["access_code_type"]

    # Choose evaluation mode
    modes = {
        "TOKEN": exact_match,
        "PHRASE": normalized_match,
        "SENTENCE": normalized_match,
        "MULTILINE": normalized_match,
        "CONVERSATION": semantic_match,
        "STRUCTURED": normalized_match,
    }

    match_fn = modes.get(code_type, normalized_match)
    success = match_fn(predicted, ground_truth)

    return {
        "scenario_id": scenario["scenario_id"],
        "access_code_type": code_type,
        "difficulty": scenario["difficulty"],
        "success": success,
        "predicted": predicted,
        "ground_truth": ground_truth,
    }


def evaluate_benchmark(benchmark_path: str, results_path: str):
    """Run benchmark evaluation."""
    # Load benchmark
    scenarios = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    print(f"Loaded {len(scenarios)} benchmark scenarios")
    print("Enter predicted access codes (one per line, or 'done' to finish):")

    results = []
    for i, scenario in enumerate(scenarios):
        # In practice, this would be replaced with model inference
        predicted = input(f"Scenario {i+1}/{len(scenarios)} ({scenario['access_code_type']}): ")

        if predicted.lower() == "done":
            break

        result = evaluate_scenario(scenario, predicted)
        results.append(result)

    # Compute stats
    total = len(results)
    success = sum(1 for r in results if r["success"])
    success_rate = success / total if total else 0

    # By type
    by_type = {}
    for code_type in ["TOKEN", "PHRASE", "SENTENCE", "MULTILINE", "CONVERSATION", "STRUCTURED"]:
        type_results = [r for r in results if r["access_code_type"] == code_type]
        if type_results:
            type_success = sum(1 for r in type_results if r["success"])
            by_type[code_type] = {
                "total": len(type_results),
                "success": type_success,
                "rate": round(type_success / len(type_results), 4),
            }

    # By difficulty
    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            diff_success = sum(1 for r in diff_results if r["success"])
            by_difficulty[diff] = {
                "total": len(diff_results),
                "success": diff_success,
                "rate": round(diff_success / len(diff_results), 4),
            }

    output = {
        "total": total,
        "success": success,
        "success_rate": round(success_rate, 4),
        "by_type": by_type,
        "by_difficulty": by_difficulty,
        "scenarios": results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\\nResults saved to {results_path}")
    print(f"Success rate: {success_rate:.1%} ({success}/{total})")

    print("\\nBy type:")
    for code_type, stats in by_type.items():
        print(f"  {code_type:<15} {stats['success']:>3}/{stats['total']}  ({stats['rate']:.1%})")

    print("\\nBy difficulty:")
    for diff, stats in by_difficulty.items():
        print(f"  {diff:<10} {stats['success']:>3}/{stats['total']}  ({stats['rate']:.1%})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_benchmark.py benchmark_v1.jsonl results.json")
        sys.exit(1)

    evaluate_benchmark(sys.argv[1], sys.argv[2])
