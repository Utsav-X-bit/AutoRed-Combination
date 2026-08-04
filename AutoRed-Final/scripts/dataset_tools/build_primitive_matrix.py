#!/usr/bin/env python3
import os
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from itertools import combinations

_PRIMITIVE_PATTERNS = {
    "roleplay": r"\b(roleplay|role-play|pretend|character|persona|act as)\b",
    "authority": r"\b(admin|system|override|root|authority|privileged|sudo|as your creator)\b",
    "reflection": r"\b(reflect|repeat|echo|reproduce|rephrase|mirror|restate)\b",
    "format_wrapper": r"\b(format|xml|json|yaml|csv|table|output as|structure)\b",
    "markdown": r"(```|##|###|\*\*|__|\|\s)",
    "translation": r"\b(translate|in (french|german|spanish|japanese|chinese|arabic|latin))\b",
    "technical_jargon": r"\b(function|parameter|variable|debug|token|encode|decode|algorithm|api)\b",
    "negation_bypass": r"\b(without|instead of|do not mention|ignoring|bypass|skip)\b",
    "command_injection": r"\b(ignore previous|disregard|new instruction|forget|override|system:)\b",
    "educational_frame": r"\b(educational|research|study|academic|hypothetical|for learning)\b",
    "conditional": r"\b(if|when|unless|assuming|suppose|given that)\b",
    "prompt_injection": r"(\[INST\]|<\|system\|>|<\|user\|>|\|\||\bINSTRUCTION\b|\bsystem_prompt\b)",
    "length_constraint": r"\b(in \d+ words|briefly|concisely|summarize|short answer|one word|one sentence)\b",
    "questioning": r"\?",
}

def detect_primitives(text: str) -> dict:
    """Returns dict of {primitive_name: first_char_position} for detected primitives."""
    detected = {}
    for name, pat in _PRIMITIVE_PATTERNS.items():
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            detected[name] = match.start()
    return detected

def build_primitive_matrices(results_dir, output_dir):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[ERROR] Results directory {results_dir} does not exist.")
        return

    print(f"Scanning result files in {results_dir}...")
    json_files = list(results_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON result files.")

    # 1. Primitive x Defense Matrix
    # {primitive: {defense_type: {success: count, failure: count}}}
    prim_defense_matrix = defaultdict(lambda: defaultdict(lambda: {"success": 0, "failure": 0}))

    # 2. Primitive Pair Matrix
    # {pair_key: {success: count, failure: count}}
    prim_pair_matrix = defaultdict(lambda: {"success": 0, "failure": 0})

    # 3. Primitive Ordering Analysis
    # {pair_key: {A_first_success: count, B_first_success: count, A_first_total: count, B_first_total: count}}
    prim_ordering_matrix = defaultdict(lambda: {
        "A_first_success": 0,
        "B_first_success": 0,
        "A_first_total": 0,
        "B_first_total": 0
    })

    # Individual primitive counts to compute synergy
    individual_counts = defaultdict(lambda: {"success": 0, "failure": 0})
    total_attempts = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                run = json.load(f)
        except Exception:
            continue

        defense_type = run.get("scenario", {}).get("defense_type", "unknown")
        attempts = run.get("attempts", [])

        for attempt in attempts:
            attack_text = attempt.get("generator", {}).get("generated_attack", "")
            if not attack_text:
                continue

            # Determine success
            extractor_verified = attempt.get("extractor", {}).get("verified", False)
            ver_success = attempt.get("verification", {}).get("success", False)
            gt_leaked = attempt.get("ground_truth_found", False) or attempt.get("extractor", {}).get("ground_truth_leaked", False)
            success = bool(ver_success or extractor_verified or gt_leaked)

            detected = detect_primitives(attack_text)
            prims_list = list(detected.keys())
            total_attempts += 1

            # Update individual counts and defense matrix
            for prim in prims_list:
                if success:
                    individual_counts[prim]["success"] += 1
                    prim_defense_matrix[prim][defense_type]["success"] += 1
                else:
                    individual_counts[prim]["failure"] += 1
                    prim_defense_matrix[prim][defense_type]["failure"] += 1

            # Update pair matrix
            for prim_a, prim_b in combinations(sorted(prims_list), 2):
                pair_key = f"{prim_a} + {prim_b}"
                if success:
                    prim_pair_matrix[pair_key]["success"] += 1
                else:
                    prim_pair_matrix[pair_key]["failure"] += 1

            # Update ordering analysis
            for prim_a, prim_b in combinations(prims_list, 2):
                pos_a = detected[prim_a]
                pos_b = detected[prim_b]
                
                # We sort keys alphabetically to make the pair key unique
                sorted_pair = sorted([prim_a, prim_b])
                pair_key = f"{sorted_pair[0]} + {sorted_pair[1]}"
                
                # Check who is first (A = sorted_pair[0], B = sorted_pair[1])
                is_a_first = detected[sorted_pair[0]] < detected[sorted_pair[1]]
                
                if is_a_first:
                    prim_ordering_matrix[pair_key]["A_first_total"] += 1
                    if success:
                        prim_ordering_matrix[pair_key]["A_first_success"] += 1
                else:
                    prim_ordering_matrix[pair_key]["B_first_total"] += 1
                    if success:
                        prim_ordering_matrix[pair_key]["B_first_success"] += 1

    # Format Output Dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Primitive x Defense Matrix
    with open(out_dir / "primitive_defense_matrix_v1.json", "w", encoding="utf-8") as f:
        json.dump(prim_defense_matrix, f, indent=2)

    # 2. Compute Synergy and Save Primitive Pair Matrix
    pair_synergy_results = {}
    for pair_key, counts in prim_pair_matrix.items():
        prim_a, prim_b = pair_key.split(" + ")
        
        success_a = individual_counts[prim_a]["success"]
        total_a = success_a + individual_counts[prim_a]["failure"]
        success_b = individual_counts[prim_b]["success"]
        total_b = success_b + individual_counts[prim_b]["failure"]
        
        rate_a = success_a / total_a if total_a > 0 else 0
        rate_b = success_b / total_b if total_b > 0 else 0
        
        total_pair = counts["success"] + counts["failure"]
        pair_rate = counts["success"] / total_pair if total_pair > 0 else 0
        
        # Synergy score: pair_rate / (rate_a * rate_b)
        expected_rate = rate_a * rate_b
        synergy = pair_rate / expected_rate if expected_rate > 0 else 1.0
        
        pair_synergy_results[pair_key] = {
            "success": counts["success"],
            "failure": counts["failure"],
            "total": total_pair,
            "success_rate": pair_rate,
            "synergy": synergy
        }
        
    with open(out_dir / "primitive_pairs_matrix_v1.json", "w", encoding="utf-8") as f:
        json.dump(pair_synergy_results, f, indent=2)

    # 3. Save Primitive Ordering Analysis
    with open(out_dir / "primitive_ordering_v1.json", "w", encoding="utf-8") as f:
        json.dump(prim_ordering_matrix, f, indent=2)

    print(f"\nMatrices successfully generated and saved to {output_dir}/")
    print(f"Processed {total_attempts} attempts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Primitive Intelligence matrices")
    parser.add_argument("--results-dir", type=str, default="results/", help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default="data/", help="Output directory for matrices")
    args = parser.parse_args()

    build_primitive_matrices(args.results_dir, args.output_dir)
