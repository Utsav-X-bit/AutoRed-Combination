import json
import argparse
from pathlib import Path
from collections import defaultdict


def _load_defense_type_map(data_dir="data"):
    """Build defense_id -> primary_type mapping from the classifier dataset if available."""
    defense_to_type = {}
    classifier_path = Path(data_dir) / "defense_classifier_dataset.jsonl"
    if not classifier_path.exists():
        return defense_to_type
    try:
        with open(classifier_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                defense_to_type[str(d.get("defense_id"))] = d.get("primary_type", "unknown")
    except Exception:
        pass
    return defense_to_type


def build_oracle(results_dirs, output_file, data_dir="data"):
    # Transition counts: (From_Strategy, To_Strategy) -> success count
    transition_success_counts = defaultdict(int)
    first_strategy_success = defaultdict(lambda: defaultdict(int))

    run_files = []
    for d in results_dirs:
        path = Path(d)
        # Recursive so run traces in results/<date>/<time>/run_*.json are included.
        run_files.extend(path.rglob("run_*.json"))

    defense_to_type = _load_defense_type_map(data_dir)

    print(f"Mining {len(run_files)} runs for oracle...")

    for run_file in run_files:
        try:
            with open(run_file, "r") as f:
                run_data = json.load(f)
        except Exception:
            continue

        attempts = run_data.get("attempts", [])
        if not attempts:
            continue

        run_success = run_data.get("result", {}).get("verified_success", False) or \
                      run_data.get("result", {}).get("ground_truth_success", False)

        raw_defense_type = run_data.get("raw_dataset_entry", {}).get("defense_type")
        defense_id = str(run_data.get("raw_dataset_entry", {}).get("defense_id", ""))
        defense_type = raw_defense_type or defense_to_type.get(defense_id, "unknown")

        # Analyze first strategy
        first_strat = attempts[0].get("generator", {}).get("strategy", "unknown")
        if run_success and attempts[0].get("verification_success", False):
            first_strategy_success[defense_type][first_strat] += 1

        # Transitions
        for i in range(len(attempts) - 1):
            strat_curr = attempts[i].get("generator", {}).get("strategy", "unknown")
            strat_next = attempts[i+1].get("generator", {}).get("strategy", "unknown")

            if strat_curr == strat_next:
                continue

            # If the next strategy succeeded, count it heavily.
            if attempts[i+1].get("verification_success", False) or attempts[i+1].get("ground_truth_found", False):
                transition_success_counts[(strat_curr, strat_next)] += 1

    # Build a lookup table: current_strategy -> [list of best next strategies sorted by success]
    oracle_transitions = {}

    # Group by current strategy
    grouped = defaultdict(list)
    for (s_curr, s_next), succ in transition_success_counts.items():
        grouped[s_curr].append((s_next, succ))

    for s_curr, nexts in grouped.items():
        # Sort by success descending
        nexts.sort(key=lambda x: x[1], reverse=True)
        # Store just the strategy names in order of preference
        oracle_transitions[s_curr] = [s[0] for s in nexts if s[1] > 0]

    # Also save best first strategies by defense
    best_first = {}
    for d_type, strats in first_strategy_success.items():
        sorted_strats = sorted(strats.items(), key=lambda x: x[1], reverse=True)
        best_first[d_type] = [s[0] for s in sorted_strats if s[1] > 0]

    oracle = {
        "transitions": oracle_transitions,
        "best_first": best_first
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(oracle, f, indent=2)

    print(f"Oracle rules saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="data/oracle_rules.json")
    args = parser.parse_args()
    build_oracle(args.results_dirs, args.output)
