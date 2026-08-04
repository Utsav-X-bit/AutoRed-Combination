import os
import json
from collections import defaultdict
from pathlib import Path


def build_strategy_kb(
    success_data_path="data/autored_successes_v1.jsonl",
    failure_data_path="data/autored_failures_v1.jsonl",
    defense_data_path="data/defense_classifier_dataset.jsonl",
    output_path="data/strategy_knowledge_base.json",
    verbose=True,
):
    """Rebuild the strategy effectiveness matrix from accumulated success/failure records."""

    # Map defense_id -> primary_type
    defense_to_type = {}
    if verbose:
        print(f"Loading defense intelligence from {defense_data_path}...")
    if not os.path.exists(defense_data_path):
        if verbose:
            print(f"Error: {defense_data_path} not found. Please run Phase 1 first.")
        return

    with open(defense_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            defense_to_type[data["defense_id"]] = data["primary_type"]

    if verbose:
        print(f"Loaded {len(defense_to_type)} mapped defenses.")

    # Data structures for tracking
    # matrix[defense_type][strategy] = {"successes": 0, "failures": 0}
    matrix = defaultdict(lambda: defaultdict(lambda: {"successes": 0, "failures": 0}))
    global_strategy_stats = defaultdict(lambda: {"successes": 0, "failures": 0})
    global_defense_stats = defaultdict(lambda: {"successes": 0, "failures": 0})

    def process_file(path, is_success):
        if not os.path.exists(path):
            if verbose:
                print(f"Warning: {path} not found.")
            return 0

        count = 0
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                scenario_id = str(data["scenario_id"])
                strategy = data["strategy"]

                def_type = defense_to_type.get(scenario_id, "unknown")

                if is_success:
                    matrix[def_type][strategy]["successes"] += 1
                    global_strategy_stats[strategy]["successes"] += 1
                    global_defense_stats[def_type]["successes"] += 1
                else:
                    matrix[def_type][strategy]["failures"] += 1
                    global_strategy_stats[strategy]["failures"] += 1
                    global_defense_stats[def_type]["failures"] += 1

                count += 1
        return count

    if verbose:
        print(f"Processing successes...")
    s_count = process_file(success_data_path, is_success=True)
    if verbose:
        print(f"Processing failures...")
    f_count = process_file(failure_data_path, is_success=False)

    if verbose:
        print(f"Processed {s_count} successes and {f_count} failures.")

    # Calculate rates
    final_matrix = {}
    for d_type, strategies in matrix.items():
        final_matrix[d_type] = {}
        for strat, counts in strategies.items():
            total = counts["successes"] + counts["failures"]
            rate = (counts["successes"] / total * 100) if total > 0 else 0
            final_matrix[d_type][strat] = {
                "success_rate": round(rate, 2),
                "total_attempts": total,
                "successes": counts["successes"],
                "failures": counts["failures"],
            }

    # Format output
    output_data = {
        "matrix": final_matrix,
        "global_strategy_stats": {},
        "global_defense_stats": {},
    }

    for strat, counts in global_strategy_stats.items():
        total = counts["successes"] + counts["failures"]
        rate = (counts["successes"] / total * 100) if total > 0 else 0
        output_data["global_strategy_stats"][strat] = {
            "success_rate": round(rate, 2),
            "total_attempts": total,
        }

    for d_type, counts in global_defense_stats.items():
        total = counts["successes"] + counts["failures"]
        rate = (counts["successes"] / total * 100) if total > 0 else 0
        output_data["global_defense_stats"][d_type] = {
            "success_rate": round(rate, 2),
            "total_attempts": total,
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nSaved Effectiveness Matrix to {output_path}")

        print("\nTop Strategy per Defense Type:")
        for d_type, strategies in final_matrix.items():
            # Filter strategies with at least 5 attempts to avoid noise
            valid_strats = {k: v for k, v in strategies.items() if v["total_attempts"] >= 5}
            if not valid_strats:
                continue
            best_strat = max(valid_strats.items(), key=lambda x: x[1]["success_rate"])
            print(
                f"  {d_type:<18} -> {best_strat[0]:<25} "
                f"({best_strat[1]['success_rate']}% on {best_strat[1]['total_attempts']} attempts)"
            )


def main():
    build_strategy_kb()


if __name__ == "__main__":
    main()
