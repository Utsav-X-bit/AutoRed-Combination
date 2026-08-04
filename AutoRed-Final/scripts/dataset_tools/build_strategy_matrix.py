import json
import os
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/home/utsav/Github/Research/AutoRed/data")

def main():
    print("Loading defense lookup from classifier datasets...")
    defense_lookup = {}
    
    for part in ["Part1", "Part2"]:
        path = DATA_DIR / f"defense_classifier_dataset-{part}.jsonl"
        if not path.exists():
            print(f"Warning: {path} not found.")
            continue
            
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                did = str(data["defense_id"])
                defense_lookup[did] = {
                    "primary_type": data.get("primary_type", "unknown"),
                    "secondary_flags": data.get("secondary_flags", []),
                    "access_code_type": data.get("access_code_type", "UNKNOWN"),
                    "defense_length": data.get("word_count", 0)
                }
                
    print(f"Loaded {len(defense_lookup)} defenses into lookup.")
    
    raw_data = []
    
    # matrix[tuple][strategy] = {"successes": 0, "failures": 0}
    matrix = defaultdict(lambda: defaultdict(lambda: {"successes": 0, "failures": 0}))
    
    def process_file(path, is_success_file):
        if not path.exists():
            print(f"Warning: {path} not found.")
            return 0
            
        count = 0
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                sid = str(data["scenario_id"])
                
                # Fetch defense metadata
                d_meta = defense_lookup.get(sid, {
                    "primary_type": "unknown",
                    "secondary_flags": [],
                    "access_code_type": data.get("access_code_type", "UNKNOWN"),
                    "defense_length": 0
                })
                
                comp = data.get("defense_complexity", "unknown")
                strat = data["strategy"]
                
                # We consider "success" as generator success, but maybe we want "verified" for the real success?
                # The prompt says: Label: success (boolean), ground_truth_leaked (boolean), verified (boolean)
                # For strategy effectiveness, let's track the actual `success` boolean.
                # Actually, the task says: labels: `success`, `ground_truth_leaked`, `verified`
                # Let's track verified for the aggregator, or success? Let's track both!
                
                row = {
                    "scenario_id": sid,
                    "primary_type": d_meta["primary_type"],
                    "secondary_flags": d_meta["secondary_flags"],
                    "access_code_type": d_meta["access_code_type"],
                    "defense_complexity": comp,
                    "defense_length": d_meta["defense_length"],
                    "strategy_used": strat,
                    "success": data.get("success", is_success_file),
                    "ground_truth_leaked": data.get("ground_truth_leaked", False),
                    "verified": data.get("verification_success", False)
                }
                raw_data.append(row)
                
                # Aggregation
                # Tuple: (primary_type, access_code_type, defense_complexity)
                tuple_key = f"{d_meta['primary_type']}|{d_meta['access_code_type']}|{comp}"
                
                # Define success metric for aggregation. Let's use `verified` as the gold standard?
                # The generator's 'success' is just self-assessed and highly inflated (56.6%).
                # We will use verified for real matrix, or perhaps just `success` to match historical?
                # The plan states we want to train on "what succeeded". The generator success is highly inflated, verified is real.
                # Let's just track BOTH in the matrix!
                
                stats = matrix[tuple_key][strat]
                if data.get("verification_success", False):
                    stats["successes"] += 1
                else:
                    stats["failures"] += 1
                    
                count += 1
        return count

    print("Processing successes...")
    s_count = process_file(DATA_DIR / "autored_successes_v1.jsonl", True)
    
    print("Processing failures...")
    f_count = process_file(DATA_DIR / "autored_failures_v1.jsonl", False)
    
    print(f"Processed {s_count} successes and {f_count} failures.")
    
    print("Writing raw data...")
    raw_path = DATA_DIR / "strategy_matrix_raw_v1.jsonl"
    with open(raw_path, "w") as f:
        for row in raw_data:
            f.write(json.dumps(row) + "\n")
            
    print("Writing aggregated matrix...")
    agg_matrix = {}
    for t_key, strats in matrix.items():
        agg_matrix[t_key] = {}
        for strat, counts in strats.items():
            tot = counts["successes"] + counts["failures"]
            rate = (counts["successes"] / tot * 100) if tot > 0 else 0
            agg_matrix[t_key][strat] = {
                "success_rate": round(rate, 2),
                "total_attempts": tot,
                "successes": counts["successes"],
                "failures": counts["failures"]
            }
            
    agg_path = DATA_DIR / "strategy_matrix_agg_v1.json"
    with open(agg_path, "w") as f:
        json.dump(agg_matrix, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
