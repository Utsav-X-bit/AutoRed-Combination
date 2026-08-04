#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

def main():
    parser = argparse.ArgumentParser(description="Build training dataset for Strategy Predictor")
    parser.add_argument("--successes", type=str, default=str(DATA_DIR / "autored_successes_v1.jsonl"), help="Path to successes jsonl")
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "strategy_predictor_train.jsonl"), help="Output dataset path")
    args = parser.parse_args()

    successes_file = Path(args.successes)
    output_file = Path(args.output)
    
    # 1. Load Defense Features
    print("Loading defense classifier datasets...")
    defense_features = {}
    
    for part in ["defense_classifier_dataset-Part1.jsonl", "defense_classifier_dataset-Part2.jsonl"]:
        part_path = DATA_DIR / part
        if not part_path.exists():
            print(f"Warning: {part_path} does not exist.")
            continue
            
        with open(part_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                def_id = data.get("defense_id")
                if def_id:
                    defense_features[str(def_id)] = {
                        "primary_type": data.get("primary_type", "UNKNOWN"),
                        "secondary_flags": data.get("secondary_flags", [])
                    }
                    
    print(f"Loaded {len(defense_features)} defense feature mappings.")

    # 2. Match with Successes
    print(f"Processing successes from {successes_file}...")
    
    if not successes_file.exists():
        print(f"Error: Successes file {successes_file} not found.")
        return
        
    dataset = []
    missing_defenses = 0
    
    with open(successes_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            scenario_id = str(data.get("scenario_id"))
            strategy = data.get("strategy")
            
            if not strategy or not scenario_id:
                continue
                
            if scenario_id in defense_features:
                features = defense_features[scenario_id]
                dataset.append({
                    "scenario_id": scenario_id,
                    "primary_type": features["primary_type"],
                    "secondary_flags": features["secondary_flags"],
                    "strategy": strategy
                })
            else:
                missing_defenses += 1

    print(f"Matched {len(dataset)} successful attempts to defense features.")
    if missing_defenses > 0:
        print(f"Warning: {missing_defenses} successes could not be matched to a defense_id.")
        
    # 3. Save Output
    print(f"Saving dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
