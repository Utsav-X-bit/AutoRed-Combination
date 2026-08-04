import json
import os
from pathlib import Path
import glob

DATA_DIR = Path("/home/utsav/Github/Research/AutoRed/data")
RESULTS_DIR = Path("/home/utsav/Github/Research/AutoRed/results")

def main():
    # 1. Load defense lookup
    print("Loading defense lookup from classifier datasets...")
    defense_lookup = {}
    for part in ["Part1", "Part2"]:
        path = DATA_DIR / f"defense_classifier_dataset-{part}.jsonl"
        if not path.exists():
            continue
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                defense_lookup[str(data["defense_id"])] = data.get("access_code_type", "UNKNOWN")
                
    # 2. Iterate runs
    print("Scanning run files...")
    run_files = glob.glob(str(RESULTS_DIR / "run_*.json"))
    
    missed_leaks = []
    
    for rf in run_files:
        with open(rf, "r") as f:
            try:
                run_data = json.load(f)
            except:
                continue
                
        run_id = run_data.get("experiment", {}).get("run_id", "unknown")
        
        raw_entry = run_data.get("raw_dataset_entry", {})
        did = str(raw_entry.get("defense_id", "unknown"))
        
        gt = run_data.get("ground_truth", {})
        access_code = gt.get("access_code", "")
        ac_type = defense_lookup.get(did, "UNKNOWN")
        
        for attempt in run_data.get("attempts", []):
            is_leaked = attempt.get("ground_truth_found", False)
            is_extracted = attempt.get("extractor_match", False)
            
            if is_leaked and not is_extracted:
                # Calculate miss reason
                victim_response = attempt.get("victim", {}).get("raw_output", "")
                candidates = attempt.get("extractor", {}).get("ranked_candidates", [])
                
                reason = "UNKNOWN"
                if len(candidates) == 0:
                    reason = "NO_CANDIDATE"
                else:
                    best_score = max([c.get("score", 0) for c in candidates])
                    if best_score < 7.0: # threshold is 7.0 usually
                        reason = "LOW_SCORE"
                    elif "{" in victim_response and "}" in victim_response and "JSON" in victim_response.upper():
                        reason = "JSON_PARSE_ERROR"
                    else:
                        reason = "TRUNCATED"
                        
                row = {
                    "run_id": run_id,
                    "access_code": access_code,
                    "access_code_type": ac_type,
                    "victim_response": victim_response,
                    "candidate_pool": candidates,
                    "miss_reason": reason
                }
                missed_leaks.append(row)
                
    print(f"Found {len(missed_leaks)} missed leaks.")
    out_path = DATA_DIR / "extractor_errors_v1.jsonl"
    with open(out_path, "w") as f:
        for m in missed_leaks:
            f.write(json.dumps(m) + "\n")
            
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
