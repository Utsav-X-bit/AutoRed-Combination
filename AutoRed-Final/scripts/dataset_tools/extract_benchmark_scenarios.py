import json
import os
import glob
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
OUTPUT_FILE = Path("data/benchmark_v2.jsonl")

def main():
    print(f"Scanning {RESULTS_DIR} for today's benchmark runs...")
    
    # We want runs from today (2026-06-22)
    files = glob.glob(str(RESULTS_DIR / "run_20260622_*.json"))
    print(f"Found {len(files)} run JSONs from today.")
    
    scenarios = {}
    
    for fpath in files:
        with open(fpath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        # Check if it was part of a benchmark
        exp = data.get("experiment", {})
        if not exp.get("benchmark_mode", False):
            continue
            
        # The scenario data we want to freeze
        raw_entry = data.get("raw_dataset_entry")
        if not raw_entry:
            continue
            
        # Generate a unique hash based on content instead of fake defense_id
        content_key = f"{raw_entry.get('opening_defense', '')}|{raw_entry.get('closing_defense', '')}|{raw_entry.get('access_code', '')}"
        
        if content_key in scenarios:
            continue
            
        scenarios[content_key] = raw_entry

    print(f"Extracted {len(scenarios)} unique benchmark scenarios.")
    
    # Write to benchmark_v2.jsonl
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for idx, s in enumerate(scenarios.values()):
            s["defense_id"] = f"bench_v2_{idx:04d}"
            f.write(json.dumps(s) + "\n")
            
    print(f"Successfully saved {len(scenarios)} scenarios to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
