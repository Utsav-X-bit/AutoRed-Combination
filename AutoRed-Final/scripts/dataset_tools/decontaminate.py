import json
import os
import bz2
import pandas as pd
from tqdm import tqdm

def get_hash(opening, closing, code):
    return f"{str(opening).strip()}|{str(closing).strip()}|{str(code).strip()}"

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data")
    
    # 1. Load the 200 holdout scenarios from v2 (which were text-hashed)
    holdout_file = os.path.join(data_dir, "benchmark_holdout_v2.jsonl")
    
    holdout_hashes = set()
    with open(holdout_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            h = get_hash(d.get("opening_defense", ""), d.get("closing_defense", ""), d.get("access_code", ""))
            holdout_hashes.add(h)
            
    print(f"Loaded {len(holdout_hashes)} unique holdout scenario text signatures.")
    
    # 2. Map them back to real defense_ids from raw_dump_defenses
    raw_file = os.path.join(project_root, "experiment", "raw_dump_defenses.jsonl.bz2")
    real_holdout_ids = set()
    
    print("Scanning raw_dump_defenses to find original IDs...")
    with bz2.open(raw_file, 'rt') as f:
        for line in tqdm(f):
            if not line.strip(): continue
            d = json.loads(line)
            h = get_hash(d.get("opening_defense", ""), d.get("closing_defense", ""), d.get("access_code", ""))
            if h in holdout_hashes:
                real_holdout_ids.add(str(d.get("defense_id")))
                
    print(f"Found {len(real_holdout_ids)} original defense_ids matching the holdout set.")
    
    # Write them out so we have them
    with open(os.path.join(data_dir, "holdout_original_ids.txt"), "w") as f:
        for rid in real_holdout_ids:
            f.write(f"{rid}\n")
            
    # 3. Decontaminate
    files_to_clean = [
        "autored_successes_v1.jsonl",
        "autored_positive_v1.jsonl",
        "autored_verified_v1.jsonl",
        "autored_extractor_failures_v1.jsonl",
        "generator_sft_dataset.jsonl",
        "strategy_predictor_train.jsonl"
    ]
    
    import shutil
    for filename in files_to_clean:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename} - not found.")
            continue
            
        clean_filepath = filepath + ".clean"
        kept = 0
        dropped = 0
        
        with open(filepath, 'r') as fin, open(clean_filepath, 'w') as fout:
            for line in fin:
                if not line.strip(): continue
                d = json.loads(line)
                
                # Check scenario_id if present
                sid = str(d.get("scenario_id", "")).replace("bench_", "")
                
                # SFT dataset has a different schema ("prompt" instead of opening_defense)
                # But it has 'scenario_id' too! Wait, let's check if it does.
                # Just in case, we also check text overlap for SFT dataset
                
                is_contaminated = False
                if sid and sid in real_holdout_ids:
                    is_contaminated = True
                else:
                    h = get_hash(d.get("opening_defense", ""), d.get("closing_defense", ""), d.get("access_code", ""))
                    if h in holdout_hashes:
                        is_contaminated = True
                    elif "prompt" in d:
                        prompt_text = d["prompt"]
                        for holdout_d in holdout_hashes:
                            opening = holdout_d.split('|')[0]
                            if opening and opening in prompt_text:
                                is_contaminated = True
                                break
                            
                if is_contaminated:
                    dropped += 1
                else:
                    fout.write(line)
                    kept += 1
                    
        print(f"Processed {filename}: kept {kept}, dropped {dropped} ({(dropped/(kept+dropped)*100) if (kept+dropped)>0 else 0:.1f}%)")
        shutil.move(clean_filepath, filepath)

if __name__ == "__main__":
    main()
