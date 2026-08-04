import os
import json
import glob
import hashlib
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def get_hash(op, cl):
    text = f"{str(op).strip()}|{str(cl).strip()}"
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def clean_gen_name(name):
    if not name:
        return "Unknown"
    name = str(name)
    if "generator_sft_adapter" in name or "qlo_verified_v1" in name:
        return "SFT Generator (Verified)"
    elif "qlo_positive_v1" in name:
        return "SFT Generator (Positive)"
    elif "generator_dpo_adapter" in name:
        return "DPO Generator"
    elif "Llama-3.1-8B-Lexi-Uncensored-V2" in name:
        return "Lexi Base (Uncensored)"
    elif "t5-base" in name:
        return "Legacy T5"
    return name

def main():
    print("=== Loading Reference Datasets ===")
    lookup = {}
    
    # 1. Load benchmark_v1.jsonl (high quality metadata)
    v1_path = "data/benchmark_v1.jsonl"
    if os.path.exists(v1_path):
        print(f"Reading {v1_path}...")
        try:
            df_v1 = pd.read_json(v1_path, lines=True)
            for _, row in df_v1.iterrows():
                h = get_hash(row.get('opening_defense'), row.get('closing_defense'))
                lookup[h] = {
                    'access_code_type': row.get('access_code_type', 'UNKNOWN'),
                    'defense_type': 'UNKNOWN',
                    'difficulty': row.get('difficulty', 'UNKNOWN')
                }
        except Exception as e:
            print("Error loading benchmark_v1.jsonl:", e)

    # 2. Load defense classifier datasets (Part 1 and Part 2)
    for part in ["Part1", "Part2"]:
        part_path = f"data/defense_classifier_dataset-{part}.jsonl"
        if os.path.exists(part_path):
            print(f"Reading {part_path}...")
            try:
                # read in chunks or stream to save memory
                with open(part_path, 'r') as f:
                    for line in f:
                        row = json.loads(line)
                        h = get_hash(row.get('opening_defense'), row.get('closing_defense'))
                        # Only update if not already set or unknown
                        meta = lookup.setdefault(h, {
                            'access_code_type': 'UNKNOWN',
                            'defense_type': 'UNKNOWN',
                            'difficulty': 'UNKNOWN'
                        })
                        if row.get('access_code_type') and meta['access_code_type'] == 'UNKNOWN':
                            meta['access_code_type'] = row.get('access_code_type')
                        if row.get('defense_type') and meta['defense_type'] == 'UNKNOWN':
                            meta['defense_type'] = row.get('defense_type')
                        if row.get('difficulty') and meta['difficulty'] == 'UNKNOWN':
                            meta['difficulty'] = row.get('difficulty')
            except Exception as e:
                print(f"Error loading {part_path}:", e)
                
    print(f"Metadata lookup table built with {len(lookup)} unique entries.")

    print("\n=== Finding Run JSON Files ===")
    results_files = glob.glob("results/**/*.json", recursive=True)
    results_bak_files = glob.glob("results-bak/**/*.json", recursive=True)
    all_files = results_files + results_bak_files
    print(f"Found {len(results_files)} files in results/")
    print(f"Found {len(results_bak_files)} files in results-bak/")
    print(f"Total files to parse: {len(all_files)}")

    records = []
    
    print("\n=== Parsing Run Files ===")
    for fn in tqdm(all_files, desc="Parsing JSON runs"):
        # Skip summary files or temporary configurations
        if "summary" in fn or "metadata" in fn:
            continue
        try:
            with open(fn, 'r') as f:
                d = json.load(f)
                
            exp = d.get('experiment', {})
            models = d.get('models', {})
            scenario = d.get('scenario', {})
            result = d.get('result', {})
            summary = d.get('summary', {})
            
            # Skip if not a valid run JSON
            if not exp or not result:
                continue
                
            run_id = exp.get('run_id', 'unknown')
            
            # Determine run date
            # run_id pattern: run_YYYYMMDD_HHMMSS_...
            date_part = None
            if len(run_id.split('_')) > 1:
                date_part = run_id.split('_')[1]
                
            # If date_part is digits (like YYYYMMDD), check if after June 22, 2026
            strategy_selector = "No"
            if date_part and date_part.isdigit() and len(date_part) == 8:
                try:
                    run_date = datetime.strptime(date_part, "%Y%m%d")
                    cutoff_date = datetime(2026, 6, 22)
                    if run_date >= cutoff_date:
                        strategy_selector = "Yes"
                except ValueError:
                    pass
            elif date_part and len(date_part) > 10:  # timestamp
                try:
                    ts = int(date_part) / 1000.0  # ms to s
                    run_date = datetime.fromtimestamp(ts)
                    cutoff_date = datetime(2026, 6, 22)
                    if run_date >= cutoff_date:
                        strategy_selector = "Yes"
                except:
                    pass

            op = scenario.get('pre_defense', '')
            cl = scenario.get('post_defense', '')
            h = get_hash(op, cl)
            meta = lookup.get(h, {})
            
            access_code_type = meta.get('access_code_type', 'UNKNOWN')
            defense_type = meta.get('defense_type', 'UNKNOWN')
            difficulty = meta.get('difficulty', 'UNKNOWN')
            
            # Fallback heuristics for access code type if still unknown
            if access_code_type == 'UNKNOWN':
                ac = scenario.get('access_code', '')
                if not ac:
                    access_code_type = 'UNKNOWN'
                elif '\n' in ac:
                    access_code_type = 'MULTILINE'
                elif len(ac.split()) > 4:
                    access_code_type = 'SENTENCE'
                elif len(ac.split()) > 1:
                    access_code_type = 'PHRASE'
                else:
                    access_code_type = 'TOKEN'
                    
            # Determine SFT/DPO Run group
            gen_name = models.get('generator', {}).get('name', '')
            run_group = clean_gen_name(gen_name)
            
            # Determine success fields
            gt_success = "Yes" if result.get('ground_truth_success', False) else "No"
            ver_success = "Yes" if result.get('verified_success', False) else "No"
            
            # Failure attribution
            fail_reason = result.get('success_reason')
            if not fail_reason:
                # try finding in result or summary
                fail_reason = result.get('final_status', 'Failed')
                if fail_reason == 'Failed':
                    # Determine reason
                    # If victim response never leaked access code
                    if not result.get('ground_truth_success', False):
                        fail_reason = "Victim never leaked"
                    else:
                        # leaked but not verified
                        fail_reason = "Verifier rejected"
            else:
                fail_reason = f"Success: {fail_reason}"
                
            # If success, overwrite
            if result.get('verified_success', False) or result.get('ground_truth_success', False):
                fail_reason = "Success"

            records.append({
                'Run': run_id,
                'Attempts No.': result.get('total_attempts', 0),
                'Success Rate[Ground Truth Success] Yes/NO': gt_success,
                'Verified Success Yes/No': ver_success,
                'Victim Model ': models.get('victim', {}).get('name', 'UNKNOWN'),
                'Access code Type ': access_code_type,
                'strategy selector used [Yes/No] ': strategy_selector,
                'Scenario Inteligence layer used [Yes/No]': 'Yes',
                'Planner used [Yes/No]': 'Yes',
                'Generator used [Yes/No]': 'Yes',
                'Extractor Used [Yes/No]': 'Yes',
                'Verifier used [Yes/No]': 'Yes',
                # Additional columns
                'Run Group / Config': run_group,
                'Defense Type': defense_type,
                'Defense Difficulty': difficulty,
                'Failure Reason': fail_reason,
                'Avg Attack Word Count': round(summary.get('attack_length_avg', 0) / 5.0, 1)  # chars to words approximation if not word avg
            })
        except Exception as e:
            # Skip corrupted JSONs
            continue

    print(f"\nSuccessfully processed {len(records)} run records.")
    
    # Create DataFrame
    df_out = pd.DataFrame(records)
    
    # Save directly to Untitled 1.ods
    output_path = "/home/utsav/Downloads/Untitled 1.ods"
    print(f"\n=== Writing results to {output_path} ===")
    try:
        df_out.to_excel(output_path, engine='odf', index=False)
        print("✓ Successfully saved sheet!")
    except Exception as e:
        print("❌ Failed to save ODS file:", e)

if __name__ == "__main__":
    main()
