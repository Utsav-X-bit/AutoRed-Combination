import os
import json
import argparse
from tqdm import tqdm

def build_sft_dataset(results_dir: str, successes_file: str, output_file: str, max_samples: int = None):
    print(f"Loading successes from {successes_file}...")
    success_keys = set()
    with open(successes_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            # We identify the exact successful attempt by run_id and attempt_number
            key = f"{data['run_id']}_{data['attempt_number']}"
            success_keys.add(key)
            
    print(f"Loaded {len(success_keys)} successful attempts to extract.")
    
    sft_data = []
    
    # Iterate over all run files in results_dir
    run_files = [f for f in os.listdir(results_dir) if f.startswith("run_") and f.endswith(".json")]
    
    print(f"Scanning {len(run_files)} run JSON files for internal prompts...")
    for rf in tqdm(run_files):
        run_id = rf.replace(".json", "")
        file_path = os.path.join(results_dir, rf)
        
        try:
            with open(file_path, 'r') as f:
                run_data = json.load(f)
                
            attempts = run_data.get("attempts", [])
            for att in attempts:
                attempt_num = att.get("attempt_number")
                key = f"{run_id}_{attempt_num}"
                
                if key in success_keys:
                    generator_data = att.get("generator", {})
                    internal_prompt = generator_data.get("internal_prompt", "")
                    generated_attack = generator_data.get("generated_attack", "")
                    
                    if internal_prompt and generated_attack:
                        # Llama-3 Chat Format
                        sft_row = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": internal_prompt
                                },
                                {
                                    "role": "assistant",
                                    "content": generated_attack
                                }
                            ]
                        }
                        sft_data.append(sft_row)
                        
                        if max_samples and len(sft_data) >= max_samples:
                            break
                            
            if max_samples and len(sft_data) >= max_samples:
                break
                
        except Exception as e:
            print(f"Error reading {rf}: {e}")
            continue

    print(f"Successfully extracted {len(sft_data)} SFT examples.")
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for row in sft_data:
            f.write(json.dumps(row) + '\n')
            
    print(f"[SAVED] SFT dataset saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT dataset for Generator Retraining")
    parser.add_argument("--results-dir", type=str, default="results/", help="Directory containing run_*.json files")
    parser.add_argument("--successes-file", type=str, default="data/autored_successes_v1.jsonl", help="Path to successes JSONL")
    parser.add_argument("--output", type=str, default="data/generator_sft_dataset.jsonl", help="Output SFT JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to extract")
    
    args = parser.parse_args()
    
    build_sft_dataset(args.results_dir, args.successes_file, args.output, args.max_samples)
