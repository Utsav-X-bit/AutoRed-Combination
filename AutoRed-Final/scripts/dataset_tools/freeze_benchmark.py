import json
import hashlib
import random
from collections import defaultdict
import os

def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(project_root, "data", "benchmark_v2.jsonl")
    holdout_file = os.path.join(project_root, "data", "benchmark_holdout_v2.jsonl")
    dev_file = os.path.join(project_root, "data", "benchmark_dev_v2.jsonl")
    manifest_file = os.path.join(project_root, "data", "benchmark_manifest_v2.json")
    
    split_seed = 42
    random.seed(split_seed)
    
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    groups = defaultdict(list)
    for entry in data:
        key = (entry.get('access_code_type', 'UNKNOWN'), entry.get('difficulty', 'UNKNOWN'))
        groups[key].append(entry)
        
    holdout_data = []
    dev_data = []
    
    target_holdout = 200
    total_items = len(data)
    
    # Stratified proportional split
    for key, items in sorted(groups.items()):
        random.shuffle(items)
        n_holdout = round(len(items) * target_holdout / total_items)
        holdout_data.extend(items[:n_holdout])
        dev_data.extend(items[n_holdout:])
        
    # Adjust for rounding errors
    diff = len(holdout_data) - target_holdout
    if diff > 0:
        move = holdout_data[:diff]
        holdout_data = holdout_data[diff:]
        dev_data.extend(move)
    elif diff < 0:
        move = dev_data[:-diff]
        dev_data = dev_data[-diff:]
        holdout_data.extend(move)
        
    random.shuffle(holdout_data)
    random.shuffle(dev_data)
    
    with open(holdout_file, 'w') as f:
        for entry in holdout_data:
            f.write(json.dumps(entry) + "\n")
            
    with open(dev_file, 'w') as f:
        for entry in dev_data:
            f.write(json.dumps(entry) + "\n")
            
    # Hash
    hasher = hashlib.sha256()
    with open(holdout_file, 'rb') as f:
        hasher.update(f.read())
    holdout_hash = hasher.hexdigest()
    
    manifest = {
        "version": "v2",
        "split_seed": split_seed,
        "holdout_hash": holdout_hash,
        "holdout_size": len(holdout_data),
        "dev_size": len(dev_data),
        "source_file": "data/benchmark_v2.jsonl",
        "created_date": "2026-06-22"
    }
    
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Created holdout with {len(holdout_data)} scenarios and dev with {len(dev_data)} scenarios.")
    print(f"Holdout Hash: {holdout_hash}")

if __name__ == '__main__':
    main()
