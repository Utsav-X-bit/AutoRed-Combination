import json
import math
import sys
from collections import defaultdict
from pathlib import Path

def calc_entropy(counts):
    total = sum(counts.values())
    if total == 0: return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def analyze_runs(file_list):
    total_attempts = 0
    total_conf = 0.0
    total_len = 0
    total_duplicates = 0
    total_ttr = 0.0
    valid_ttr_attempts = 0
    
    unique_prompts = set()
    strat_counts = defaultdict(int)
    
    first_pick_success = 0
    total_successes = 0
    
    with open(file_list, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
        
    for path in files:
        with open(path, 'r') as f:
            data = json.load(f)
            
        attempts = data.get("attempts", [])
        
        if attempts and attempts[0].get("generator_success") or attempts and attempts[0].get("ground_truth_found") or attempts and attempts[0].get("verification"):
            first_pick_success += 1
            
        for att in attempts:
            total_attempts += 1
            
            judge = att.get("judge", {})
            total_conf += judge.get("confidence", 0)
            
            ttr = att.get("attempt_time_ms")
            if ttr is not None:
                total_ttr += ttr
                valid_ttr_attempts += 1
            
            gen = att.get("generator", {})
            total_len += gen.get("attack_length", 0)
            if gen.get("duplicate_attack", False):
                total_duplicates += 1
                
            strat = gen.get("strategy", "unknown")
            strat_counts[strat] += 1
            
            att_hash = gen.get("attack_hash")
            if att_hash:
                unique_prompts.add(att_hash)
            
    avg_conf = total_conf / total_attempts if total_attempts > 0 else 0
    avg_len = total_len / total_attempts if total_attempts > 0 else 0
    dup_rate = total_duplicates / total_attempts if total_attempts > 0 else 0
    avg_ttr = (total_ttr / valid_ttr_attempts / 1000.0) if valid_ttr_attempts > 0 else 0
    entropy = calc_entropy(strat_counts)
    first_pick_acc = first_pick_success / len(files) if len(files) > 0 else 0
    novel_prompts = len(unique_prompts)
    
    return {
        "avg_conf": avg_conf,
        "avg_len": avg_len,
        "dup_rate": dup_rate,
        "entropy": entropy,
        "total_attempts": total_attempts,
        "total_duplicates": total_duplicates,
        "avg_ttr": avg_ttr,
        "first_pick_acc": first_pick_acc,
        "novel_prompts": novel_prompts,
    }

def format_delta(old_val, new_val, is_percent=False):
    delta = new_val - old_val
    prefix = "+" if delta > 0 else ""
    if is_percent:
        return f"{prefix}{delta * 100:.2f}%"
    return f"{prefix}{delta:.4f}"

def format_val(val, is_percent=False):
    if is_percent:
        return f"{val * 100:.2f}%"
    if isinstance(val, int):
        return str(val)
    return f"{val:.4f}"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <old_runs.txt> <new_runs.txt>")
        sys.exit(1)
        
    old_metrics = analyze_runs(sys.argv[1])
    new_metrics = analyze_runs(sys.argv[2])
    
    print("Comparing detailed metrics...")
    
    # Append to report
    with open("data/comparison_report.md", "a") as f:
        f.write("\n### Deep Metrics (Layers 3-20)\n\n")
        f.write("| Metric | Old (1000r) | New (500r) | Δ |\n")
        f.write("|---|---|---|---|\n")
        
        m = "Average Confidence"
        o = old_metrics['avg_conf']; n = new_metrics['avg_conf']
        f.write(f"| {m} | {format_val(o)} | {format_val(n)} | {format_delta(o, n)} |\n")
        
        m = "First Pick Accuracy"
        o = old_metrics['first_pick_acc']; n = new_metrics['first_pick_acc']
        f.write(f"| {m} | {format_val(o, True)} | {format_val(n, True)} | {format_delta(o, n, True)} |\n")
        
        m = "Strategy Entropy"
        o = old_metrics['entropy']; n = new_metrics['entropy']
        f.write(f"| {m} | {format_val(o)} | {format_val(n)} | {format_delta(o, n)} |\n")
        
        m = "Avg TTR (s)"
        o = old_metrics['avg_ttr']; n = new_metrics['avg_ttr']
        f.write(f"| {m} | {format_val(o)} | {format_val(n)} | {format_delta(o, n)} |\n")
        
        m = "Novel Prompts"
        o = old_metrics['novel_prompts']; n = new_metrics['novel_prompts']
        f.write(f"| {m} | {o} | {n} | {format_delta(o, n)} |\n")
        
        m = "Average Length (chars)"
        o = old_metrics['avg_len']; n = new_metrics['avg_len']
        f.write(f"| {m} | {format_val(o)} | {format_val(n)} | {format_delta(o, n)} |\n")
        
        m = "Repeated Attacks %"
        o = old_metrics['dup_rate']; n = new_metrics['dup_rate']
        f.write(f"| {m} | {format_val(o, True)} | {format_val(n, True)} | {format_delta(o, n, True)} |\n")
    print("Done! Updated data/comparison_report.md")
