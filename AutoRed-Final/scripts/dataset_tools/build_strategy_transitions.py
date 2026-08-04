#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

def build_transitions():
    print("Building strategy transition graph from attempts...")
    
    # We want to trace attempts sequentially per scenario.
    # We will load both successes and failures, group them by run_id + scenario_id.
    
    # Dictionary structure: scenario_history[run_id][scenario_id] = [attempt_dict, ...]
    scenario_history = defaultdict(lambda: defaultdict(list))
    
    files_to_process = [
        DATA_DIR / "autored_failures_v1.jsonl",
        DATA_DIR / "autored_successes_v1.jsonl"
    ]
    
    total_records = 0
    for file_path in files_to_process:
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            continue
            
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                run_id = data.get("run_id", "unknown")
                scen_id = data.get("scenario_id", "unknown")
                scenario_history[run_id][scen_id].append(data)
                total_records += 1
                
    print(f"Loaded {total_records} records.")
    
    # transition_counts[prev_strategy]["success" / "failure"][next_strategy] = count
    transitions = defaultdict(lambda: {"success": defaultdict(int), "failure": defaultdict(int)})
    
    valid_transitions = 0
    
    for run_id, scenarios in scenario_history.items():
        for scen_id, attempts in scenarios.items():
            # Sort attempts by attempt_number to recreate the sequence
            sorted_attempts = sorted(attempts, key=lambda x: x.get("attempt_number", 0))
            
            for i in range(len(sorted_attempts) - 1):
                prev_attempt = sorted_attempts[i]
                next_attempt = sorted_attempts[i+1]
                
                prev_strat = prev_attempt.get("strategy", "unknown")
                next_strat = next_attempt.get("strategy", "unknown")
                
                # Check if prev attempt was a success or failure
                # We usually transition because it failed. But let's track both just in case.
                # A verified success means we stopped, but if it wasn't verified maybe we continued.
                # We'll use ground_truth_leaked or extractor_success or success
                is_success = prev_attempt.get("verification_success", False) or prev_attempt.get("success", False)
                state = "success" if is_success else "failure"
                
                transitions[prev_strat][state][next_strat] += 1
                valid_transitions += 1
                
    print(f"Processed {valid_transitions} valid transitions.")
    
    # Convert counts to probabilities
    transition_probs = defaultdict(lambda: {"success": {}, "failure": {}})
    
    for prev_strat, states in transitions.items():
        for state, next_strats in states.items():
            total = sum(next_strats.values())
            if total > 0:
                for next_strat, count in next_strats.items():
                    transition_probs[prev_strat][state][next_strat] = count / total
                    
    output_path = DATA_DIR / "strategy_transitions.json"
    with open(output_path, "w") as f:
        json.dump(transition_probs, f, indent=4)
        
    print(f"Saved transition probabilities to {output_path}")

if __name__ == "__main__":
    build_transitions()
