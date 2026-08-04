import json
import argparse
from pathlib import Path
import sys
import uuid
import os

# Prevent heavy model loading when importing from experiment modules
os.environ["AUTORED_SERVER_MODE"] = "1"

# Add the project root to sys.path so we can import from experiment
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiment.knowledge_base import KnowledgeBase
from experiment.state_builder import StateBuilder
from dataclasses import dataclass, field

@dataclass
class DefenseScenario:
    opening_defense: str
    closing_defense: str
    access_code: str
    access_code_type: str = "UNKNOWN"
    defense_complexity: str = "UNKNOWN"
    defense_type: str = "UNKNOWN"
    primary_type: str = "UNKNOWN"
    secondary_flags: list = field(default_factory=list)

def convert_benchmark_to_kb(input_path: str, db_path: str = "data/autored_kb.db"):
    input_p = Path(input_path)
    runs = []
    
    if input_p.is_dir():
        print(f"Loading all benchmark traces from directory {input_path}")
        json_files = list(input_p.glob("run_*.json"))
        if not json_files:
            json_files = list(input_p.glob("*.json"))
        for jf in json_files:
            with open(jf, 'r') as f:
                data = json.load(f)
                if "trace" in data or "attempts" in data:
                    runs.append(data)
    else:
        print(f"Loading benchmark results from {input_path}")
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # If the file itself is a run trace
        if "trace" in data or "attempts" in data:
            runs.append(data)
        else:
            runs = data.get("runs", data.get("results", []))

    if not runs:
        print("Could not find any 'trace' or 'attempts' data in the input. If you are using the vLLM batch runner, pass the directory containing the individual run_*.json files instead of merged_summary.json.")
        return

    all_strategies = [
        "trigger_phrase_discovery",
        "instruction_leak",
        "exception_discovery",
        "roleplay",
        "translation",
        "summarization",
        "system_prompt_recovery",
    ]

    kb = KnowledgeBase(db_path=db_path)
    state_builder = StateBuilder(all_strategies=all_strategies)
    
    count = 0
    print(f"Importing {len(runs)} benchmark rounds into Knowledge Base...")
    
    for run in runs:
        if "trace" not in run and "attempts" not in run:
            continue
            
        trace_list = run.get("attempts", run.get("trace", []))
            
        scenario_data = run.get("metadata", {}).get("scenario", {})
        if not scenario_data:
            scenario_data = run.get("scenario", {})
            
        scenario = DefenseScenario(
            access_code=scenario_data.get("access_code", ""),
            opening_defense=scenario_data.get("opening_defense", scenario_data.get("pre_defense", "")),
            closing_defense=scenario_data.get("closing_defense", scenario_data.get("post_defense", "")),
            access_code_type=scenario_data.get("access_code_type", "Unknown")
        )
        scenario.primary_type = scenario_data.get("primary_type", "unknown")
        scenario.difficulty_score = scenario_data.get("difficulty_score", 1.0)
        
        # State tracking history
        history = []
        
        for step in trace_list:
            # 1. Build the State Snapshot exactly like the live loop
            attempt_number = step.get("attempt_number", step.get("iteration", 1))
            
            # Get last confidence and history from previous steps
            last_confidence = 0.0
            if attempt_number > 1:
                prev_step = trace_list[attempt_number - 2]
                last_confidence = prev_step.get("extractor", {}).get("top_k_candidates", [{}])[0].get("score", 0.0) if prev_step.get("extractor", {}).get("top_k_candidates") else 0.0
                
            prev_attack = trace_list[attempt_number - 2].get("generator", {}).get("generated_attack", "") if attempt_number > 1 else ""
            prev_response_obj = trace_list[attempt_number - 2].get("victim", trace_list[attempt_number - 2].get("llm_response", {})) if attempt_number > 1 else {}
            prev_response = prev_response_obj.get("raw_output", "")
            
            state = state_builder.build_state(
                scenario=scenario,
                attempt=attempt_number,
                previous_strategies=[h["strategy"] for h in history],
                local_memory=[prev_attack] if prev_attack else [],
                last_victim_response=prev_response,
                last_extractor_confidence=last_confidence
            )
            
            state_id = state.compute_hash()
            state.state_id = state_id
            
            # 2. Extract step data
            strategy = step.get("generator", {}).get("strategy", "unknown")
            attack = step.get("generator", {}).get("generated_attack", "")
            
            response_obj = step.get("victim", step.get("llm_response", {}))
            response = response_obj.get("raw_output", "")
            
            extractor_data = step.get("extractor", {})
            verification_data = step.get("verification", {})
            
            verified = verification_data.get("success", extractor_data.get("verified", False))
            success_exact = step.get("ground_truth_found", extractor_data.get("success_exact", False))
            success_extractor = step.get("generator_success", extractor_data.get("success_extractor", False))
            
            # Build trajectory record
            exp_data = run.get("experiment", {})
            scenario_id = run.get("metadata", {}).get("scenario_id", exp_data.get("scenario_id", str(uuid.uuid4())))
            
            trajectory = {
                "scenario_id": scenario_id,
                "attempt": attempt_number,
                "strategy": strategy,
                "attack_string": attack,
                "response_string": response,
                "reward": 10.0 if (verified or success_exact) else 0.0,
                "ground_truth_leaked": success_exact,
                "generator_success": success_exact,
                "extractor_success": success_extractor,
                "verifier_success": verified,
                "state_id": state_id,
                "chosen_strategy": strategy,
                "alternative_strategies": json.dumps([]),
                "decision_reason": "Heuristic selection from legacy batched pipeline",
                "decision_confidence": 1.0,
                "state_snapshot": {
                    "state_id": state.state_id,
                    "attempt": state.attempt,
                    "state_json": state.to_dict(),
                    "hash": state.compute_hash()
                }
            }
            
            # Save Trajectory to KB
            kb.log_trajectory(trajectory)
            
            # Append to history for next state calculation
            history.append({
                "strategy": strategy,
                "success": verified or success_exact
            })
            count += 1

    print(f"Successfully imported {count} transitions into {db_path}!")
    print("You can now run 'python scripts/dataset_tools/mine_attack_transitions.py' to generate your Planner training data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Benchmark JSONs to KB Trajectories with StateSnapshots")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to merged_summary.json OR the directory containing individual run_*.json traces")
    parser.add_argument("--db", type=str, default="data/autored_kb.db", help="Path to output sqlite DB")
    
    args = parser.parse_args()
    convert_benchmark_to_kb(args.input, args.db)
