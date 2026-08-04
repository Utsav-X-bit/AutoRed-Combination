import os
import json
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Build verifier failure dataset")
    parser.add_argument("--results-dir", type=str, default="results/2026-07-12", help="Path to benchmark results")
    parser.add_argument("--output", type=str, default="data/verifier_failures_v1.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.results_dir, "*.json"))
    failures = []
    
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        scenario = data.get("scenario", {})
        scenario_id = data.get("scenario_id", scenario.get("scenario_id", "unknown"))
        defense_type = scenario.get("defense_type", "unknown")
        access_code_type = scenario.get("access_code_type", "unknown")
        ground_truth = scenario.get("access_code", "")
        pre_defense = scenario.get("pre_defense", "")
        post_defense = scenario.get("post_defense", "")
        
        for attempt in data.get("attempts", []):
            # Only care about instances where the victim actually leaked the ground truth
            # but the verification still failed!
            if not attempt.get("ground_truth_found"):
                continue

            verif = attempt.get("verification", {})
            extr = attempt.get("extractor", {})
            
            traces = verif.get("traces", [])
            if not traces:
                traces = extr.get("verification_traces", [])
            
            candidate_sent = verif.get("candidate_sent")
            if not candidate_sent and traces:
                candidate_sent = traces[0].get("candidate")
            
            if traces and not verif.get("success") and not any(t.get("success") for t in traces):
                primary_trace = traces[0]
                cand = primary_trace.get("candidate", "")
                victim_resp = primary_trace.get("victim_response", "")
                rank = primary_trace.get("rank", -1)
                
                if not primary_trace.get("accepted_by_victim"):
                    failure_reason = "rejected_by_victim"
                elif not primary_trace.get("complete_match"):
                    failure_reason = "format_mismatch"
                else:
                    failure_reason = "unknown"
                    
                verification_prompt = f"{pre_defense}\n{cand}\n{post_defense}"
                
                failures.append({
                    "scenario_id": scenario_id,
                    "defense_type": defense_type,
                    "access_code_type": access_code_type,
                    "ground_truth": ground_truth,
                    "extracted_candidate": cand,
                    "rank": rank,
                    "verification_prompt": verification_prompt,
                    "victim_response": victim_resp,
                    "failure_reason": failure_reason,
                    "attempt_number": attempt.get("attempt_number", -1)
                })
                
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for item in failures:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(failures)} verifier failures in {args.output}")

if __name__ == "__main__":
    main()
