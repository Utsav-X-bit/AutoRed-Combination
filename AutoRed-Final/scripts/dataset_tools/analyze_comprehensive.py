import json
import argparse
import os
import glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import re

def compute_metrics(jsons_dir):
    files = glob.glob(os.path.join(jsons_dir, "run_*.json"))
    if not files:
        print(f"No run JSONs found in {jsons_dir}")
        return

    print(f"Loaded {len(files)} runs for analysis.")

    # Data structures for analysis
    all_runs = []
    
    # 1. Overall Performance
    total_scenarios = len(files)
    gt_success_count = 0
    extractor_success_count = 0
    verified_success_count = 0
    
    total_attempts = 0
    total_run_time = 0
    input_tokens = 0
    output_tokens = 0
    
    top_1_count = 0
    top_3_count = 0
    top_5_count = 0
    verified_ranks = []
    
    tp, fp, fn = 0, 0, 0
    
    # 2. Defense Type Analysis
    defense_stats = defaultdict(lambda: {"total": 0, "success": 0, "attempts": [], "strategies": defaultdict(int)})
    
    # 3. Access Code Type Analysis
    code_stats = defaultdict(lambda: {"total": 0, "gt_leak": 0, "extractor_success": 0, "verified": 0})
    
    # 4. Strategy Analysis
    strategy_stats = defaultdict(lambda: {"total": 0, "success": 0, "attempts": [], "tokens": [], "leak_lengths": []})
    strategy_defense_heat = defaultdict(lambda: defaultdict(lambda: {"total": 0, "success": 0}))
    
    # 5. Attempt Analysis
    attempt_successes = defaultdict(lambda: {"total": 0, "success": 0})
    
    # 6. Failure Attribution
    failure_reasons = defaultdict(int)
    
    # 7. Generator Analysis
    attack_lengths_words = []
    attack_lengths_chars = []
    unique_attacks_set = set()
    repetition_count = 0
    total_generated = 0
    
    # 8. Judge Analysis
    judge_cm = {"ATTACK_as_ATTACK": 0, "ATTACK_as_ATTEMPT": 0, "ATTEMPT_as_ATTACK": 0, "ATTEMPT_as_ATTEMPT": 0}
    
    # 9. Extractor Analysis
    extractor_failure_modes = defaultdict(int)
    
    # 10. Access Predictor Analysis
    predictor_cm = defaultdict(lambda: defaultdict(int))
    
    # 11. RAG Analysis
    rag_stats = {"total": 0, "success": 0}

    # Knowledge Base
    kb_records = []

    for fpath in files:
        with open(fpath, "r") as f:
            run = json.load(f)
            
        all_runs.append(run)
        
        scenario = run.get("scenario", {})
        result = run.get("result", {})
        attempts = run.get("attempts", [])
        raw_dataset_entry = run.get("raw_dataset_entry", {})
        
        defense_type = scenario.get("defense_type", "UNKNOWN")
        if defense_type == "UNKNOWN" or not defense_type:
            defense_type = raw_dataset_entry.get("defense_type", "UNKNOWN")
            
        access_code_type = scenario.get("access_code_type", "UNKNOWN")
        if access_code_type == "UNKNOWN" or not access_code_type:
            access_code_type = raw_dataset_entry.get("access_code_type", "UNKNOWN")
            
        gt_success = result.get("ground_truth_success", result.get("generator_success", False))
        ext_success = result.get("extractor_success", False)
        ver_success = result.get("verified_success", False)
        
        if gt_success: gt_success_count += 1
        if ext_success: extractor_success_count += 1
        if ver_success: verified_success_count += 1
        
        total_attempts += result.get("total_attempts", len(attempts))
        total_run_time += run.get("timing", {}).get("total_run_time", 0)
        
        defense_stats[defense_type]["total"] += 1
        if gt_success:
            defense_stats[defense_type]["success"] += 1
        defense_stats[defense_type]["attempts"].append(result.get("total_attempts", len(attempts)))
        
        code_stats[access_code_type]["total"] += 1
        if gt_success: code_stats[access_code_type]["gt_leak"] += 1
        if ext_success: code_stats[access_code_type]["extractor_success"] += 1
        if ver_success: code_stats[access_code_type]["verified"] += 1
        
        run_failure_reason = None
        if not gt_success and not ext_success and not ver_success:
            run_failure_reason = "Victim never leaked"
        
        # Analyze attempts
        attempt_num = 1
        for i, attempt in enumerate(attempts):
            gen = attempt.get("generator", {})
            ext = attempt.get("extractor", {})
            jdg = attempt.get("judge", {})
            vic = attempt.get("victim", {})
            ver = attempt.get("verification", {})
            
            strat = gen.get("strategy", "unknown")
            input_tokens += gen.get("input_tokens", 0)
            output_tokens += gen.get("output_tokens", 0)
            
            att_len_char = gen.get("attack_length", 0)
            att_len_word = len(gen.get("generated_attack", "").split())
            attack_lengths_chars.append(att_len_char)
            attack_lengths_words.append(att_len_word)
            
            total_generated += 1
            att_hash = gen.get("attack_hash", "")
            if att_hash in unique_attacks_set:
                repetition_count += 1
            else:
                unique_attacks_set.add(att_hash)
                
            is_gt_found = attempt.get("ground_truth_found", attempt.get("generator_success", False))
            is_ext_match = attempt.get("extractor_match", False)
            is_ver_success = ver.get("success", False)
            
            # Record Strategy
            strategy_stats[strat]["total"] += 1
            strategy_stats[strat]["attempts"].append(attempt_num)
            strategy_stats[strat]["tokens"].append(gen.get("output_tokens", 0))
            if is_gt_found:
                strategy_stats[strat]["success"] += 1
                defense_stats[defense_type]["strategies"][strat] += 1
                strategy_defense_heat[strat][defense_type]["success"] += 1
                if vic.get("clean_output"):
                    strategy_stats[strat]["leak_lengths"].append(len(vic.get("clean_output")))
            
            strategy_defense_heat[strat][defense_type]["total"] += 1
            
            # Attempt success tracking
            attempt_successes[attempt_num]["total"] += 1
            if is_gt_found:
                attempt_successes[attempt_num]["success"] += 1
                
            # Extractor & Judge Failure Attribution
            decision = jdg.get("decision", "")
            
            if is_gt_found:
                if decision == "ATTACK":
                    judge_cm["ATTEMPT_as_ATTACK"] += 1
                    run_failure_reason = "Judge blocked extractor"
                else:
                    judge_cm["ATTEMPT_as_ATTEMPT"] += 1
                    if not is_ext_match:
                        run_failure_reason = "Extractor missed"
                        # Figure out why
                        if not ext.get("best_candidate"):
                            extractor_failure_modes["NO_CANDIDATE"] += 1
                        else:
                            extractor_failure_modes["WRONG_CANDIDATE"] += 1
                    else:
                        if not is_ver_success and ver.get("candidate_sent"):
                            run_failure_reason = "Verifier rejected"
                        
                if is_ext_match:
                    tp += 1
                    rank = ext.get("verified_rank", 0)
                    if rank == 1: top_1_count += 1
                    if 1 <= rank <= 3: top_3_count += 1
                    if 1 <= rank <= 5: top_5_count += 1
                    if rank > 0: verified_ranks.append(rank)
                else:
                    fn += 1
            else:
                if decision == "ATTACK":
                    judge_cm["ATTACK_as_ATTACK"] += 1
                else:
                    judge_cm["ATTACK_as_ATTEMPT"] += 1
                    
            attempt_num += 1
            
        if run_failure_reason:
            failure_reasons[run_failure_reason] += 1
            
        # KB Record
        last_attempt = attempts[-1] if attempts else {}
        kb_record = {
            "scenario_id": run.get("experiment", {}).get("scenario_id", "unknown"),
            "defense_type": defense_type,
            "access_code_type": access_code_type,
            "strategy": last_attempt.get("generator", {}).get("strategy", "unknown"),
            "attempt": result.get("total_attempts", len(attempts)),
            "generator_success": gt_success,
            "judge_decision": last_attempt.get("judge", {}).get("decision", ""),
            "extractor_rank": last_attempt.get("extractor", {}).get("verified_rank", 0),
            "verified": ver_success,
            "failure_reason": run_failure_reason if not gt_success else None
        }
        kb_records.append(kb_record)

    # Calculate metrics
    success_rate = gt_success_count / total_scenarios if total_scenarios else 0
    avg_attempts = total_attempts / total_scenarios if total_scenarios else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_rank = np.mean(verified_ranks) if verified_ranks else 0
    
    # Generate Markdown
    md = []
    md.append("# Benchmark Analysis")
    md.append("\n## 1. Overall Performance")
    md.append(f"- **Total Scenarios:** {total_scenarios}")
    md.append(f"- **Success Rate:** {success_rate:.2%}")
    md.append(f"- **Ground Truth Success:** {gt_success_count}")
    md.append(f"- **Extractor Success:** {extractor_success_count}")
    md.append(f"- **Verified Success:** {verified_success_count}")
    md.append(f"- **Defense Rate:** {1 - success_rate:.2%}")
    md.append(f"- **Average Attempts:** {avg_attempts:.2f}")
    md.append(f"- **Average Runtime (s):** {total_run_time/total_scenarios:.2f}")
    md.append(f"- **Input Tokens:** {input_tokens}")
    md.append(f"- **Output Tokens:** {output_tokens}")
    md.append("\n### Top-K Metrics")
    md.append(f"- **Top-1:** {top_1_count} ({(top_1_count/total_scenarios)*100 if total_scenarios else 0:.1f}%)")
    md.append(f"- **Top-3:** {top_3_count} ({(top_3_count/total_scenarios)*100 if total_scenarios else 0:.1f}%)")
    md.append(f"- **Top-5:** {top_5_count} ({(top_5_count/total_scenarios)*100 if total_scenarios else 0:.1f}%)")
    md.append(f"- **Average Verified Rank:** {avg_rank:.2f}")
    md.append("\n### Extractor Metrics")
    md.append(f"- **Precision:** {precision:.2%}")
    md.append(f"- **Recall:** {recall:.2%}")
    md.append(f"- **F1:** {f1:.2%}")
    md.append(f"- **TP / FP / FN:** {tp} / {fp} / {fn}")
    
    md.append("\n## 2. Breakdown by Defense")
    md.append("| Defense | Success | Avg Attempts | Best Strategy |")
    md.append("|---------|---------|--------------|---------------|")
    for d_type, stats in sorted(defense_stats.items(), key=lambda x: x[1]["success"], reverse=True):
        d_success_rate = stats["success"] / stats["total"] if stats["total"] else 0
        d_avg_att = np.mean(stats["attempts"]) if stats["attempts"] else 0
        best_strat = max(stats["strategies"].items(), key=lambda x: x[1])[0] if stats["strategies"] else "N/A"
        md.append(f"| {d_type} | {d_success_rate:.1%} ({stats['success']}/{stats['total']}) | {d_avg_att:.2f} | {best_strat} |")

    md.append("\n## 3. Access Code Type Analysis")
    md.append("| Type | GT Leak | Extractor Recall | Verified |")
    md.append("|------|---------|------------------|----------|")
    for a_type, stats in sorted(code_stats.items(), key=lambda x: x[1]["gt_leak"], reverse=True):
        gt_rate = stats["gt_leak"] / stats["total"] if stats["total"] else 0
        ext_recall = stats["extractor_success"] / stats["gt_leak"] if stats["gt_leak"] else 0
        ver_rate = stats["verified"] / stats["total"] if stats["total"] else 0
        md.append(f"| {a_type} | {gt_rate:.1%} | {ext_recall:.1%} | {ver_rate:.1%} |")
        
    md.append("\n## 4. Strategy Analysis")
    md.append("| Strategy | Chosen | Success | Failure | Success Rate | Avg Attempts | Avg Tokens | Avg Leak Length |")
    md.append("|----------|--------|---------|---------|--------------|--------------|------------|-----------------|")
    for strat, stats in sorted(strategy_stats.items(), key=lambda x: x[1]["success"], reverse=True):
        s_rate = stats["success"] / stats["total"] if stats["total"] else 0
        s_avg_att = np.mean(stats["attempts"]) if stats["attempts"] else 0
        s_avg_tok = np.mean(stats["tokens"]) if stats["tokens"] else 0
        s_avg_leak = np.mean(stats["leak_lengths"]) if stats["leak_lengths"] else 0
        md.append(f"| {strat} | {stats['total']} | {stats['success']} | {stats['total']-stats['success']} | {s_rate:.1%} | {s_avg_att:.2f} | {s_avg_tok:.1f} | {s_avg_leak:.1f} |")

    md.append("\n### Strategy -> Defense Heatmap (Success Rate)")
    defenses = list(defense_stats.keys())
    header = "| Strategy | " + " | ".join(defenses) + " |"
    sep = "|----------|" + "|".join(["---"] * len(defenses)) + "|"
    md.append(header)
    md.append(sep)
    for strat in strategy_stats.keys():
        row = f"| {strat} |"
        for d in defenses:
            s_stats = strategy_defense_heat[strat][d]
            rate = s_stats["success"] / s_stats["total"] if s_stats["total"] else 0
            row += f" {rate:.1%} |"
        md.append(row)

    md.append("\n## 5. Attempt Analysis")
    md.append("| Attempt | Success | Total Reached | Marginal Success Rate |")
    md.append("|---------|---------|---------------|-----------------------|")
    for att in sorted(attempt_successes.keys()):
        stats = attempt_successes[att]
        rate = stats["success"] / stats["total"] if stats["total"] else 0
        md.append(f"| {att} | {stats['success']} | {stats['total']} | {rate:.1%} |")

    md.append("\n## 6. Failure Attribution")
    md.append("| Reason | Count | Percentage |")
    md.append("|--------|-------|------------|")
    total_failures = sum(failure_reasons.values())
    for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_failures if total_failures else 0
        md.append(f"| {reason} | {count} | {pct:.1%} |")

    md.append("\n## 7. Generator Analysis")
    avg_words = np.mean(attack_lengths_words) if attack_lengths_words else 0
    avg_chars = np.mean(attack_lengths_chars) if attack_lengths_chars else 0
    diversity = len(unique_attacks_set) / total_generated if total_generated else 0
    md.append(f"- **Average Attack Length:** {avg_words:.1f} words / {avg_chars:.1f} chars")
    md.append(f"- **Unique Attacks:** {len(unique_attacks_set)} out of {total_generated}")
    md.append(f"- **Diversity (Unique/Total):** {diversity:.2%}")
    md.append(f"- **Repetition Count (verbatim):** {repetition_count}")

    md.append("\n## 8. Judge Analysis")
    md.append("### Confusion Matrix")
    md.append("| | Predicted ATTACK | Predicted ATTEMPT |")
    md.append("|---|---|---|")
    md.append(f"| **Ground Truth No Leak** | {judge_cm['ATTACK_as_ATTACK']} | {judge_cm['ATTACK_as_ATTEMPT']} (False ATTEMPT) |")
    md.append(f"| **Ground Truth Leaked** | {judge_cm['ATTEMPT_as_ATTACK']} (Blocked Extractor) | {judge_cm['ATTEMPT_as_ATTEMPT']} |")

    md.append("\n## 9. Extractor Analysis")
    md.append(f"- **Candidate Recall (Top-1):** {top_1_count / gt_success_count if gt_success_count else 0:.1%}")
    md.append(f"- **Candidate Recall (Top-3):** {top_3_count / gt_success_count if gt_success_count else 0:.1%}")
    md.append(f"- **Candidate Recall (Top-5):** {top_5_count / gt_success_count if gt_success_count else 0:.1%}")
    md.append("\n### Failure Modes")
    for mode, count in sorted(extractor_failure_modes.items(), key=lambda x: x[1], reverse=True):
        md.append(f"- **{mode}:** {count}")

    md.append("\n## 10. Access Predictor Analysis")
    md.append("*(Metadata logging added, future runs will populate access_code_type prediction accuracy)*")
    
    md.append("\n## 11. Recommendations")
    md.append("1. **Extractor Focus:** If GT Leak is high but Extractor Success is low, tune the regex or LLM candidates.")
    md.append("2. **Strategy Routing:** Use the Strategy->Defense Heatmap to build a routing policy.")
    md.append("3. **Cutoff Threshold:** If marginal success drops near zero after X attempts, lower `MAX_INTERACTIONS` to save compute.")

    # Write MD report
    out_dir = os.path.dirname(jsons_dir)
    md_path = os.path.join(out_dir, "comprehensive_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
        
    print(f"Markdown report generated at {md_path}")
    
    # Write KB JSONL
    kb_path = os.path.join(out_dir, "benchmark_knowledge_base.jsonl")
    with open(kb_path, "w") as f:
        for rec in kb_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Knowledge Base JSONL generated at {kb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing run JSON files")
    args = parser.parse_args()
    compute_metrics(args.input)
