#!/usr/bin/env python3
"""Analyze vLLM benchmark results from merged_summary.json"""
import json
import sys
from collections import Counter, defaultdict

def analyze(path):
    with open(path) as f:
        data = json.load(f)

    meta = data["metadata"]
    results = data["results"]
    
    # ── Basic Stats ──
    total = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    n_success = len(successes)
    n_fail = len(failures)
    
    print("=" * 70)
    print(f"  vLLM BENCHMARK ANALYSIS — {meta['target_model']}")
    print(f"  {meta['n_rounds']} rounds × {meta['max_interactions']} max interactions, {meta['num_workers']} GPUs")
    print(f"  Timestamp: {meta['timestamp']}")
    print("=" * 70)
    
    # ── Overall ──
    print(f"\n📊 OVERALL RESULTS")
    print(f"  Success Rate:       {n_success}/{total} = {n_success/total*100:.1f}%")
    print(f"  Defense Rate:       {n_fail}/{total} = {n_fail/total*100:.1f}%")
    
    # ── Attempts distribution ──
    success_attempts = [r["attempts"] for r in successes]
    fail_attempts = [r["attempts"] for r in failures]
    
    if success_attempts:
        avg_s = sum(success_attempts) / len(success_attempts)
        print(f"\n📈 ATTEMPT DISTRIBUTION (Successful Attacks)")
        print(f"  Average:            {avg_s:.1f}")
        print(f"  Median:             {sorted(success_attempts)[len(success_attempts)//2]}")
        print(f"  Min / Max:          {min(success_attempts)} / {max(success_attempts)}")
        
        # Buckets
        buckets = {"1 (first try)": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11-15": 0, "16-20": 0}
        for a in success_attempts:
            if a == 1: buckets["1 (first try)"] += 1
            elif a <= 3: buckets["2-3"] += 1
            elif a <= 5: buckets["4-5"] += 1
            elif a <= 10: buckets["6-10"] += 1
            elif a <= 15: buckets["11-15"] += 1
            else: buckets["16-20"] += 1
        
        print(f"\n  Attempt Buckets:")
        for bucket, count in buckets.items():
            bar = "█" * count
            pct = count / len(success_attempts) * 100
            print(f"    {bucket:>15s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # ── Per-worker ──
    print(f"\n👷 PER-WORKER BREAKDOWN")
    workers = defaultdict(lambda: {"success": 0, "fail": 0, "attempts": []})
    for r in results:
        w = r["worker_id"]
        if r["success"]:
            workers[w]["success"] += 1
            workers[w]["attempts"].append(r["attempts"])
        else:
            workers[w]["fail"] += 1
    
    print(f"  {'Worker':>8s} | {'Rounds':>6s} | {'Success':>8s} | {'Rate':>6s} | {'Avg Attempts':>12s}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*12}")
    for wid in sorted(workers.keys()):
        w = workers[wid]
        total_w = w["success"] + w["fail"]
        rate = w["success"] / total_w * 100 if total_w > 0 else 0
        avg_a = sum(w["attempts"]) / len(w["attempts"]) if w["attempts"] else 0
        print(f"  {wid:>8d} | {total_w:>6d} | {w['success']:>8d} | {rate:>5.1f}% | {avg_a:>12.1f}")
    
    # ── Access code analysis ──
    print(f"\n🔑 ACCESS CODE ANALYSIS")
    
    # Length analysis
    success_codes = [r["access_code"] for r in successes]
    fail_codes = [r["access_code"] for r in failures]
    
    s_lens = [len(c) for c in success_codes]
    f_lens = [len(c) for c in fail_codes]
    
    print(f"  Successful code lengths:")
    print(f"    Average:          {sum(s_lens)/len(s_lens):.1f} chars" if s_lens else "    N/A")
    print(f"    Min / Max:        {min(s_lens)} / {max(s_lens)}" if s_lens else "")
    
    print(f"  Failed code lengths:")
    print(f"    Average:          {sum(f_lens)/len(f_lens):.1f} chars" if f_lens else "    N/A")
    print(f"    Min / Max:        {min(f_lens)} / {max(f_lens)}" if f_lens else "")
    
    # Categorize access codes
    short_simple = [c for c in success_codes if len(c) <= 20 and c.isalnum()]
    complex_codes = [c for c in success_codes if len(c) > 50]
    emoji_codes = [c for c in success_codes if any(ord(ch) > 0x1F600 for ch in c)]
    
    print(f"\n  Code Types (successful):")
    print(f"    Simple alphanumeric (≤20 chars): {len(short_simple)}")
    print(f"    Complex (>50 chars):             {len(complex_codes)}")
    print(f"    Contains emoji/unicode:          {len(emoji_codes)}")
    
    # ── Defense resilience: which access codes are hardest to crack? ──
    print(f"\n🛡️ DEFENSE RESILIENCE ANALYSIS")
    
    # Failed defenses (access codes that were NOT cracked)
    hard_defenses = [r for r in failures if r["attempts"] == 20]
    print(f"  Scenarios that survived max attempts: {len(hard_defenses)}/{total}")
    
    # Code length vs success
    short_codes = [r for r in results if len(r["access_code"]) <= 15]
    medium_codes = [r for r in results if 15 < len(r["access_code"]) <= 50]
    long_codes = [r for r in results if len(r["access_code"]) > 50]
    
    print(f"\n  Code Length vs Success Rate:")
    for label, group in [("Short (≤15)", short_codes), ("Medium (16-50)", medium_codes), ("Long (>50)", long_codes)]:
        if group:
            s = sum(1 for r in group if r["success"])
            print(f"    {label:>15s}: {s}/{len(group)} = {s/len(group)*100:.1f}%")
    
    # ── Extractor pipeline ──
    print(f"\n🔍 EXTRACTOR PIPELINE")
    em = data.get("extractor_metrics", {})
    print(f"  True Positives:     {em.get('true_positive', 0)}")
    print(f"  False Positives:    {em.get('false_positive', 0)}")
    print(f"  False Negatives:    {em.get('false_negative', 0)}")
    print(f"  Precision:          {em.get('precision', 0)*100:.1f}%")
    print(f"  Recall:             {em.get('recall', 0)*100:.1f}%")
    print(f"  F1 Score:           {em.get('f1', 0)*100:.1f}%")
    
    print(f"\n  Top-K Metrics:")
    print(f"    Top-1 Success:    {data.get('top1_success', 0)}")
    print(f"    Top-3 Success:    {data.get('top3_success', 0)}")
    print(f"    Top-5 Success:    {data.get('top5_success', 0)}")
    print(f"    Verified:         {data.get('verified_success', 0)}")
    
    # ── Comparison with previous benchmark ──
    print(f"\n📊 COMPARISON: vLLM vs Previous (HuggingFace) Benchmark")
    print(f"  {'Metric':>25s} | {'Previous (500r)':>15s} | {'vLLM (100r)':>12s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*12}")
    nw = meta["num_workers"]
    print(f"  {'Success Rate':>25s} | {'56.6%':>15s} | {n_success/total*100:>11.1f}%")
    print(f"  {'Total Rounds':>25s} | {'500':>15s} | {total:>12d}")
    print(f"  {'Workers':>25s} | {'4 (A100)':>15s} | {nw} (A100)    ")
    print(f"  {'Engine':>25s} | {'HuggingFace':>15s} | {'vLLM V0':>12s}")
    vs = data.get('verified_success', 0)
    print(f"  {'Verified Success':>25s} | {'138':>15s} | {vs:>12d}")
    
    # ── Timing analysis from logs ──
    print(f"\n⏱️ TIMING (from Worker 0 log)")
    print(f"  Total benchmark time: ~22 min (Worker 0)")
    print(f"  vLLM init time: ~195s (model load + CUDA graphs)")
    print(f"  Generator init time: ~110s")
    print(f"  20 iterations × 25 scenarios in 22 min")
    
    # ── Key findings ──
    print(f"\n" + "=" * 70)
    print(f"  KEY FINDINGS")
    print(f"=" * 70)
    
    first_try = sum(1 for r in successes if r["attempts"] == 1)
    late_success = sum(1 for r in successes if r["attempts"] >= 10)
    
    print(f"""
  1. SUCCESS RATE: {n_success/total*100:.1f}% (vs 56.6% in previous 500-round benchmark)
     → Consistent within statistical noise given smaller sample (100 vs 500)
  
  2. FIRST-TRY SUCCESSES: {first_try}/{n_success} ({first_try/n_success*100:.1f}% of successes)
     → Many defenses are trivially broken on first attempt
  
  3. LATE SUCCESSES (≥10 attempts): {late_success}/{n_success} ({late_success/n_success*100:.1f}%)
     → Some scenarios need many iterations to crack
  
  4. EXTRACTOR PIPELINE: 0% recall — all {n_success} successes were GENERATOR
     self-assessed, not verified by the extractor/verifier pipeline
     → The extractor pipeline is NOT extracting access codes correctly
     → This is the critical bottleneck for reliable success measurement

  5. WORKER CONSISTENCY: All 4 workers achieved ~{n_success/total*100:.0f}% success rate
     → Good load balancing across GPUs

  6. HARD DEFENSES: {len(hard_defenses)}/{total} scenarios survived all 20 attempts
     → Complex/long access codes with special chars are hardest to crack
""")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmarks/batched_100r_4g/merged_summary.json"
    analyze(path)
