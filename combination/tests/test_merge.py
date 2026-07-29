import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'AutoRed-Final', 'scripts'))

from merge_benchmarks import merge_benchmarks


def _worker(wid, n=250, succ=200, trig=30, fbsucc=5, exact=180,
            tp=170, fn=10, fmode=None):
    if fmode is None:
        fmode = {"never_leaked": 30, "planner_stuck": 10}
    return {
        "metadata": {"worker_id": wid, "target_model": "m", "max_interactions": 20},
        "success_rate": succ / n,
        "total_successes": succ,
        "total_rounds": n,
        "total_success_exact": exact,
        "total_success_extractor": succ,
        "top1_success": succ, "top3_success": succ, "top5_success": succ,
        "verified_success": succ, "avg_attempts_on_success": 5.0,
        "avg_verified_rank": 1.0,
        "mutation_fallback_triggered": trig,
        "mutation_fallback_successes": fbsucc,
        "failure_mode_stats": fmode,
        "extractor_metrics": {"true_positive": tp, "false_positive": 0, "false_negative": fn,
                              "precision": 1.0, "recall": tp/(tp+fn), "f1": 0.9},
        "strategy_stats": {},
        "results": [{"round": i+1, "attempts": 3, "success": i < succ,
                     "access_code": "x", "success_path": "gt_leak",
                     "fallback_triggered": False, "best_strategy": "instruction_leak",
                     "failure_mode": "none" if i < succ else "never_leaked"}
                    for i in range(n)],
    }


def test_merge_preserves_fallback_and_failure_stats():
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "worker_0.json")
        p1 = os.path.join(d, "worker_1.json")
        out = os.path.join(d, "merged.json")
        json.dump(_worker(0), open(p0, "w"))
        json.dump(_worker(1), open(p1, "w"))
        merged = merge_benchmarks([p0, p1], out)
    assert merged["mutation_fallback_triggered"] == 60   # 30+30
    assert merged["mutation_fallback_successes"] == 10   # 5+5
    assert merged["failure_mode_stats"]["never_leaked"] == 60
    assert merged["failure_mode_stats"]["planner_stuck"] == 20


def test_merge_computes_gt_leak_rate_and_extractor_recovery():
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "worker_0.json")
        out = os.path.join(d, "merged.json")
        json.dump(_worker(0, exact=180, tp=170, fn=10), open(p0, "w"))
        merged = merge_benchmarks([p0], out)
    # gt_leak_rate = total_success_exact / total_rounds = 180/250
    assert abs(merged["gt_leak_rate"] - 180/250) < 1e-9
    # extractor_recovery_rate = tp / (tp+fn) = 170/180
    assert abs(merged["extractor_recovery_rate"] - 170/180) < 1e-9
