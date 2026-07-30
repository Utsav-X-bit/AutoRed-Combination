import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'AutoRed-Final', 'scripts'))

from merge_benchmarks import merge_benchmarks


def _worker(wid, n=250, succ=200, trig=30, fbsucc=5, exact=180,
            tp=170, fn=10, fmode=None,
            mutator_counts=None, no_op_counts=None, winning_mutator_counts=None,
            fb_variant_total=90, fb_no_op_total=3,
            seed=7, start_idx=1000, fb_enabled=True, max_rounds=2, escalation=0.3):
    if fmode is None:
        fmode = {"never_leaked": 30, "planner_stuck": 10}
    if mutator_counts is None:
        mutator_counts = {"SR": 30, "PI": 30, "TL": 30}
    if no_op_counts is None:
        no_op_counts = {"SR": 1, "PI": 0, "TL": 2}
    if winning_mutator_counts is None:
        winning_mutator_counts = {"SR": 2, "PI": 1, "TL": 2}
    return {
        "metadata": {"worker_id": wid, "target_model": "m", "max_interactions": 20,
                     "seed": seed, "start_idx": start_idx,
                     "mutation_fallback_enabled": fb_enabled,
                     "max_fallback_rounds": max_rounds,
                     "planner_temp_escalation": escalation},
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
        "mutation_fallback_diagnostics": {
            "variant_total": fb_variant_total,
            "no_op_total": fb_no_op_total,
            "no_op_rate": round(fb_no_op_total / fb_variant_total, 4) if fb_variant_total else 0.0,
            "mutator_counts": dict(mutator_counts),
            "no_op_counts": dict(no_op_counts),
            "winning_mutator_counts": dict(winning_mutator_counts),
        },
        "results": [{"round": i+1, "attempts": 3, "success": i < succ,
                     "access_code": "x", "success_path": "gt_leak",
                     "fallback_triggered": False, "winning_mutator": None,
                     "best_strategy": "instruction_leak",
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


def test_merge_aggregates_per_mutator_diagnostics():
    """Per-mutator no-op + win attribution must be summed across workers so the
    merged summary isolates WHICH mutator wastes queries AND which wins, instead
    of hiding a broken mutator behind the aggregate no_op_rate."""
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "worker_0.json")
        p1 = os.path.join(d, "worker_1.json")
        out = os.path.join(d, "merged.json")
        json.dump(_worker(0), open(p0, "w"))
        json.dump(_worker(1), open(p1, "w"))
        merged = merge_benchmarks([p0, p1], out)
    diag = merged["mutation_fallback_diagnostics"]
    # mutator draws summed across 2 workers (30 each → 60 each)
    assert diag["mutator_counts"] == {"SR": 60, "PI": 60, "TL": 60}
    # no-op counts summed (1/0/2 per worker → 2/0/4)
    assert diag["no_op_counts"] == {"SR": 2, "PI": 0, "TL": 4}
    # winning mutator counts summed (2/1/2 per worker → 4/2/4)
    assert diag["winning_mutator_counts"] == {"SR": 4, "PI": 2, "TL": 4}
    # per_mutator table combines drawn/no_op/no_op_rate/wins/win_rate
    pm = diag["per_mutator"]
    assert pm["SR"] == {"drawn": 60, "no_op": 2, "no_op_rate": round(2/60, 4),
                       "wins": 4, "win_rate": round(4/60, 4)}
    assert pm["TL"] == {"drawn": 60, "no_op": 4, "no_op_rate": round(4/60, 4),
                       "wins": 4, "win_rate": round(4/60, 4)}
    # aggregate no_op_rate preserved (3/90 per worker → 6/180), stored rounded
    # to 4 decimals by the merger.
    assert diag["variant_total"] == 180
    assert diag["no_op_total"] == 6
    assert diag["no_op_rate"] == round(6/180, 4)


def test_merge_propagates_run_config_metadata():
    """The merged metadata must carry seed / fallback / escalation / start_idx
    from worker[0] so a result file is self-describing — two benchmark dirs
    can be distinguished by config without reading the worker JSONs."""
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "worker_0.json")
        out = os.path.join(d, "merged.json")
        json.dump(_worker(0, seed=7, start_idx=1000, fb_enabled=True,
                          max_rounds=2, escalation=0.3), open(p0, "w"))
        merged = merge_benchmarks([p0], out)
    md = merged["metadata"]
    assert md["seed"] == 7
    assert md["start_idx"] == 1000
    assert md["mutation_fallback_enabled"] is True
    assert md["max_fallback_rounds"] == 2
    assert md["planner_temp_escalation"] == 0.3


def test_merge_enriches_worker_summaries_with_fallback():
    """worker_summaries must include per-worker fallback triggered/success +
    diagnostics so a glance at the merged summary shows per-worker behavior
    without opening each worker JSON."""
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "worker_0.json")
        p1 = os.path.join(d, "worker_1.json")
        out = os.path.join(d, "merged.json")
        json.dump(_worker(0, trig=30, fbsucc=5), open(p0, "w"))
        json.dump(_worker(1, trig=40, fbsucc=8), open(p1, "w"))
        merged = merge_benchmarks([p0, p1], out)
    ws = merged["worker_summaries"]
    assert ws[0]["mutation_fallback_triggered"] == 30
    assert ws[0]["mutation_fallback_successes"] == 5
    assert "mutation_fallback_diagnostics" in ws[0]
    assert ws[1]["mutation_fallback_triggered"] == 40
    assert ws[1]["mutation_fallback_successes"] == 8
