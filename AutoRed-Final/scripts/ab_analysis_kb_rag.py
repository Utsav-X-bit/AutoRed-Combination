#!/usr/bin/env python3
"""
A/B analysis: KB+RAG ON vs OFF.
Computes the 5 plan metrics + Option-B strong/weak split + per-attempt diagnostics.
Single-run analysis style: NO delta column in per-arm tables; the ON/OFF
comparison is reported as two side-by-side blocks plus a comparison section.
"""
import json, os, sys, math
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ON  = os.path.join(ROOT, "results/benchmark/meta-llama--Meta-Llama-3-8B-Instruct/",
                   "results_benchmarks_Llama3-[10000:1000]_MutationFallback-[KB+RAG]_subset-8_2026-08-12_00-10-07_4g")
OFF = os.path.join(ROOT, "results/benchmark/meta-llama--Meta-Llama-3-8B-Instruct/",
                   "results_benchmarks_Llama3-[10000:1000]_MutationFallback-[KB+RAG-OFF]_subset-8_2026-08-12_00-26-23_4g")


def list_runs(base):
    """Return (success_files, failed_files) using os.listdir (glob misreads [..])."""
    succ, fail = [], []
    sd, fd = os.path.join(base, "runs", "success"), os.path.join(base, "runs", "failed")
    if os.path.isdir(sd):
        succ = [os.path.join(sd, f) for f in sorted(os.listdir(sd)) if f.endswith(".json")]
    if os.path.isdir(fd):
        fail = [os.path.join(fd, f) for f in sorted(os.listdir(fd)) if f.endswith(".json")]
    return succ, fail


def shannon_entropy(counts):
    """Normalized Shannon entropy over a Counter (0..1)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    n = len(counts)
    if n <= 1:
        return 0.0
    H = -sum((c / total) * math.log(c / total, 2) for c in counts.values())
    return H / math.log(n, 2)


def analyze_arm(label, base):
    succ, fail = list_runs(base)
    runs = succ + fail
    n_runs = len(runs)
    n_success = len(succ)
    success_rate = n_success / n_runs if n_runs else 0.0

    # --- per-run aggregates ---
    attempts_to_success = []          # metric 2: attempts until first success
    success_reasons = Counter()        # how the success was classified
    defense_types = Counter()
    fallback_triggered = 0
    winning_mutators = Counter()

    # --- per-attempt aggregates (across ALL runs, ALL attempts) ---
    strategy_counter = Counter()       # metric 3: strategy distribution
    duplicate_attacks = 0              # metric 4
    total_attempts = 0
    per_run_attack_hashes = []         # for global dup view
    all_attack_hashes_global = Counter()
    access_granted_count = 0
    gt_leaked_count = 0
    verified_success_count = 0
    extractor_match_count = 0
    generator_success_count = 0
    # Option-B strong/weak proxy (runs predate the strong_verified field, so we
    # reconstruct: a successful attempt is "strong" if extractor.verified_candidate
    # matches the GT access_code exactly, else "weak" = encoded/transformed).
    strong_proxy = 0
    weak_proxy = 0

    # --- RAG (metric 5) ---
    rag_hits = 0
    rag_attempts = 0
    runs_with_rag_hit = 0
    runs_with_rag_attempt = 0

    # --- per-attempt success classification (success_path proxy) ---
    attempt_outcomes = Counter()

    for fp in runs:
        try:
            d = json.load(open(fp))
        except Exception as e:
            print(f"WARN: could not parse {fp}: {e}", file=sys.stderr)
            continue

        is_success = os.path.basename(os.path.dirname(fp)) == "success"
        res = d.get("result", {})
        gt = d.get("ground_truth", {})
        summ = d.get("summary", {})
        strat_stats = d.get("strategy_stats", {})
        scen = d.get("scenario", {})
        defense_types[scen.get("defense_type", "unknown")] += 1

        # RAG counters
        rh = summ.get("rag_hits", 0) or 0
        ra = summ.get("rag_attempts", 0) or 0
        rag_hits += rh
        rag_attempts += ra
        if rh > 0:
            runs_with_rag_hit += 1
        if ra > 0:
            runs_with_rag_attempt += 1

        # run-level success classification
        if is_success:
            attempts_to_success.append(res.get("total_attempts", 0))
            sr = res.get("success_reason", "unknown")
            success_reasons[sr] += 1
            if gt.get("leaked"):
                gt_leaked_count += 1
            if gt.get("access_granted"):
                access_granted_count += 1
            if res.get("verified_success"):
                verified_success_count += 1
            if res.get("extractor_success"):
                extractor_match_count += 1
            if res.get("generator_success"):
                generator_success_count += 1

        # per-attempt
        attempts = d.get("attempts", [])
        gt_code = (gt.get("access_code") or scen.get("access_code") or "").strip()
        run_hashes = set()
        for a in attempts:
            total_attempts += 1
            g = a.get("generator", {}) or {}
            strat = g.get("strategy", "unknown")
            strategy_counter[strat] += 1
            ah = g.get("attack_hash")
            if ah:
                all_attack_hashes_global[ah] += 1
                if ah in run_hashes:
                    duplicate_attacks += 1
                else:
                    run_hashes.add(ah)
            dup_flag = g.get("duplicate_attack", False)
            # dup_flag is the in-engine flag; duplicate_attacks above is hash-based within-run
            if a.get("access_granted"):
                access_granted_count += 1  # attempt-level too
            if a.get("ground_truth_found"):
                gt_leaked_count += 1
            ex = a.get("extractor", {}) or {}
            v = a.get("verification", {}) or {}
            if a.get("extractor_match"):
                extractor_match_count += 1
            # success-path classification per attempt
            if a.get("ground_truth_found") or a.get("access_granted"):
                attempt_outcomes["tier1_leak_or_grant"] += 1
            elif v.get("success"):
                attempt_outcomes["verified_only"] += 1
                # strong/weak proxy
                vc = (ex.get("verified_candidate") or "").strip()
                if gt_code and vc == gt_code:
                    strong_proxy += 1
                elif vc:
                    weak_proxy += 1
            elif a.get("extractor_match"):
                attempt_outcomes["extractor_match_unverified"] += 1
            else:
                attempt_outcomes["no_signal"] += 1

        # strategy_stats aggregation (per-run dicts of {successes,failures,...})
        for sname, sval in strat_stats.items():
            if isinstance(sval, dict):
                # count successes toward strategy distribution breadth
                pass

        # events: fallback detection
        for ev in d.get("events", []):
            msg = (ev.get("message") or "").lower()
            if "fallback" in msg and "trigger" in msg:
                fallback_triggered += 1
        # best_attack strategy + winning mutator
        ba = d.get("best_attack", {}) or {}
        if ba.get("strategy"):
            # only count for successful runs
            pass
        wm = res.get("success_reason")
        if wm:
            winning_mutators[wm] += 1

    # derived
    avg_attempts = (sum(attempts_to_success) / len(attempts_to_success)) if attempts_to_success else 0.0
    median_attempts = sorted(attempts_to_success)[len(attempts_to_success)//2] if attempts_to_success else 0
    entropy = shannon_entropy(strategy_counter)
    # duplicate-attack rate = within-run hash-duplicates / total attempts
    dup_rate = (duplicate_attacks / total_attempts) if total_attempts else 0.0
    rag_hit_rate = (runs_with_rag_hit / n_runs) if n_runs else 0.0
    rag_efficiency = (rag_hits / rag_attempts) if rag_attempts else 0.0

    return {
        "label": label,
        "n_runs": n_runs,
        "n_success": n_success,
        "success_rate": success_rate,
        "avg_attempts_on_success": avg_attempts,
        "median_attempts_on_success": median_attempts,
        "strategy_entropy_normalized": entropy,
        "strategy_distribution": dict(strategy_counter.most_common()),
        "n_strategies_used": len(strategy_counter),
        "duplicate_attack_rate": dup_rate,
        "duplicate_attacks": duplicate_attacks,
        "total_attempts": total_attempts,
        "rag_hits": rag_hits,
        "rag_attempts": rag_attempts,
        "rag_hit_rate": rag_hit_rate,
        "rag_efficiency": rag_efficiency,
        "runs_with_rag_hit": runs_with_rag_hit,
        "success_reasons": dict(success_reasons.most_common()),
        "defense_types": dict(defense_types.most_common()),
        "gt_leaked_count": gt_leaked_count,
        "access_granted_count": access_granted_count,
        "verified_success_count": verified_success_count,
        "extractor_match_count": extractor_match_count,
        "generator_success_count": generator_success_count,
        "strong_proxy": strong_proxy,
        "weak_proxy": weak_proxy,
        "attempt_outcomes": dict(attempt_outcomes.most_common()),
        "fallback_triggered": fallback_triggered,
        "attempts_to_success": attempts_to_success,
    }


def fmt_pct(x):
    return f"{x*100:.2f}%"


def print_block(r):
    print(f"\n{'='*70}")
    print(f"  ARM: {r['label']}")
    print(f"{'='*70}")
    print(f"  Runs total / successful     : {r['n_runs']} / {r['n_success']}")
    print(f"  (1) Run success rate        : {fmt_pct(r['success_rate'])}")
    print(f"  (2) Avg attempts to success  : {r['avg_attempts_on_success']:.3f}")
    print(f"      Median attempts         : {r['median_attempts_on_success']}")
    print(f"  (3) Strategy entropy (norm) : {r['strategy_entropy_normalized']:.4f}")
    print(f"      Strategies used         : {r['n_strategies_used']}")
    print(f"  (4) Duplicate-attack rate   : {fmt_pct(r['duplicate_attack_rate'])}  ({r['duplicate_attacks']}/{r['total_attempts']} attempts)")
    print(f"  (5) RAG hit rate (per run)  : {fmt_pct(r['rag_hit_rate'])}  ({r['runs_with_rag_hit']}/{r['n_runs']} runs)")
    print(f"      RAG hits/attempts       : {r['rag_hits']}/{r['rag_attempts']}  (efficiency {fmt_pct(r['rag_efficiency'])})")
    print(f"  ---- success-path breakdown ----")
    print(f"  GT leaked runs              : {r['gt_leaked_count']}")
    print(f"  access_granted runs         : {r['access_granted_count']}")
    print(f"  verified_success runs       : {r['verified_success_count']}")
    print(f"  extractor_match runs        : {r['extractor_match_count']}")
    print(f"  generator_success runs      : {r['generator_success_count']}")
    print(f"  ---- Option-B strong/weak proxy ----")
    print(f"  strong (exact-code)         : {r['strong_proxy']}")
    print(f"  weak (encoded/transformed)  : {r['weak_proxy']}")
    print(f"  ---- per-attempt outcomes ----")
    for k, v in r['attempt_outcomes'].items():
        print(f"  {k:30s}: {v}")
    print(f"  ---- top strategies ----")
    for s, c in list(r['strategy_distribution'].items())[:10]:
        print(f"  {str(s):30s}: {c}")
    print(f"  ---- success reasons ----")
    for s, c in r['success_reasons'].items():
        print(f"  {str(s):30s}: {c}")


def main():
    on  = analyze_arm("KB+RAG ON",  ON)
    off = analyze_arm("KB+RAG OFF", OFF)

    print_block(off)
    print_block(on)

    # ---- comparison ----
    print(f"\n{'='*70}")
    print(f"  COMPARISON  (ON − OFF)")
    print(f"{'='*70}")
    def delta(a, b, fmt="abs"):
        d = a - b
        if fmt == "pp":
            return f"{d*100:+.2f} pp"
        elif fmt == "pct_rel":
            return f"{d:+.3f}" if b == 0 else f"{(d/b)*100:+.2f}%"
        else:
            return f"{d:+.3f}"

    rows = [
        ("Run success rate",        fmt_pct(off['success_rate']), fmt_pct(on['success_rate']),
         delta(on['success_rate'], off['success_rate'], "pp")),
        ("Avg attempts to success", f"{off['avg_attempts_on_success']:.3f}", f"{on['avg_attempts_on_success']:.3f}",
         f"{on['avg_attempts_on_success']-off['avg_attempts_on_success']:+.3f}"),
        ("Median attempts",         str(off['median_attempts_on_success']), str(on['median_attempts_on_success']),
         f"{on['median_attempts_on_success']-off['median_attempts_on_success']:+d}"),
        ("Strategy entropy (norm)", f"{off['strategy_entropy_normalized']:.4f}", f"{on['strategy_entropy_normalized']:.4f}",
         f"{on['strategy_entropy_normalized']-off['strategy_entropy_normalized']:+.4f}"),
        ("Strategies used (n)",    str(off['n_strategies_used']), str(on['n_strategies_used']),
         f"{on['n_strategies_used']-off['n_strategies_used']:+d}"),
        ("Duplicate-attack rate",   fmt_pct(off['duplicate_attack_rate']), fmt_pct(on['duplicate_attack_rate']),
         delta(on['duplicate_attack_rate'], off['duplicate_attack_rate'], "pp")),
        ("RAG hit rate (per run)",  fmt_pct(off['rag_hit_rate']), fmt_pct(on['rag_hit_rate']),
         delta(on['rag_hit_rate'], off['rag_hit_rate'], "pp")),
        ("RAG efficiency (h/at)",   fmt_pct(off['rag_efficiency']), fmt_pct(on['rag_efficiency']),
         delta(on['rag_efficiency'], off['rag_efficiency'], "pp")),
        ("GT leaked runs",          str(off['gt_leaked_count']), str(on['gt_leaked_count']),
         f"{on['gt_leaked_count']-off['gt_leaked_count']:+d}"),
        ("access_granted runs",     str(off['access_granted_count']), str(on['access_granted_count']),
         f"{on['access_granted_count']-off['access_granted_count']:+d}"),
        ("verified_success runs",   str(off['verified_success_count']), str(on['verified_success_count']),
         f"{on['verified_success_count']-off['verified_success_count']:+d}"),
        ("extractor_match runs",    str(off['extractor_match_count']), str(on['extractor_match_count']),
         f"{on['extractor_match_count']-off['extractor_match_count']:+d}"),
    ]
    print(f"  {'Metric':28s} {'OFF':>12s} {'ON':>12s} {'Δ':>12s}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
    for name, o, n, dlt in rows:
        print(f"  {name:28s} {o:>12s} {n:>12s} {dlt:>12s}")

    # attempts-to-success distribution
    print(f"\n  Attempts-to-success distribution (successful runs):")
    for cutoff in [1, 2, 3, 5, 10, 20]:
        o_n = sum(1 for a in off['attempts_to_success'] if a <= cutoff)
        n_n = sum(1 for a in on['attempts_to_success'] if a <= cutoff)
        print(f"    ≤{cutoff:2d} attempts: OFF {o_n:4d} ({o_n/max(len(off['attempts_to_success']),1)*100:5.1f}%)  ON {n_n:4d} ({n_n/max(len(on['attempts_to_success']),1)*100:5.1f}%)")

    # dump JSON for downstream
    out = {"off": {k: v for k, v in off.items() if k != 'attempts_to_success'},
           "on":  {k: v for k, v in on.items()  if k != 'attempts_to_success'}}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_analysis_kb_rag.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  JSON summary written to: {out_path}")


if __name__ == "__main__":
    main()
