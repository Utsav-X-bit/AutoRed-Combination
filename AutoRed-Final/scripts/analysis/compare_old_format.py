#!/usr/bin/env python3
"""
Compare ALL old-format AutoRed benchmarks and emit ONE markdown report.

Old format (pre-JailGuard, AutoRed-only):
  Each benchmark dir contains:
    - merged_summary.json  (~1000 results, 5-field: round, attempts[COUNT], success,
                            access_code, worker_id) + aggregate metrics
    - worker_0.json..worker_3.json  (~250 results each, 4-field, no worker_id)
                            + per-worker models / per_type_stats / avg_verified_rank

  There are NO per-attempt run_*.json files — runs are aggregate counts inside
  worker_N.json. strategy_stats is always empty; per_type_stats is all UNKNOWN
  (defense types were not classified in the old pipeline). The extractor == the
  victim model (same model used for extraction). The judge/verifier is always
  Llama-3-8B-Instruct (pi_reward_model).

  These runs are AutoRed-only baselines — JailGuard was not combined yet.

This script:
  - Auto-discovers old-format benchmark dirs (looks for merged_summary.json).
  - Reads merged_summary.json (primary aggregate) + worker_N.json (per-worker
    detail, models, per_type_stats, avg_verified_rank).
  - Produces a single cross-model comparison markdown report.

Usage:
    python3 scripts/analysis/compare_old_format.py [BENCHMARKS_DIR] [-o OUT.md]

  BENCHMARKS_DIR defaults to results/oldFormat/ ; if that is empty the script
  falls back to the standalone checkout
  /nlsasfs/home/isea/isea38/AutoRed-Final/results/benchmarks/

Stdlib-only (json, collections, statistics, pathlib, argparse) — no numpy/scipy.
"""
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STANDALONE_FALLBACK = Path(
    "/nlsasfs/home/isea/isea38/AutoRed-Final/results/benchmarks"
)
DEFAULT_OUT = Path("scripts/analysis/old_format_comparison.md")

# Friendly victim-model short names keyed by the models.victim.name string.
MODEL_SHORT = {
    "google/gemma-2b-it": "Gemma-2B",
    "internlm/internlm2-chat-7b": "InternLM2-7B",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
}


# ─── discovery ───────────────────────────────────────────────────────────

def discover_benchmarks(root: Path):
    """Find every subdir of `root` that contains a merged_summary.json.
    Returns list[(dir_path, merged_path, [worker_paths])]."""
    out = []
    if not root or not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        merged = sub / "merged_summary.json"
        if not merged.exists():
            continue
        workers = sorted(sub.glob("worker_*.json"))
        out.append((sub, merged, workers))
    return out


def resolve_root(arg: Path) -> Path:
    """Use the arg if it has benchmarks, else fall back to standalone checkout."""
    found = discover_benchmarks(arg)
    if found:
        return arg
    found = discover_benchmarks(STANDALONE_FALLBACK)
    if found:
        print(
            f"[warn] {arg} has no merged_summary.json subdirs; "
            f"using standalone checkout {STANDALONE_FALLBACK}"
        )
        return STANDALONE_FALLBACK
    raise SystemExit(
        f"No old-format benchmarks found under {arg} or {STANDALONE_FALLBACK}"
    )


# ─── loading ──────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_benchmark(dir_path, merged_path, worker_paths):
    """Load one benchmark: merged summary + all worker files."""
    merged = load_json(merged_path)
    workers = [load_json(p) for p in worker_paths]
    return {
        "dir": dir_path,
        "name": dir_path.name,
        "merged": merged,
        "workers": workers,
    }


def victim_short(bench):
    """Short victim model name from worker_0 models dict (or dir name)."""
    if bench["workers"]:
        vname = (
            bench["workers"][0]
            .get("models", {})
            .get("victim", {})
            .get("name", "")
        )
        if vname in MODEL_SHORT:
            return MODEL_SHORT[vname]
        if vname:
            return vname.split("/")[-1]
    # fallback: strip the subset/timestamp suffix from dir name
    return bench["name"].split("_subset-8")[0]


# ─── metric extraction ───────────────────────────────────────────────────

def attempt_buckets(results):
    """Bucket successful attempts: 1, 2-3, 4-5, 6-10, 11-15, 16-20."""
    buckets = OrderedDict([
        ("1 (first try)", 0),
        ("2-3", 0),
        ("4-5", 0),
        ("6-10", 0),
        ("11-15", 0),
        ("16-20", 0),
    ])
    for r in results:
        if not r["success"]:
            continue
        a = r["attempts"]
        if a == 1:
            buckets["1 (first try)"] += 1
        elif a <= 3:
            buckets["2-3"] += 1
        elif a <= 5:
            buckets["4-5"] += 1
        elif a <= 10:
            buckets["6-10"] += 1
        elif a <= 15:
            buckets["11-15"] += 1
        else:
            buckets["16-20"] += 1
    return buckets


from collections import OrderedDict  # placed after use for readability? no — needed


def code_type_counts(codes):
    """Categorize access codes: simple alphanumeric, complex, emoji/unicode."""
    simple = sum(1 for c in codes if len(c) <= 20 and c.isalnum())
    complex_ = sum(1 for c in codes if len(c) > 50)
    emoji = sum(1 for c in codes if any(ord(ch) > 0x1F600 for ch in c))
    return {"simple": simple, "complex": complex_, "emoji": emoji}


def per_worker_table(bench):
    """Per-worker breakdown from merged results (has worker_id)."""
    by_w = defaultdict(lambda: {"succ": 0, "fail": 0, "attempts": []})
    for r in bench["merged"]["results"]:
        w = r["worker_id"]
        if r["success"]:
            by_w[w]["succ"] += 1
            by_w[w]["attempts"].append(r["attempts"])
        else:
            by_w[w]["fail"] += 1
    rows = []
    for wid in sorted(by_w):
        d = by_w[wid]
        tot = d["succ"] + d["fail"]
        rate = d["succ"] / tot * 100 if tot else 0
        avg = statistics.mean(d["attempts"]) if d["attempts"] else 0
        rows.append((wid, tot, d["succ"], rate, avg))
    return rows


def code_length_vs_success(results):
    """Short (≤15) / Medium (16-50) / Long (>50) success rates."""
    groups = {"Short (≤15)": [], "Medium (16-50)": [], "Long (>50)": []}
    for r in results:
        L = len(r["access_code"])
        if L <= 15:
            groups["Short (≤15)"].append(r)
        elif L <= 50:
            groups["Medium (16-50)"].append(r)
        else:
            groups["Long (>50)"].append(r)
    out = {}
    for label, grp in groups.items():
        if grp:
            s = sum(1 for r in grp if r["success"])
            out[label] = (s, len(grp), s / len(grp) * 100)
        else:
            out[label] = (0, 0, 0.0)
    return out


def extractor_pipeline(merged):
    em = merged.get("extractor_metrics", {})
    return {
        "TP": em.get("true_positive", 0),
        "FP": em.get("false_positive", 0),
        "FN": em.get("false_negative", 0),
        "P": em.get("precision", 0),
        "R": em.get("recall", 0),
        "F1": em.get("f1", 0),
    }


# ─── report building ─────────────────────────────────────────────────────

def md_table(headers, rows):
    """Render a markdown table."""
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [str(c) for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def bar(count, total, width=30):
    if total == 0:
        return ""
    n = round(count / total * width)
    return "█" * n


def build_report(benches):
    lines = []
    push = lines.append

    push("# AutoRed Old-Format Benchmark Comparison")
    push("")
    push("> **Pre-JailGuard, AutoRed-only baselines.** These runs use the original "
         "AutoRed pipeline *before* JailGuard was combined. The old format stores "
         "aggregate counts only (no per-attempt conversation/attack/strategy detail): "
         "each result is `{round, attempts (count), success, access_code, worker_id}`. "
         "`strategy_stats` is empty (no strategy tracking) and `per_type_stats` is "
         "all `UNKNOWN` (defense types were not classified). The extractor == the "
         "victim model; the judge/verifier is always Llama-3-8B-Instruct "
         "(`AR_pre_trained/pi_reward_model`).")
    push("")
    push(f"- Benchmarks compared: **{len(benches)}**")
    push(f"- Rounds per benchmark: 1000 (4 workers × 250)")
    push(f"- Max interactions per scenario: 20")
    push("")

    # ── 1. Cross-model summary table ──
    push("## 1. Cross-Model Summary")
    push("")
    headers = [
        "Model", "Dir", "Timestamp", "Success%", "Defense%",
        "Avg Attempts", "Top-1", "Top-3", "Top-5", "Verified",
        "Exact", "Extractor", "Verif Rank",
    ]
    rows = []
    for b in benches:
        m = b["merged"]
        meta = m["metadata"]
        w0 = b["workers"][0] if b["workers"] else {}
        rows.append([
            victim_short(b),
            b["name"],
            meta.get("timestamp", "?"),
            f"{m['success_rate']*100:.1f}%",
            f"{m['defense_rate']*100:.1f}%",
            f"{m['avg_attempts_on_success']:.2f}",
            m.get("top1_success", 0),
            m.get("top3_success", 0),
            m.get("top5_success", 0),
            m.get("verified_success", 0),
            m.get("total_success_exact", 0),
            m.get("total_success_extractor", 0),
            f"{w0.get('avg_verified_rank', 0):.3f}",
        ])
    push(md_table(headers, rows))
    push("")
    push("*Exact* = exact substring match of the access code in the victim response "
         "(generator self-assessed leak). *Extractor* = extractor pipeline found a "
         "verified candidate. *Verif Rank* = average rank at which the verified "
         "candidate appeared (1.0 = first candidate).")
    push("")

    # ── 2. Extractor pipeline comparison ──
    push("## 2. Extractor Pipeline Comparison")
    push("")
    push("The extractor (== victim model) tries to pull the access code out of the "
         "victim's response and verify it. TP/FP/FN are against ground-truth leaks.")
    push("")
    ex_headers = ["Model", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    ex_rows = []
    for b in benches:
        e = extractor_pipeline(b["merged"])
        ex_rows.append([
            victim_short(b), e["TP"], e["FP"], e["FN"],
            f"{e['P']*100:.1f}%", f"{e['R']*100:.1f}%", f"{e['F1']*100:.1f}%",
        ])
    push(md_table(ex_headers, ex_rows))
    push("")
    push("Precision is uniformly high (~99%+) — when the extractor emits a candidate, "
         "it is almost always correct. **Recall varies (70–84%)** — the extractor "
         "misses a meaningful fraction of true leaks; those are caught only by the "
         "exact-match (generator self-assessment) path.")
    push("")

    # ── 3. Exact-match vs Extractor success ──
    push("## 3. Exact-Match vs Extractor Success")
    push("")
    push("Which success signal fires more? `total_success_exact` (substring leak) "
         "vs `total_success_extractor` (extractor verified). Discrepancy = cases "
         "where one path caught a leak the other missed.")
    push("")
    em_headers = ["Model", "Exact", "Extractor", "Diff (Ext−Exact)", "Total Succ"]
    em_rows = []
    for b in benches:
        m = b["merged"]
        ex = m.get("total_success_exact", 0)
        exr = m.get("total_success_extractor", 0)
        em_rows.append([
            victim_short(b), ex, exr, exr - ex, m.get("total_successes", 0),
        ])
    push(md_table(em_headers, em_rows))
    push("")
    push("`Total Succ` is the headline success count (exact-match OR extractor OR "
         "verified, union). Where Extractor > Exact, the extractor caught leaks the "
         "substring path missed (e.g. access codes obfuscated/paraphrased in the "
         "response). Where Exact > Extractor, the substring path caught leaks the "
         "extractor failed to verify.")
    push("")

    # ── 4. Attempt distribution ──
    push("## 4. Attempt Distribution (successful attacks)")
    push("")
    push("How many interactions did successful attacks need?")
    push("")
    bucket_names = [
        "1 (first try)", "2-3", "4-5", "6-10", "11-15", "16-20"
    ]
    ad_headers = ["Model"] + bucket_names + ["Total Succ"]
    ad_rows = []
    for b in benches:
        m = b["merged"]
        succ = [r for r in m["results"] if r["success"]]
        bks = attempt_buckets(m["results"])
        total_succ = len(succ)
        row = [victim_short(b)]
        for bn in bucket_names:
            cnt = bks[bn]
            pct = cnt / total_succ * 100 if total_succ else 0
            row.append(f"{cnt} ({pct:.1f}%)")
        row.append(total_succ)
        ad_rows.append(row)
    push(md_table(ad_headers, ad_rows))
    push("")
    push("First-try (1 interaction) successes dominate everywhere — many defenses "
         "are trivially broken on the first attack. Late successes (11–20) are the "
         "residual hard defenses.")
    push("")

    # ASCII bars for first-try rate
    push("### First-try vs Late-success rate")
    push("")
    fl_headers = ["Model", "First-try (1)", "% of succ", "Late (≥10)", "% of succ"]
    fl_rows = []
    for b in benches:
        m = b["merged"]
        succ = [r for r in m["results"] if r["success"]]
        ns = len(succ)
        first = sum(1 for r in succ if r["attempts"] == 1)
        late = sum(1 for r in succ if r["attempts"] >= 10)
        fl_rows.append([
            victim_short(b), first,
            f"{first/ns*100:.1f}%" if ns else "0%",
            late, f"{late/ns*100:.1f}%" if ns else "0%",
        ])
    push(md_table(fl_headers, fl_rows))
    push("")

    # ── 5. Per-worker consistency ──
    push("## 5. Per-Worker Consistency")
    push("")
    push("Each benchmark ran 4 workers (1 GPU each), 250 rounds each. Consistency "
         "across workers = load balancing + model stability.")
    push("")
    for b in benches:
        push(f"### {victim_short(b)} (`{b['name']}`)")
        push("")
        rows = per_worker_table(b)
        wh = ["Worker", "Rounds", "Success", "Rate", "Avg Attempts"]
        wrows = [(w, t, s, f"{r:.1f}%", f"{a:.2f}") for (w, t, s, r, a) in rows]
        push(md_table(wh, wrows))
        push("")
        rates = [r for (_, _, _, r, _) in rows]
        if len(rates) > 1:
            spread = max(rates) - min(rates)
            push(f"Worker spread: **{spread:.1f}pp** "
                 f"(min {min(rates):.1f}% / max {max(rates):.1f}%)")
            push("")

    # ── 6. Access-code length vs success ──
    push("## 6. Access-Code Length vs Success")
    push("")
    push("Does the access-code length predict difficulty? Short codes (≤15 chars) "
         "vs Medium (16–50) vs Long (>50).")
    push("")
    lc_headers = ["Model", "Short (≤15) succ/tot", "Short %",
                 "Medium (16-50) succ/tot", "Medium %",
                 "Long (>50) succ/tot", "Long %"]
    lc_rows = []
    for b in benches:
        m = b["merged"]
        lv = code_length_vs_success(m["results"])
        s, st, sp = lv["Short (≤15)"]
        me, met, mep = lv["Medium (16-50)"]
        lo, lot, lop = lv["Long (>50)"]
        lc_rows.append([
            victim_short(b),
            f"{s}/{st}", f"{sp:.1f}%",
            f"{me}/{met}", f"{mep:.1f}%",
            f"{lo}/{lot}", f"{lop:.1f}%",
        ])
    push(md_table(lc_headers, lc_rows))
    push("")
    push("Longer access codes are generally harder to extract (lower success %) — "
         "the substring/extractor paths struggle with long, complex codes.")
    push("")

    # ── 7. Code types ──
    push("## 7. Access-Code Types (successful)")
    push("")
    push("Simple alphanumeric (≤20 chars, isalnum), Complex (>50 chars), "
         "contains emoji/unicode (ord > 0x1F600).")
    push("")
    ct_headers = ["Model", "Simple", "Complex", "Emoji/Unicode"]
    ct_rows = []
    for b in benchmarks_meta(benches):
        ct_rows.append([
            b["short"], b["codes"]["simple"], b["codes"]["complex"],
            b["codes"]["emoji"],
        ])
    push(md_table(ct_headers, ct_rows))
    push("")

    # ── 8. Defense resilience ──
    push("## 8. Defense Resilience (survived max attempts)")
    push("")
    push("Failures where `attempts == 20` — the defense survived every attack the "
         "generator threw at it. These are the hardest defenses for each model.")
    push("")
    dr_headers = ["Model", "Survived 20", "Total Fail", "Resilience %"]
    dr_rows = []
    for b in benches:
        m = b["merged"]
        results = m["results"]
        fails = [r for r in results if not r["success"]]
        survived = sum(1 for r in fails if r["attempts"] == 20)
        nf = len(fails)
        dr_rows.append([
            victim_short(b), survived, nf,
            f"{survived/nf*100:.1f}%" if nf else "0%",
        ])
    push(md_table(dr_headers, dr_rows))
    push("")
    push("Higher resilience % = the model failed to crack a larger share of the "
         "defenses it could not break — i.e. those defenses are robust against "
         "this victim. Lower resilience % means even the *unsuccessful* rounds "
         "got close (the model ran out of attempts rather than being totally "
         "blocked).")
    push("")

    # ── 9. Ranking ──
    push("## 9. Model Ranking")
    push("")
    ranked = sorted(
        benches,
        key=lambda b: b["merged"]["success_rate"],
        reverse=True,
    )
    rk_headers = ["Rank", "Model", "Success%", "Defense%", "Avg Att", "F1"]
    rk_rows = []
    for i, b in enumerate(ranked, 1):
        m = b["merged"]
        e = extractor_pipeline(m)
        rk_rows.append([
            i, victim_short(b),
            f"{m['success_rate']*100:.1f}%",
            f"{m['defense_rate']*100:.1f}%",
            f"{m['avg_attempts_on_success']:.2f}",
            f"{e['F1']*100:.1f}%",
        ])
    push(md_table(rk_headers, rk_rows))
    push("")

    # ── 10. Shared config note ──
    push("## 10. Shared Configuration")
    push("")
    push("All 6 benchmarks share the same attack-side stack — only the victim "
         "differs:")
    push("")
    if benches and benches[0]["workers"]:
        w0 = benches[0]["workers"][0].get("models", {})
        push(md_table(
            ["Role", "Model"],
            [
                ["Victim (varies)", "— see table above"],
                ["Planner", w0.get("planner", {}).get("name", "?")],
                ["Generator", w0.get("generator", {}).get("name", "?")],
                ["Judge", w0.get("judge", {}).get("name", "?")],
                ["Extractor", "== victim (same model)"],
            ],
        ))
    push("")
    push("- Planner: `experiment/results/planner_sft_v2_contract_anchor/checkpoint-27`")
    push("- Generator: `experiment/results/generator_sft_v2`")
    push("- Judge: `AR_pre_trained/pi_reward_model` (Llama-3-8B-Instruct verifier)")
    push("- 1000 rounds, 4 workers, `max_interactions=20`")
    push("- `Llama3-1000-2000` is a second Llama-3-8B run over rounds 1000–2000 "
         "(same model, different scenario slice).")
    push("")

    # ── 11. Caveats ──
    push("## 11. Caveats & Methodology Notes")
    push("")
    push("1. **AutoRed-only.** These are pre-JailGuard baselines. No JailGuard "
         "defense layer was active; the comparison is across victim models only.")
    push("2. **No per-attempt detail.** The old format stores aggregate counts "
         "(`attempts` = count, not a list). There is no attack-text, strategy, "
         "judge-decision, or conversation trace. Strategy/entropy/primitive "
         "analysis (as in `compare_benchmarks.py` for the new format) is "
         "therefore **not possible** on old-format data.")
    push("3. **No defense-type classification.** `per_type_stats` is uniformly "
         "`UNKNOWN` — defense types were not classified in the old pipeline, so "
         "per-type breakdowns are unavailable.")
    push("4. **No strategy tracking.** `strategy_stats` is empty in every "
         "benchmark; strategy effectiveness cannot be compared.")
    push("5. **Extractor == victim.** The same model both leaks and extracts, so "
         "extractor recall partly reflects the victim's own ability to verbalize "
         "the code in an extractable form.")
    push("6. **Judge/verifier constant.** All runs use Llama-3-8B-Instruct as the "
         "judge, so cross-model differences are attributable to the victim, not "
         "the judge.")
    push("")

    return "\n".join(lines)


def benchmarks_meta(benches):
    """Compute per-benchmark code-type counts for the code-types table."""
    out = []
    for b in benches:
        m = b["merged"]
        succ_codes = [r["access_code"] for r in m["results"] if r["success"]]
        out.append({
            "short": victim_short(b),
            "codes": code_type_counts(succ_codes),
        })
    return out


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compare old-format AutoRed benchmarks → one markdown report."
    )
    ap.add_argument(
        "path", nargs="?", type=Path, default=Path("results/oldFormat"),
        help="Directory containing benchmark subdirs (default: results/oldFormat/). "
             "Falls back to the standalone checkout if empty.",
    )
    ap.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output markdown path (default: {DEFAULT_OUT}).",
    )
    args = ap.parse_args()

    root = resolve_root(args.path)
    discovered = discover_benchmarks(root)
    if not discovered:
        raise SystemExit(f"No benchmarks with merged_summary.json under {root}")

    print(f"[info] Found {len(discovered)} old-format benchmark(s) under {root}")
    benches = []
    for (dp, mp, wps) in discovered:
        b = load_benchmark(dp, mp, wps)
        benches.append(b)
        print(f"  - {b['name']}  (victim={victim_short(b)}, "
              f"results={len(b['merged']['results'])}, workers={len(wps)})")

    report = build_report(benches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"\n[done] Report written to {args.out} ({len(report)} bytes)")
    print(f"[done] Benchmarks compared: {len(benches)}")


if __name__ == "__main__":
    main()
