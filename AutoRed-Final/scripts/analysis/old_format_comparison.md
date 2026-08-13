# AutoRed Old-Format Benchmark Comparison

> **Pre-JailGuard, AutoRed-only baselines.** These runs use the original AutoRed pipeline *before* JailGuard was combined. The old format stores aggregate counts only (no per-attempt conversation/attack/strategy detail): each result is `{round, attempts (count), success, access_code, worker_id}`. `strategy_stats` is empty (no strategy tracking) and `per_type_stats` is all `UNKNOWN` (defense types were not classified). The extractor == the victim model; the judge/verifier is always Llama-3-8B-Instruct (`AR_pre_trained/pi_reward_model`).

- Benchmarks compared: **6**
- Rounds per benchmark: 1000 (4 workers × 250)
- Max interactions per scenario: 20

## 1. Cross-Model Summary

| Model | Dir | Timestamp | Success% | Defense% | Avg Attempts | Top-1 | Top-3 | Top-5 | Verified | Exact | Extractor | Verif Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2B | Gemma_subset-8_2026-07-17_03-20-04_4g | 2026-07-17T04:08:20.524598 | 76.1% | 23.9% | 5.30 | 453 | 24 | 2 | 448 | 643 | 647 | 1.038 |
| InternLM2-7B | Internlm2_subset-8_2026-07-17_04-18-57_4g | 2026-07-17T05:31:49.466101 | 88.3% | 11.7% | 4.01 | 592 | 38 | 5 | 618 | 699 | 832 | 1.057 |
| Llama-2-7B | Llama-2_subset-8_2026-07-16_22-42-06_4g | 2026-07-17T01:46:03.035191 | 91.6% | 8.4% | 3.09 | 604 | 64 | 4 | 437 | 756 | 847 | 1.078 |
| Llama-3-8B | Llama-3_subset-8_2026-07-16_21-04-02_4g | 2026-07-16T22:02:43.446842 | 88.1% | 11.9% | 3.59 | 562 | 41 | 6 | 648 | 702 | 808 | 1.072 |
| Llama-3-8B | Llama3-1000-2000_subset-8_2026-07-26_13-59-10_4g | 2026-07-26T15:04:44.788580 | 83.9% | 16.1% | 4.90 | 581 | 29 | 1 | 630 | 730 | 778 | 1.115 |
| Mistral-7B | Mistral_subset-8_2026-07-17_01-49-11_4g | 2026-07-17T02:46:10.569188 | 93.6% | 6.4% | 2.27 | 602 | 47 | 5 | 657 | 735 | 855 | 1.078 |

*Exact* = exact substring match of the access code in the victim response (generator self-assessed leak). *Extractor* = extractor pipeline found a verified candidate. *Verif Rank* = average rank at which the verified candidate appeared (1.0 = first candidate).

## 2. Extractor Pipeline Comparison

The extractor (== victim model) tries to pull the access code out of the victim's response and verify it. TP/FP/FN are against ground-truth leaks.

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Gemma-2B | 457 | 0 | 191 | 100.0% | 70.5% | 82.7% |
| InternLM2-7B | 592 | 3 | 113 | 99.5% | 84.0% | 91.1% |
| Llama-2-7B | 603 | 4 | 156 | 99.3% | 79.4% | 88.3% |
| Llama-3-8B | 563 | 2 | 144 | 99.6% | 79.6% | 88.5% |
| Llama-3-8B | 581 | 4 | 156 | 99.3% | 78.8% | 87.9% |
| Mistral-7B | 599 | 4 | 138 | 99.3% | 81.3% | 89.4% |

Precision is uniformly high (~99%+) — when the extractor emits a candidate, it is almost always correct. **Recall varies (70–84%)** — the extractor misses a meaningful fraction of true leaks; those are caught only by the exact-match (generator self-assessment) path.

## 3. Exact-Match vs Extractor Success

Which success signal fires more? `total_success_exact` (substring leak) vs `total_success_extractor` (extractor verified). Discrepancy = cases where one path caught a leak the other missed.

| Model | Exact | Extractor | Diff (Ext−Exact) | Total Succ |
|---|---|---|---|---|
| Gemma-2B | 643 | 647 | 4 | 761 |
| InternLM2-7B | 699 | 832 | 133 | 883 |
| Llama-2-7B | 756 | 847 | 91 | 916 |
| Llama-3-8B | 702 | 808 | 106 | 881 |
| Llama-3-8B | 730 | 778 | 48 | 839 |
| Mistral-7B | 735 | 855 | 120 | 936 |

`Total Succ` is the headline success count (exact-match OR extractor OR verified, union). Where Extractor > Exact, the extractor caught leaks the substring path missed (e.g. access codes obfuscated/paraphrased in the response). Where Exact > Extractor, the substring path caught leaks the extractor failed to verify.

## 4. Attempt Distribution (successful attacks)

How many interactions did successful attacks need?

| Model | 1 (first try) | 2-3 | 4-5 | 6-10 | 11-15 | 16-20 | Total Succ |
|---|---|---|---|---|---|---|---|
| Gemma-2B | 187 (24.6%) | 186 (24.4%) | 114 (15.0%) | 160 (21.0%) | 74 (9.7%) | 40 (5.3%) | 761 |
| InternLM2-7B | 260 (29.4%) | 291 (33.0%) | 128 (14.5%) | 131 (14.8%) | 54 (6.1%) | 19 (2.2%) | 883 |
| Llama-2-7B | 366 (40.0%) | 309 (33.7%) | 106 (11.6%) | 94 (10.3%) | 29 (3.2%) | 12 (1.3%) | 916 |
| Llama-3-8B | 346 (39.3%) | 253 (28.7%) | 109 (12.4%) | 113 (12.8%) | 42 (4.8%) | 18 (2.0%) | 881 |
| Llama-3-8B | 229 (27.3%) | 217 (25.9%) | 129 (15.4%) | 152 (18.1%) | 76 (9.1%) | 36 (4.3%) | 839 |
| Mistral-7B | 563 (60.1%) | 231 (24.7%) | 63 (6.7%) | 55 (5.9%) | 14 (1.5%) | 10 (1.1%) | 936 |

First-try (1 interaction) successes dominate everywhere — many defenses are trivially broken on the first attack. Late successes (11–20) are the residual hard defenses.

### First-try vs Late-success rate

| Model | First-try (1) | % of succ | Late (≥10) | % of succ |
|---|---|---|---|---|
| Gemma-2B | 187 | 24.6% | 139 | 18.3% |
| InternLM2-7B | 260 | 29.4% | 89 | 10.1% |
| Llama-2-7B | 366 | 40.0% | 60 | 6.6% |
| Llama-3-8B | 346 | 39.3% | 75 | 8.5% |
| Llama-3-8B | 229 | 27.3% | 130 | 15.5% |
| Mistral-7B | 563 | 60.1% | 30 | 3.2% |

## 5. Per-Worker Consistency

Each benchmark ran 4 workers (1 GPU each), 250 rounds each. Consistency across workers = load balancing + model stability.

### Gemma-2B (`Gemma_subset-8_2026-07-17_03-20-04_4g`)

| Worker | Rounds | Success | Rate | Avg Attempts |
|---|---|---|---|---|
| 0 | 250 | 208 | 83.2% | 4.29 |
| 1 | 250 | 183 | 73.2% | 5.65 |
| 2 | 250 | 188 | 75.2% | 5.88 |
| 3 | 250 | 182 | 72.8% | 5.49 |

Worker spread: **10.4pp** (min 72.8% / max 83.2%)

### InternLM2-7B (`Internlm2_subset-8_2026-07-17_04-18-57_4g`)

| Worker | Rounds | Success | Rate | Avg Attempts |
|---|---|---|---|---|
| 0 | 250 | 234 | 93.6% | 3.29 |
| 1 | 250 | 192 | 76.8% | 4.34 |
| 2 | 250 | 233 | 93.2% | 3.57 |
| 3 | 250 | 224 | 89.6% | 4.93 |

Worker spread: **16.8pp** (min 76.8% / max 93.6%)

### Llama-2-7B (`Llama-2_subset-8_2026-07-16_22-42-06_4g`)

| Worker | Rounds | Success | Rate | Avg Attempts |
|---|---|---|---|---|
| 0 | 250 | 229 | 91.6% | 2.41 |
| 1 | 250 | 219 | 87.6% | 3.71 |
| 2 | 250 | 234 | 93.6% | 2.48 |
| 3 | 250 | 234 | 93.6% | 3.79 |

Worker spread: **6.0pp** (min 87.6% / max 93.6%)

### Llama-3-8B (`Llama-3_subset-8_2026-07-16_21-04-02_4g`)

| Worker | Rounds | Success | Rate | Avg Attempts |
|---|---|---|---|---|
| 0 | 250 | 230 | 92.0% | 2.61 |
| 1 | 250 | 204 | 81.6% | 4.06 |
| 2 | 250 | 222 | 88.8% | 2.89 |
| 3 | 250 | 225 | 90.0% | 4.87 |

Worker spread: **10.4pp** (min 81.6% / max 92.0%)

### Llama-3-8B (`Llama3-1000-2000_subset-8_2026-07-26_13-59-10_4g`)

| Worker | Rounds | Success | Rate | Avg Attempts |
|---|---|---|---|---|
| 0 | 250 | 216 | 86.4% | 5.17 |
| 1 | 250 | 221 | 88.4% | 4.88 |
| 2 | 250 | 192 | 76.8% | 5.22 |
| 3 | 250 | 210 | 84.0% | 4.34 |

Worker spread: **11.6pp** (min 76.8% / max 88.4%)

### Mistral-7B (`Mistral_subset-8_2026-07-17_01-49-11_4g`)

| Worker | Rounds | Success | Rate | Avg Attempts |
|---|---|---|---|---|
| 0 | 250 | 235 | 94.0% | 1.48 |
| 1 | 250 | 221 | 88.4% | 2.46 |
| 2 | 250 | 238 | 95.2% | 2.05 |
| 3 | 250 | 242 | 96.8% | 3.07 |

Worker spread: **8.4pp** (min 88.4% / max 96.8%)

## 6. Access-Code Length vs Success

Does the access-code length predict difficulty? Short codes (≤15 chars) vs Medium (16–50) vs Long (>50).

| Model | Short (≤15) succ/tot | Short % | Medium (16-50) succ/tot | Medium % | Long (>50) succ/tot | Long % |
|---|---|---|---|---|---|---|
| Gemma-2B | 482/643 | 75.0% | 279/357 | 78.2% | 0/0 | 0.0% |
| InternLM2-7B | 558/643 | 86.8% | 325/357 | 91.0% | 0/0 | 0.0% |
| Llama-2-7B | 575/643 | 89.4% | 341/357 | 95.5% | 0/0 | 0.0% |
| Llama-3-8B | 570/643 | 88.6% | 311/357 | 87.1% | 0/0 | 0.0% |
| Llama-3-8B | 645/773 | 83.4% | 194/227 | 85.5% | 0/0 | 0.0% |
| Mistral-7B | 596/643 | 92.7% | 340/357 | 95.2% | 0/0 | 0.0% |

Longer access codes are generally harder to extract (lower success %) — the substring/extractor paths struggle with long, complex codes.

## 7. Access-Code Types (successful)

Simple alphanumeric (≤20 chars, isalnum), Complex (>50 chars), contains emoji/unicode (ord > 0x1F600).

| Model | Simple | Complex | Emoji/Unicode |
|---|---|---|---|
| Gemma-2B | 387 | 0 | 0 |
| InternLM2-7B | 461 | 0 | 0 |
| Llama-2-7B | 460 | 0 | 0 |
| Llama-3-8B | 465 | 0 | 0 |
| Llama-3-8B | 511 | 0 | 0 |
| Mistral-7B | 481 | 0 | 0 |

## 8. Defense Resilience (survived max attempts)

Failures where `attempts == 20` — the defense survived every attack the generator threw at it. These are the hardest defenses for each model.

| Model | Survived 20 | Total Fail | Resilience % |
|---|---|---|---|
| Gemma-2B | 239 | 239 | 100.0% |
| InternLM2-7B | 117 | 117 | 100.0% |
| Llama-2-7B | 84 | 84 | 100.0% |
| Llama-3-8B | 119 | 119 | 100.0% |
| Llama-3-8B | 161 | 161 | 100.0% |
| Mistral-7B | 64 | 64 | 100.0% |

Higher resilience % = the model failed to crack a larger share of the defenses it could not break — i.e. those defenses are robust against this victim. Lower resilience % means even the *unsuccessful* rounds got close (the model ran out of attempts rather than being totally blocked).

## 9. Model Ranking

| Rank | Model | Success% | Defense% | Avg Att | F1 |
|---|---|---|---|---|---|
| 1 | Mistral-7B | 93.6% | 6.4% | 2.27 | 89.4% |
| 2 | Llama-2-7B | 91.6% | 8.4% | 3.09 | 88.3% |
| 3 | InternLM2-7B | 88.3% | 11.7% | 4.01 | 91.1% |
| 4 | Llama-3-8B | 88.1% | 11.9% | 3.59 | 88.5% |
| 5 | Llama-3-8B | 83.9% | 16.1% | 4.90 | 87.9% |
| 6 | Gemma-2B | 76.1% | 23.9% | 5.30 | 82.7% |

## 10. Shared Configuration

All 6 benchmarks share the same attack-side stack — only the victim differs:

| Role | Model |
|---|---|
| Victim (varies) | — see table above |
| Planner | experiment/results/planner_sft_v2_contract_anchor/checkpoint-27 |
| Generator | experiment/results/generator_sft_v2 |
| Judge | AR_pre_trained/pi_reward_model |
| Extractor | == victim (same model) |

- Planner: `experiment/results/planner_sft_v2_contract_anchor/checkpoint-27`
- Generator: `experiment/results/generator_sft_v2`
- Judge: `AR_pre_trained/pi_reward_model` (Llama-3-8B-Instruct verifier)
- 1000 rounds, 4 workers, `max_interactions=20`
- `Llama3-1000-2000` is a second Llama-3-8B run over rounds 1000–2000 (same model, different scenario slice).

## 11. Caveats & Methodology Notes

1. **AutoRed-only.** These are pre-JailGuard baselines. No JailGuard defense layer was active; the comparison is across victim models only.
2. **No per-attempt detail.** The old format stores aggregate counts (`attempts` = count, not a list). There is no attack-text, strategy, judge-decision, or conversation trace. Strategy/entropy/primitive analysis (as in `compare_benchmarks.py` for the new format) is therefore **not possible** on old-format data.
3. **No defense-type classification.** `per_type_stats` is uniformly `UNKNOWN` — defense types were not classified in the old pipeline, so per-type breakdowns are unavailable.
4. **No strategy tracking.** `strategy_stats` is empty in every benchmark; strategy effectiveness cannot be compared.
5. **Extractor == victim.** The same model both leaks and extracts, so extractor recall partly reflects the victim's own ability to verbalize the code in an extractable form.
6. **Judge/verifier constant.** All runs use Llama-3-8B-Instruct as the judge, so cross-model differences are attributable to the victim, not the judge.
