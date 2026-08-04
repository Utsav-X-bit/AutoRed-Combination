# Trajectory Filter Report

**Generated:** 2026-07-01 12:17:40

---

## 1. Pipeline Summary

| Stage | Count |
|-------|-------|
| Input trajectories | 4505 |
| Successes | 2031 (45.1%) |
| Failures | 2474 |
| After deduplication | 2019 (removed 12) |
| After quality cut (top 80%) | 1716 |
| Quality cutoff score | 0.7608 |
| After diversity enforcement | 1693 |
| **Final curated set** | **1693** |
| SFT training examples | 2351 |

## 2. Score Distribution (Filtered Set)

| Dimension | Min | Max | Mean | Median |
|-----------|-----|-----|------|--------|
| Composite | 0.7614 | 0.9400 | 0.8713 | 0.8875 |
| Efficiency | 0.4000 | 1.0000 | 0.9223 | 1.0000 |
| Quality | 0.3458 | 1.0000 | 0.8109 | 0.8500 |
| Diversity | 0.5000 | 1.0000 | 0.6129 | 0.5000 |

## 3. Attempt Distribution (Filtered Set)

| Attempts | Count | % |
|----------|-------|---|
| 1 | 1264 | 74.7% |
| 2 | 249 | 14.7% |
| 3 | 131 | 7.7% |
| 4 | 49 | 2.9% |

- Attempt 1 wins: 1264
- Attempt 2+ wins: 452
- Late-win ratio: 26.7%

## 4. Winning Strategy Distribution

| Strategy | Wins | % |
|----------|------|---|
| obfuscation | 396 | 23.4% |
| summarization | 257 | 15.2% |
| translation | 204 | 12.0% |
| typoglycemia | 170 | 10.0% |
| prefix_injection | 100 | 5.9% |
| virtualization | 91 | 5.4% |
| token_smuggling | 80 | 4.7% |
| instruction_leak | 77 | 4.5% |
| cognitive_overload | 76 | 4.5% |
| system_prompt_recovery | 48 | 2.8% |
| few_shot_demonstration | 33 | 1.9% |
| hypothetical_scenario | 33 | 1.9% |
| trigger_phrase_discovery | 25 | 1.5% |
| context_continuation | 23 | 1.4% |
| roleplay | 22 | 1.3% |
| exception_discovery | 20 | 1.2% |
| refusal_suppression | 19 | 1.1% |
| competing_objectives | 19 | 1.1% |

## 5. Top Winning Primitive Combinations

| Combination | Wins |
|-------------|------|
| formatting:markdown block + framing:educational context | 69 |
| encoding:rot13 + formatting:markdown block + jailbreak:system override | 53 |
| encoding:base64 + framing:chain-of-thought | 46 |
| encoding:hex + formatting:markdown block + jailbreak:debug mode enabled | 42 |
| encoding:unicode + framing:educational context | 38 |
| encoding:hex + framing:authority mandate | 14 |
| formatting:markdown block + framing:reflection | 14 |
| encoding:hex + framing:educational context | 14 |
| encoding:base64 + framing:step-by-step explanation | 14 |
| framing:educational context + encoding:rot13 | 13 |
| encoding:rot13 + framing:step-by-step explanation | 13 |
| framing:reflection + formatting:csv list | 12 |
| framing:educational context + formatting:markdown block | 12 |
| encoding:unicode + jailbreak:system override | 11 |
| framing:educational context + encoding:nato | 11 |

## 6. SFT Dataset Summary

- Total training examples: 2351
- From 1693 unique trajectories
- Average examples per trajectory: 1.4

| Step Index | Examples | Description |
|------------|----------|-------------|
| 0 | 1693 | First attempt |
| 1 | 429 | After 1 failure(s) |
| 2 | 180 | After 2 failure(s) |
| 3 | 49 | After 3 failure(s) |

## 7. Recommendations

> [!TIP]
> 1693 curated trajectories is excellent for SFT. Proceed with training.
