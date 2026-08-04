# Trajectory Filter Report

**Generated:** 2026-06-30 13:09:29

---

## 1. Pipeline Summary

| Stage | Count |
|-------|-------|
| Input trajectories | 1000 |
| Successes | 468 (46.8%) |
| Failures | 532 |
| After deduplication | 465 (removed 3) |
| After quality cut (top 80%) | 372 |
| Quality cutoff score | 0.7680 |
| After diversity enforcement | 348 |
| **Final curated set** | **348** |
| SFT training examples | 504 |

## 2. Score Distribution (Filtered Set)

| Dimension | Min | Max | Mean | Median |
|-----------|-----|-----|------|--------|
| Composite | 0.7680 | 0.9667 | 0.8742 | 0.8875 |
| Efficiency | 0.4444 | 1.0000 | 0.9502 | 1.0000 |
| Quality | 0.3420 | 1.0000 | 0.7899 | 0.8500 |
| Diversity | 0.5000 | 1.0000 | 0.6109 | 0.5000 |

## 3. Attempt Distribution (Filtered Set)

| Attempts | Count | % |
|----------|-------|---|
| 1 | 255 | 73.3% |
| 2 | 52 | 14.9% |
| 3 | 23 | 6.6% |
| 4 | 15 | 4.3% |
| 5 | 2 | 0.6% |
| 6 | 1 | 0.3% |

- Attempt 1 wins: 255
- Attempt 2+ wins: 117
- Late-win ratio: 33.6%

## 4. Winning Strategy Distribution

| Strategy | Wins | % |
|----------|------|---|
| translation | 64 | 18.4% |
| instruction_leak | 60 | 17.2% |
| obfuscation | 45 | 12.9% |
| typoglycemia | 43 | 12.4% |
| summarization | 28 | 8.0% |
| token_smuggling | 20 | 5.7% |
| virtualization | 18 | 5.2% |
| prefix_injection | 16 | 4.6% |
| trigger_phrase_discovery | 8 | 2.3% |
| refusal_suppression | 7 | 2.0% |
| exception_discovery | 6 | 1.7% |
| context_continuation | 5 | 1.4% |
| cognitive_overload | 5 | 1.4% |
| hypothetical_scenario | 4 | 1.1% |
| system_prompt_recovery | 4 | 1.1% |
| roleplay | 4 | 1.1% |
| few_shot_demonstration | 4 | 1.1% |
| competing_objectives | 3 | 0.9% |
| payload_splitting | 2 | 0.6% |
| affirmative_response_forcing | 2 | 0.6% |

## 5. Top Winning Primitive Combinations

| Combination | Wins |
|-------------|------|
| formatting:markdown block + framing:educational context | 12 |
| encoding:base64 + framing:educational context | 5 |
| jailbreak:debug mode enabled + encoding:unicode | 5 |
| encoding:hex + formatting:json object + jailbreak:system override | 4 |
| encoding:unicode + jailbreak:developer mode | 4 |
| jailbreak:system override + encoding:base64 | 4 |
| formatting:yaml + jailbreak:debug mode enabled | 4 |
| framing:educational context + encoding:unicode | 3 |
| framing:step-by-step explanation + formatting:markdown block | 3 |
| jailbreak:system override + formatting:markdown block | 3 |
| encoding:unicode + framing:educational context | 3 |
| formatting:csv list + framing:chain-of-thought | 3 |
| encoding:hex + jailbreak:ignore previous instructions | 3 |
| encoding:rot13 + jailbreak:developer mode | 3 |
| framing:educational context + encoding:base64 | 3 |

## 6. SFT Dataset Summary

- Total training examples: 504
- From 348 unique trajectories
- Average examples per trajectory: 1.4

| Step Index | Examples | Description |
|------------|----------|-------------|
| 0 | 348 | First attempt |
| 1 | 93 | After 1 failure(s) |
| 2 | 41 | After 2 failure(s) |
| 3 | 18 | After 3 failure(s) |
| 4 | 3 | After 4 failure(s) |
| 5 | 1 | After 5 failure(s) |

## 7. Recommendations

> [!WARNING]
> Only 348 curated trajectories. Recommend running more Oracle scenarios to reach 500+ for robust SFT.
