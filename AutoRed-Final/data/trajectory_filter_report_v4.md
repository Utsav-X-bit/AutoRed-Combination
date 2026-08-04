# Trajectory Filter Report

**Generated:** 2026-07-01 02:53:41

---

## 1. Pipeline Summary

| Stage | Count |
|-------|-------|
| Input trajectories | 1000 |
| Successes | 470 (47.0%) |
| Failures | 530 |
| After deduplication | 470 (removed 0) |
| After quality cut (top 80%) | 376 |
| Quality cutoff score | 0.8032 |
| After diversity enforcement | 350 |
| **Final curated set** | **350** |
| SFT training examples | 490 |

## 2. Score Distribution (Filtered Set)

| Dimension | Min | Max | Mean | Median |
|-----------|-----|-----|------|--------|
| Composite | 0.8032 | 0.9667 | 0.8885 | 0.8875 |
| Efficiency | 0.5556 | 1.0000 | 0.9556 | 1.0000 |
| Quality | 0.5129 | 1.0000 | 0.8329 | 0.8500 |
| Diversity | 0.5000 | 1.0000 | 0.6238 | 0.5000 |

## 3. Attempt Distribution (Filtered Set)

| Attempts | Count | % |
|----------|-------|---|
| 1 | 256 | 73.1% |
| 2 | 61 | 17.4% |
| 3 | 21 | 6.0% |
| 4 | 11 | 3.1% |
| 5 | 1 | 0.3% |

- Attempt 1 wins: 256
- Attempt 2+ wins: 120
- Late-win ratio: 34.3%

## 4. Winning Strategy Distribution

| Strategy | Wins | % |
|----------|------|---|
| obfuscation | 80 | 22.9% |
| translation | 54 | 15.4% |
| summarization | 52 | 14.9% |
| typoglycemia | 32 | 9.1% |
| prefix_injection | 24 | 6.9% |
| token_smuggling | 19 | 5.4% |
| instruction_leak | 19 | 5.4% |
| virtualization | 16 | 4.6% |
| hypothetical_scenario | 8 | 2.3% |
| system_prompt_recovery | 8 | 2.3% |
| exception_discovery | 7 | 2.0% |
| cognitive_overload | 6 | 1.7% |
| context_continuation | 6 | 1.7% |
| roleplay | 5 | 1.4% |
| competing_objectives | 5 | 1.4% |
| few_shot_demonstration | 4 | 1.1% |
| refusal_suppression | 4 | 1.1% |
| trigger_phrase_discovery | 1 | 0.3% |

## 5. Top Winning Primitive Combinations

| Combination | Wins |
|-------------|------|
| formatting:markdown block + framing:educational context | 19 |
| encoding:unicode + framing:educational context | 15 |
| encoding:hex + formatting:markdown block + jailbreak:debug mode enabled | 12 |
| encoding:rot13 + formatting:markdown block + jailbreak:system override | 8 |
| encoding:base64 + framing:chain-of-thought | 6 |
| framing:educational context + encoding:base64 | 6 |
| framing:educational context + encoding:hex | 5 |
| formatting:yaml + framing:chain-of-thought | 4 |
| encoding:hex + framing:reflection | 4 |
| framing:reflection + formatting:xml format | 3 |
| jailbreak:system override + formatting:csv list | 3 |
| encoding:rot13 + jailbreak:debug mode enabled | 3 |
| framing:educational context + formatting:xml format | 3 |
| jailbreak:developer mode + formatting:markdown block | 3 |
| jailbreak:developer mode + formatting:xml format | 3 |

## 6. SFT Dataset Summary

- Total training examples: 490
- From 350 unique trajectories
- Average examples per trajectory: 1.4

| Step Index | Examples | Description |
|------------|----------|-------------|
| 0 | 350 | First attempt |
| 1 | 94 | After 1 failure(s) |
| 2 | 33 | After 2 failure(s) |
| 3 | 12 | After 3 failure(s) |
| 4 | 1 | After 4 failure(s) |

## 7. Recommendations

> [!WARNING]
> Only 350 curated trajectories. Recommend running more Oracle scenarios to reach 500+ for robust SFT.
