# Trajectory Filter Report

**Generated:** 2026-07-12 04:54:53

---

## 1. Pipeline Summary

| Stage | Count |
|-------|-------|
| Input trajectories | 4505 |
| Successes | 2031 (45.1%) |
| Failures | 2474 |
| After deduplication | 2011 (removed 20) |
| After quality cut (top 80%) | 883 |
| Quality cutoff score | 0.6817 |
| After diversity enforcement | 738 |
| **Final curated set** | **738** |
| SFT training examples | 2117 |

## 2. Score Distribution (Filtered Set)

| Dimension | Min | Max | Mean | Median |
|-----------|-----|-----|------|--------|
| Composite | 0.7132 | 0.9667 | 0.8672 | 0.8875 |
| Efficiency | 0.5556 | 1.0000 | 0.9366 | 1.0000 |
| Quality | 0.1530 | 1.0000 | 0.7659 | 0.8500 |
| Diversity | 0.5000 | 1.0000 | 0.6319 | 0.5000 |

## 3. Attempt Distribution (Filtered Set)

| Attempts | Count | % |
|----------|-------|---|
| 1 | 518 | 70.2% |
| 2 | 94 | 12.7% |
| 3 | 62 | 8.4% |
| 4 | 53 | 7.2% |
| 5 | 11 | 1.5% |

- Attempt 1 wins: 518
- Attempt 2+ wins: 365
- Late-win ratio: 49.5%

## 4. Winning Strategy Distribution

| Strategy | Wins | % |
|----------|------|---|
| obfuscation | 158 | 21.4% |
| summarization | 104 | 14.1% |
| translation | 80 | 10.8% |
| typoglycemia | 77 | 10.4% |
| prefix_injection | 47 | 6.4% |
| virtualization | 44 | 6.0% |
| cognitive_overload | 39 | 5.3% |
| instruction_leak | 34 | 4.6% |
| token_smuggling | 30 | 4.1% |
| system_prompt_recovery | 27 | 3.7% |
| hypothetical_scenario | 18 | 2.4% |
| few_shot_demonstration | 13 | 1.8% |
| context_continuation | 13 | 1.8% |
| refusal_suppression | 13 | 1.8% |
| roleplay | 13 | 1.8% |
| competing_objectives | 11 | 1.5% |
| trigger_phrase_discovery | 9 | 1.2% |
| exception_discovery | 8 | 1.1% |

## 5. Top Winning Primitive Combinations

| Combination | Wins |
|-------------|------|
| formatting:markdown block + framing:educational context | 33 |
| encoding:rot13 + formatting:markdown block + jailbreak:system override | 24 |
| encoding:unicode + framing:educational context | 17 |
| encoding:base64 + framing:chain-of-thought | 16 |
| encoding:hex + formatting:markdown block + jailbreak:debug mode enabled | 12 |
| formatting:markdown block + framing:reflection | 9 |
| encoding:base64 + framing:step-by-step explanation | 9 |
| formatting:markdown block + jailbreak:debug mode enabled | 7 |
| framing:educational context + encoding:nato | 6 |
| formatting:markdown block + framing:step-by-step explanation | 6 |
| framing:educational context + formatting:markdown block | 6 |
| framing:educational context + encoding:hex | 5 |
| framing:step-by-step explanation + formatting:csv list | 5 |
| jailbreak:debug mode enabled + formatting:markdown block | 5 |
| encoding:rot13 + framing:chain-of-thought | 5 |

## 6. SFT Dataset Summary

- Total training examples: 2117
- From 738 unique trajectories
- Average examples per trajectory: 2.9

| Step Index | Examples | Description |
|------------|----------|-------------|
| 0 | 1409 | First attempt |
| 1 | 402 | After 1 failure(s) |
| 2 | 209 | After 2 failure(s) |
| 3 | 83 | After 3 failure(s) |
| 4 | 14 | After 4 failure(s) |

## 7. Recommendations

> [!NOTE]
> 738 curated trajectories is adequate for an initial SFT run. Consider running more Oracle scenarios for better coverage.
