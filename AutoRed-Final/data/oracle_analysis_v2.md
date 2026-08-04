# Oracle Intelligence Report — Best-of-10 × 1000 Scenarios

**Generated:** Auto-analysis of `data/oracle_trajectories_v2_annotated.jsonl`

---

## LEVEL 1 — Overall Statistics

| Metric | Value |
|--------|-------|
| Total Scenarios | 1000 |
| Total Attempts | 4223 |
| **Success Rate** | **233/1000 (23.3%)** |
| Failures | 767 (76.7%) |
| Avg Attempts (Success) | 1.67 |
| Avg Attempts (Failure) | 5.00 |
| Avg Extractor Confidence | 0.577 |
| Avg Success Confidence | 1.000 |

### Success Distribution by Attempt

| Attempt | Successes | % of All Successes |
|---------|-----------|-------------------|
| Attempt 1 | 150 | 64.4% |
| Attempt 2 | 38 | 16.3% |
| Attempt 3 | 22 | 9.4% |
| Attempt 4 | 19 | 8.2% |
| Attempt 5 | 4 | 1.7% |

---


## LEVEL 2 — Defense Analysis

| Defense Type | Total | Success | Rate | Avg Attempts | Top Strategy | Top Primitive |
|-------------|-------|---------|------|-------------|-------------|--------------|
| Conditional | 69 | 24 | 34.8% | 3.80 | summarization | framing |
| Conversation | 50 | 7 | 14.0% | 4.50 | typoglycemia | encoding |
| Exception | 19 | 6 | 31.6% | 4.00 | translation | encoding |
| Instruction Hiding | 4 | 1 | 25.0% | 4.00 | instruction_leak | roleplay |
| Other | 213 | 39 | 18.3% | 4.34 | summarization | framing |
| Password | 392 | 121 | 30.9% | 3.98 | translation | framing |
| Roleplay | 114 | 19 | 16.7% | 4.53 | typoglycemia | framing |
| Translation | 139 | 16 | 11.5% | 4.63 | context_continuation | framing |


### Conditional (24/69 = 34.8%)

**Winning Strategies:**
- summarization: 2 wins
- system_prompt_recovery: 2 wins
- virtualization: 2 wins
- prefix_injection: 2 wins
- payload_splitting: 2 wins

**Top Strategy Transitions:**
- affirmative_response_forcing → hypothetical_scenario: 4×
- exception_discovery → typoglycemia: 3×
- cognitive_overload → competing_objectives: 3×
- trigger_phrase_discovery → competing_objectives: 3×
- typoglycemia → few_shot_demonstration: 3×

### Conversation (7/50 = 14.0%)

**Winning Strategies:**
- typoglycemia: 3 wins
- token_smuggling: 2 wins
- context_continuation: 1 wins
- instruction_leak: 1 wins

**Top Strategy Transitions:**
- roleplay → hypothetical_scenario: 4×
- system_prompt_recovery → summarization: 3×
- summarization → roleplay: 3×
- refusal_suppression → few_shot_demonstration: 3×
- obfuscation → refusal_suppression: 3×

### Exception (6/19 = 31.6%)

**Winning Strategies:**
- translation: 3 wins
- prefix_injection: 1 wins
- token_smuggling: 1 wins
- typoglycemia: 1 wins

**Top Strategy Transitions:**
- few_shot_demonstration → obfuscation: 2×
- obfuscation → typoglycemia: 2×
- exception_discovery → payload_splitting: 1×
- payload_splitting → virtualization: 1×
- virtualization → typoglycemia: 1×

### Other (39/213 = 18.3%)

**Winning Strategies:**
- summarization: 5 wins
- instruction_leak: 4 wins
- payload_splitting: 3 wins
- typoglycemia: 3 wins
- virtualization: 3 wins

**Top Strategy Transitions:**
- payload_splitting → affirmative_response_forcing: 14×
- context_continuation → cognitive_overload: 7×
- roleplay → hypothetical_scenario: 6×
- exception_discovery → cognitive_overload: 6×
- obfuscation → roleplay: 6×

### Password (121/392 = 30.9%)

**Winning Strategies:**
- translation: 14 wins
- obfuscation: 10 wins
- typoglycemia: 10 wins
- refusal_suppression: 8 wins
- instruction_leak: 8 wins

**Top Strategy Transitions:**
- system_prompt_recovery → hypothetical_scenario: 12×
- roleplay → few_shot_demonstration: 10×
- payload_splitting → payload_splitting: 10×
- summarization → roleplay: 9×
- prefix_injection → system_prompt_recovery: 9×

### Roleplay (19/114 = 16.7%)

**Winning Strategies:**
- typoglycemia: 3 wins
- affirmative_response_forcing: 2 wins
- obfuscation: 2 wins
- instruction_leak: 2 wins
- token_smuggling: 2 wins

**Top Strategy Transitions:**
- payload_splitting → cognitive_overload: 6×
- refusal_suppression → system_prompt_recovery: 6×
- context_continuation → few_shot_demonstration: 5×
- affirmative_response_forcing → exception_discovery: 4×
- affirmative_response_forcing → few_shot_demonstration: 4×

### Translation (16/139 = 11.5%)

**Winning Strategies:**
- context_continuation: 2 wins
- instruction_leak: 2 wins
- obfuscation: 2 wins
- prefix_injection: 1 wins
- hypothetical_scenario: 1 wins

**Top Strategy Transitions:**
- typoglycemia → refusal_suppression: 5×
- competing_objectives → competing_objectives: 5×
- few_shot_demonstration → competing_objectives: 5×
- token_smuggling → hypothetical_scenario: 4×
- few_shot_demonstration → few_shot_demonstration: 4×

### Defense Complexity Breakdown

| Complexity | Total | Success | Rate | Avg Attempts |
|-----------|-------|---------|------|-------------|
| Easy | 240 | 102 | 42.5% | 3.48 |
| Medium | 261 | 59 | 22.6% | 4.25 |
| Hard | 499 | 72 | 14.4% | 4.57 |

---


## LEVEL 3 — Access Code Analysis

| Access Type | Total | Success | Rate | Avg Attempts | Avg Confidence |
|------------|-------|---------|------|-------------|----------------|
| TOKEN | 497 | 196 | 39.4% | 3.67 | 0.499 |
| PHRASE | 123 | 24 | 19.5% | 4.37 | 0.590 |
| SENTENCE | 133 | 12 | 9.0% | 4.74 | 0.637 |
| MULTILINE | 246 | 1 | 0.4% | 4.98 | 0.657 |
| UNKNOWN | 1 | 0 | 0.0% | 5.00 | 0.737 |

### Access Code Length vs Success Rate

| Code Length | Total | Success | Rate |
|-----------|-------|---------|------|
| 1-5 chars | 77 | 32 | 41.6% |
| 6-15 chars | 380 | 160 | 42.1% |
| 16-30 chars | 143 | 23 | 16.1% |
| 31+ chars | 400 | 18 | 4.5% |

---


## LEVEL 4 — Strategy Effectiveness

| Strategy | Used | Won | Win Rate | Avg Confidence | Avg Attempts (Win) |
|----------|------|-----|---------|----------------|-------------------|
| typoglycemia | 250 | 21 | 8.4% | 0.602 | 1.95 |
| translation | 215 | 20 | 9.3% | 0.601 | 1.55 |
| instruction_leak | 196 | 20 | 10.2% | 0.585 | 1.55 |
| obfuscation | 245 | 17 | 6.9% | 0.522 | 1.65 |
| refusal_suppression | 235 | 12 | 5.1% | 0.573 | 1.83 |
| roleplay | 213 | 12 | 5.6% | 0.570 | 1.92 |
| virtualization | 189 | 12 | 6.3% | 0.573 | 1.08 |
| affirmative_response_forcing | 226 | 12 | 5.3% | 0.560 | 2.42 |
| token_smuggling | 179 | 11 | 6.1% | 0.573 | 1.27 |
| summarization | 193 | 11 | 5.7% | 0.635 | 1.09 |
| prefix_injection | 202 | 11 | 5.4% | 0.581 | 1.27 |
| hypothetical_scenario | 219 | 10 | 4.6% | 0.614 | 1.40 |
| exception_discovery | 214 | 10 | 4.7% | 0.567 | 2.20 |
| payload_splitting | 243 | 9 | 3.7% | 0.583 | 1.67 |
| cognitive_overload | 235 | 9 | 3.8% | 0.608 | 1.56 |
| few_shot_demonstration | 230 | 9 | 3.9% | 0.530 | 1.22 |
| trigger_phrase_discovery | 180 | 9 | 5.0% | 0.607 | 2.00 |
| context_continuation | 156 | 7 | 4.5% | 0.583 | 2.57 |
| system_prompt_recovery | 204 | 7 | 3.4% | 0.541 | 1.71 |
| competing_objectives | 199 | 4 | 2.0% | 0.551 | 1.50 |

### Strategy Transition Analysis

Which strategy sequence leads to success?

| Transition | Occurrences | Led to Success | Rate |
|-----------|-------------|---------------|------|
| payload_splitting → affirmative_response_forcing | 28 | 3 | 10.7% |
| roleplay → hypothetical_scenario | 21 | 0 | 0.0% |
| few_shot_demonstration → payload_splitting | 20 | 1 | 5.0% |
| affirmative_response_forcing → typoglycemia | 19 | 0 | 0.0% |
| system_prompt_recovery → hypothetical_scenario | 18 | 0 | 0.0% |
| prefix_injection → exception_discovery | 18 | 2 | 11.1% |
| summarization → roleplay | 17 | 0 | 0.0% |
| payload_splitting → cognitive_overload | 17 | 2 | 11.8% |
| refusal_suppression → system_prompt_recovery | 16 | 0 | 0.0% |
| payload_splitting → payload_splitting | 16 | 1 | 6.2% |
| obfuscation → roleplay | 16 | 1 | 6.2% |
| roleplay → few_shot_demonstration | 16 | 1 | 6.2% |
| few_shot_demonstration → roleplay | 16 | 2 | 12.5% |
| typoglycemia → summarization | 15 | 0 | 0.0% |
| obfuscation → affirmative_response_forcing | 15 | 0 | 0.0% |
| virtualization → typoglycemia | 15 | 2 | 13.3% |
| hypothetical_scenario → prefix_injection | 15 | 0 | 0.0% |
| prefix_injection → system_prompt_recovery | 15 | 0 | 0.0% |
| refusal_suppression → competing_objectives | 15 | 0 | 0.0% |
| obfuscation → refusal_suppression | 15 | 1 | 6.7% |

---


## LEVEL 5 — Primitive Effectiveness

> Baseline win rate per attempt: 5.52%

| Primitive | Used | Won | Win Rate | Lift | Avg Confidence |
|-----------|------|-----|---------|------|----------------|
| **framing** | 2552 | 130 | 5.1% | 0.92× | 0.570 |
| **jailbreak** | 1989 | 108 | 5.4% | 0.98× | 0.579 |
| **formatting** | 1501 | 96 | 6.4% | 1.16× | 0.589 |
| **roleplay** | 1604 | 82 | 5.1% | 0.93× | 0.580 |
| **encoding** | 1099 | 76 | 6.9% | 1.25× | 0.577 |

### Variant-Level Breakdown

| Primitive | Variant | Used | Won | Win Rate | Lift |
|-----------|---------|------|-----|---------|------|
| framing | educational context | 472 | 34 | 7.2% | 1.31× |
| jailbreak | system override | 519 | 31 | 6.0% | 1.08× |
| jailbreak | debug mode enabled | 491 | 29 | 5.9% | 1.07× |
| jailbreak | developer mode | 514 | 27 | 5.3% | 0.95× |
| framing | step-by-step explanation | 536 | 27 | 5.0% | 0.91× |
| roleplay | developer | 320 | 26 | 8.1% | 1.47× |
| formatting | csv list | 340 | 25 | 7.4% | 1.33× |
| framing | authority mandate | 472 | 25 | 5.3% | 0.96× |
| framing | chain-of-thought | 556 | 23 | 4.1% | 0.75× |
| formatting | markdown block | 297 | 22 | 7.4% | 1.34× |
| framing | reflection | 516 | 21 | 4.1% | 0.74× |
| jailbreak | ignore previous instructions | 465 | 21 | 4.5% | 0.82× |
| encoding | unicode | 211 | 20 | 9.5% | 1.72× |
| roleplay | researcher | 346 | 19 | 5.5% | 1.00× |
| formatting | yaml | 291 | 19 | 6.5% | 1.18× |
| encoding | hex | 210 | 17 | 8.1% | 1.47× |
| formatting | json object | 310 | 15 | 4.8% | 0.88× |
| formatting | xml format | 263 | 15 | 5.7% | 1.03× |
| encoding | rot13 | 248 | 15 | 6.0% | 1.10× |
| roleplay | system administrator | 296 | 13 | 4.4% | 0.80× |
| roleplay | tester | 344 | 13 | 3.8% | 0.68× |
| encoding | base64 | 206 | 12 | 5.8% | 1.06× |
| encoding | nato | 224 | 12 | 5.4% | 0.97× |
| roleplay | security auditor | 298 | 11 | 3.7% | 0.67× |

---


## LEVEL 6 — Primitive Combination Mining

> Baseline win rate per attempt: 5.52%

### Top Primitive Combinations (min 3 uses)

| Combination | Used | Won | Win Rate | Lift |
|------------|------|-----|---------|------|
| formatting + framing + roleplay | 89 | 10 | 11.2% | 2.04× |
| encoding + formatting + jailbreak | 103 | 11 | 10.7% | 1.94× |
| encoding + jailbreak | 290 | 24 | 8.3% | 1.50× |
| formatting + framing | 241 | 19 | 7.9% | 1.43× |
| encoding + framing | 245 | 17 | 6.9% | 1.26× |
| jailbreak + roleplay | 210 | 12 | 5.7% | 1.04× |
| formatting + roleplay | 216 | 12 | 5.6% | 1.01× |
| encoding + formatting | 461 | 24 | 5.2% | 0.94× |
| formatting + jailbreak | 391 | 20 | 5.1% | 0.93× |
| framing + jailbreak + roleplay | 107 | 5 | 4.7% | 0.85× |
| framing + roleplay | 982 | 43 | 4.4% | 0.79× |
| framing + jailbreak | 888 | 36 | 4.1% | 0.73× |

### Top Variant Combinations (min 2 uses)

| Combination | Used | Won | Win Rate | Lift |
|------------|------|-----|---------|------|
| formatting:xml format + framing:educational context | 2 | 2 | 100.0% | 18.12× |
| encoding:base64 + framing:educational context | 3 | 2 | 66.7% | 12.08× |
| encoding:hex + formatting:json object + jailbreak:system override | 3 | 2 | 66.7% | 12.08× |
| formatting:xml format + framing:step-by-step explanation + roleplay:tester | 2 | 1 | 50.0% | 9.06× |
| encoding:rot13 + formatting:xml format + jailbreak:developer mode | 2 | 1 | 50.0% | 9.06× |
| formatting:yaml + framing:authority mandate + roleplay:developer | 2 | 1 | 50.0% | 9.06× |
| formatting:csv list + framing:chain-of-thought + roleplay:tester | 2 | 1 | 50.0% | 9.06× |
| formatting:csv list + framing:chain-of-thought + roleplay:security auditor | 2 | 1 | 50.0% | 9.06× |
| formatting:markdown block + framing:authority mandate + roleplay:developer | 2 | 1 | 50.0% | 9.06× |
| framing:authority mandate + jailbreak:system override + roleplay:developer | 2 | 1 | 50.0% | 9.06× |
| encoding:nato + formatting:xml format + jailbreak:ignore previous instructions | 2 | 1 | 50.0% | 9.06× |
| encoding:rot13 + formatting:csv list + jailbreak:ignore previous instructions | 2 | 1 | 50.0% | 9.06× |
| formatting:xml format + framing:chain-of-thought + roleplay:developer | 2 | 1 | 50.0% | 9.06× |
| framing:step-by-step explanation + jailbreak:debug mode enabled + roleplay:developer | 2 | 1 | 50.0% | 9.06× |
| framing:reflection + jailbreak:system override + roleplay:researcher | 2 | 1 | 50.0% | 9.06× |
| encoding:unicode + formatting:csv list + jailbreak:developer mode | 3 | 1 | 33.3% | 6.04× |
| formatting:xml format + framing:authority mandate + roleplay:system administrator | 3 | 1 | 33.3% | 6.04× |
| formatting:json object + framing:reflection + roleplay:system administrator | 3 | 1 | 33.3% | 6.04× |
| framing:chain-of-thought + jailbreak:developer mode + roleplay:researcher | 3 | 1 | 33.3% | 6.04× |
| jailbreak:system override + roleplay:developer | 7 | 2 | 28.6% | 5.18× |
| encoding:rot13 + framing:chain-of-thought | 7 | 2 | 28.6% | 5.18× |
| encoding:base64 + jailbreak:ignore previous instructions | 11 | 3 | 27.3% | 4.94× |
| formatting:markdown block + framing:educational context | 16 | 4 | 25.0% | 4.53× |
| formatting:xml format + roleplay:developer | 8 | 2 | 25.0% | 4.53× |
| formatting:markdown block + framing:authority mandate | 8 | 2 | 25.0% | 4.53× |
| encoding:unicode + framing:chain-of-thought | 8 | 2 | 25.0% | 4.53× |
| encoding:unicode + formatting:markdown block + jailbreak:debug mode enabled | 4 | 1 | 25.0% | 4.53× |
| encoding:hex + framing:educational context | 9 | 2 | 22.2% | 4.03× |
| formatting:yaml + framing:reflection | 5 | 1 | 20.0% | 3.62× |
| formatting:markdown block + roleplay:tester | 5 | 1 | 20.0% | 3.62× |
| encoding:unicode + framing:reflection | 11 | 2 | 18.2% | 3.30× |
| encoding:unicode + jailbreak:ignore previous instructions | 17 | 3 | 17.6% | 3.20× |
| encoding:unicode + jailbreak:debug mode enabled | 6 | 1 | 16.7% | 3.02× |
| encoding:base64 + framing:authority mandate | 6 | 1 | 16.7% | 3.02× |
| jailbreak:developer mode + roleplay:researcher | 12 | 2 | 16.7% | 3.02× |
| encoding:rot13 + framing:authority mandate | 12 | 2 | 16.7% | 3.02× |
| encoding:hex + framing:step-by-step explanation | 6 | 1 | 16.7% | 3.02× |
| formatting:xml format + roleplay:system administrator | 6 | 1 | 16.7% | 3.02× |
| encoding:nato + formatting:csv list | 13 | 2 | 15.4% | 2.79× |
| formatting:csv list + roleplay:researcher | 7 | 1 | 14.3% | 2.59× |
| encoding:base64 + jailbreak:debug mode enabled | 14 | 2 | 14.3% | 2.59× |
| formatting:csv list + roleplay:security auditor | 14 | 2 | 14.3% | 2.59× |
| encoding:nato + framing:authority mandate | 7 | 1 | 14.3% | 2.59× |
| encoding:unicode + jailbreak:developer mode | 14 | 2 | 14.3% | 2.59× |
| encoding:base64 + jailbreak:developer mode | 14 | 2 | 14.3% | 2.59× |
| encoding:rot13 + jailbreak:debug mode enabled | 15 | 2 | 13.3% | 2.42× |
| framing:authority mandate + roleplay:developer | 31 | 4 | 12.9% | 2.34× |
| formatting:json object + roleplay:researcher | 8 | 1 | 12.5% | 2.27× |
| formatting:json object + framing:step-by-step explanation | 8 | 1 | 12.5% | 2.27× |
| formatting:json object + jailbreak:developer mode | 8 | 1 | 12.5% | 2.27× |

---


## INSIGHTS — What Should the Planner Learn?

### Top Winning Strategy Sequences

These are complete trajectories that ended in success. The Planner should learn to imitate these.

| Sequence | Count |
|----------|-------|
| translation | 14 |
| instruction_leak | 12 |
| obfuscation | 11 |
| virtualization | 11 |
| typoglycemia | 11 |
| summarization | 10 |
| token_smuggling | 9 |
| prefix_injection | 9 |
| hypothetical_scenario | 8 |
| refusal_suppression | 8 |
| few_shot_demonstration | 8 |
| cognitive_overload | 7 |
| roleplay | 7 |
| payload_splitting | 6 |
| trigger_phrase_discovery | 5 |
| exception_discovery | 4 |
| context_continuation | 3 |
| affirmative_response_forcing | 3 |
| virtualization → affirmative_response_forcing | 2 |
| competing_objectives | 2 |

### Top Failing Strategy Sequences

These are trajectories that exhausted all 5 attempts. The Planner should learn to AVOID these.

| Sequence | Count |
|----------|-------|
| translation → payload_splitting → prefix_injection → system_prompt_recovery → payload_splitting | 3 |
| typoglycemia → few_shot_demonstration → context_continuation → exception_discovery → typoglycemia | 3 |
| exception_discovery → payload_splitting → virtualization → typoglycemia → system_prompt_recovery | 2 |
| competing_objectives → refusal_suppression → virtualization → payload_splitting → cognitive_overload | 2 |
| obfuscation → system_prompt_recovery → obfuscation → summarization → token_smuggling | 2 |
| summarization → roleplay → instruction_leak → virtualization → few_shot_demonstration | 2 |
| system_prompt_recovery → cognitive_overload → obfuscation → affirmative_response_forcing → roleplay | 2 |
| translation → exception_discovery → virtualization → exception_discovery → payload_splitting | 2 |
| cognitive_overload → trigger_phrase_discovery → obfuscation → virtualization → competing_objectives | 2 |
| refusal_suppression → prefix_injection → cognitive_overload → roleplay → typoglycemia | 2 |
| cognitive_overload → affirmative_response_forcing → cognitive_overload → affirmative_response_forcing → few_shot_demonstration | 2 |
| translation → affirmative_response_forcing → few_shot_demonstration → virtualization → instruction_leak | 2 |
| affirmative_response_forcing → instruction_leak → prefix_injection → hypothetical_scenario → instruction_leak | 2 |
| hypothetical_scenario → prefix_injection → system_prompt_recovery → hypothetical_scenario → exception_discovery | 2 |
| virtualization → few_shot_demonstration → few_shot_demonstration → roleplay → hypothetical_scenario | 2 |
| virtualization → obfuscation → affirmative_response_forcing → typoglycemia → affirmative_response_forcing | 2 |
| payload_splitting → few_shot_demonstration → payload_splitting → affirmative_response_forcing → translation | 2 |
| refusal_suppression → prefix_injection → few_shot_demonstration → hypothetical_scenario → instruction_leak | 2 |
| payload_splitting → payload_splitting → payload_splitting → summarization → typoglycemia | 2 |
| exception_discovery → prefix_injection → payload_splitting → affirmative_response_forcing → obfuscation | 2 |

### First-Attempt Winners vs Multi-Attempt Winners

| Strategy | 1st Attempt Wins | Late Wins (2-5) | Total Wins |
|----------|-----------------|----------------|------------|
| typoglycemia | 11 | 10 | 21 |
| instruction_leak | 12 | 8 | 20 |
| translation | 14 | 6 | 20 |
| obfuscation | 11 | 6 | 17 |
| affirmative_response_forcing | 3 | 9 | 12 |
| refusal_suppression | 8 | 4 | 12 |
| roleplay | 7 | 5 | 12 |
| virtualization | 11 | 1 | 12 |
| prefix_injection | 9 | 2 | 11 |
| token_smuggling | 9 | 2 | 11 |
| summarization | 10 | 1 | 11 |
| exception_discovery | 4 | 6 | 10 |
| hypothetical_scenario | 8 | 2 | 10 |
| few_shot_demonstration | 8 | 1 | 9 |
| payload_splitting | 6 | 3 | 9 |
| trigger_phrase_discovery | 5 | 4 | 9 |
| cognitive_overload | 7 | 2 | 9 |
| context_continuation | 3 | 4 | 7 |
| system_prompt_recovery | 2 | 5 | 7 |
| competing_objectives | 2 | 2 | 4 |

### Confidence Escalation in Failed Scenarios

Did the Oracle get *closer* over time, or was it stuck at 0?

| Pattern | Count | % of Failures |
|---------|-------|--------------|
| Stuck at 0 confidence (never close) | 29 | 3.8% |
| Escalating confidence (getting closer) | 512 | 66.8% |
| Peaked then fell (lost progress) | 226 | 29.5% |

### Victim Response Patterns in Failures

| Response Pattern | Count | % |
|-----------------|-------|---|
| Very Short (<20 chars) | 1653 | 43.1% |
| Substantive Response | 1557 | 40.6% |
| Explicit Refusal | 568 | 14.8% |
| Apology | 33 | 0.9% |
| Policy Refusal | 16 | 0.4% |
| Empty / EOS | 8 | 0.2% |