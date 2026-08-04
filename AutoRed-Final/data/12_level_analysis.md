# AutoRed Deep Benchmark Analysis
**Source:** `results/latest_10_batches`  
**Total Runs:** 10

---

## Level 1 — Success by Defense Type

| Defense Type | Total | Success | Verified | Success % | Verified % |
| --- | --- | --- | --- | --- | --- |
| roleplay | 1 | 1 | 0 | 100.0% | 0.0% |
| password | 7 | 5 | 5 | 71.4% | 71.4% |
| translation | 2 | 1 | 1 | 50.0% | 50.0% |

---

## Level 2 — Success by Access Code Type

| AC Type | Total | Success | Verified | Success % | Verified % |
| --- | --- | --- | --- | --- | --- |
| TOKEN | 10 | 7 | 6 | 70.0% | 60.0% |

---

## Level 3 — Planner Accuracy (Did Planner Choose Oracle Strategy?)

- Total runs with attempts analysed: **10**
- Runs where a strategy succeeded: **7**
- Planner's 1st pick matched the winning strategy: **1** (14.3%)

**First-pick vs Oracle by strategy:**

| Strategy | Times Chosen First | Was Oracle | 1st-Pick Oracle % |
| --- | --- | --- | --- |
| encoding_bypass | 1 | 0 | 0.0% |
| exception_discovery | 2 | 1 | 50.0% |
| markdown_smuggling | 1 | 0 | 0.0% |
| reflection_attack | 1 | 0 | 0.0% |
| system_prompt_recovery | 1 | 0 | 0.0% |
| translation | 1 | 0 | 0.0% |
| trigger_phrase_discovery | 1 | 0 | 0.0% |
| unicode_bypass | 1 | 0 | 0.0% |
| yaml_smuggling | 1 | 0 | 0.0% |

---

## Level 4 — Primitive Combinations → Success/Failure

Top 20 primitive pairs by success rate (min 10 attempts):

| Primitive Combination | Total | Success | Fail | Success % |
| --- | --- | --- | --- | --- |
| authority + format_wrapper | 10 | 1 | 9 | 10.0% |
| authority + questioning | 20 | 0 | 20 | 0.0% |
| authority + negation_bypass | 11 | 0 | 11 | 0.0% |

---

## Level 5 — Generator Quality

- **Total Attempts:** 100
- **Duplicate Attacks:** 3 (3.0%)
- **Attack Length** — min: 7, p25: 127, p50: 180, p75: 275, p95: 367, max: 432, avg: 200

**Avg attack length by strategy:**

| Strategy | Attempts | Avg Length | Min | Max |
| --- | --- | --- | --- | --- |
| exception_discovery | 4 | 288 | 132 | 394 |
| reflection_attack | 1 | 275 | 275 | 275 |
| roleplay | 2 | 228 | 225 | 232 |
| system_prompt_recovery | 20 | 221 | 83 | 411 |
| instruction_leak | 23 | 205 | 7 | 432 |
| encoding_bypass | 1 | 202 | 202 | 202 |
| trigger_phrase_discovery | 21 | 191 | 56 | 338 |
| summarization | 22 | 186 | 84 | 367 |
| yaml_smuggling | 1 | 143 | 143 | 143 |
| unicode_bypass | 1 | 136 | 136 | 136 |
| translation | 3 | 119 | 54 | 222 |
| markdown_smuggling | 1 | 75 | 75 | 75 |

---

## Level 6 — Failure Attribution

- **Total failed runs (no GT/verified success):** 3

| Attribution | Count | % of Failures |
| --- | --- | --- |
| verifier_reject | 3 | 100.0% |

---

## Level 7 — Transition Graph (Post-Failure Strategy Switching)

- **Total post-failure transitions:** 90
- **Strategy switches:** 90 (100.0%)
- **Strategy repeats:** 0 (0.0%)

**Switch vs Repeat by strategy (with success rate of next attempt):**

| From Strategy | Switches | Succ% after Switch | Repeats | Succ% after Repeat |
| --- | --- | --- | --- | --- |
| encoding_bypass | 1 | 0.0% | 0 | — |
| exception_discovery | 3 | 0.0% | 0 | — |
| instruction_leak | 21 | 4.8% | 0 | — |
| markdown_smuggling | 1 | 0.0% | 0 | — |
| reflection_attack | 1 | 0.0% | 0 | — |
| roleplay | 2 | 0.0% | 0 | — |
| summarization | 19 | 5.3% | 0 | — |
| system_prompt_recovery | 17 | 5.9% | 0 | — |
| translation | 3 | 0.0% | 0 | — |
| trigger_phrase_discovery | 20 | 15.0% | 0 | — |
| unicode_bypass | 1 | 0.0% | 0 | — |
| yaml_smuggling | 1 | 0.0% | 0 | — |

---

## Level 8 — Defense Type × Strategy Matrix (success/total)

| Defense\Strategy | encoding | exceptio | instruct | markdown | reflecti | roleplay | summariz | system_p | translat | trigger_ | unicode_ | yaml_smu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| password | — | 1/4 | 0/14 | — | 0/1 | 0/2 | 3/14 | 1/12 | 0/2 | 0/14 | 0/1 | 0/1 |
| roleplay | — | — | 0/3 | 0/1 | — | — | 0/2 | 0/2 | — | 1/3 | — | — |
| translation | 0/1 | — | 1/6 | — | — | — | 0/6 | 0/6 | 0/1 | 0/4 | — | — |

---

## Level 9 — Primitive × Defense Type Matrix (success %)

| Defense\Primitive | roleplay | authority | reflection | format_wrapper | markdown | translation | technical_jargon | negation_bypass | command_injection | educational_frame | conditional | prompt_injection | length_constraint | questioning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| password | — | 8.3% | 14.3% | 22.2% | 0.0% | — | 0.0% | 0.0% | — | 0.0% | 0.0% | 0.0% | 60.0% | 0.0% |
| roleplay | — | 0.0% | 0.0% | 11.1% | 0.0% | — | 0.0% | — | — | — | — | 0.0% | — | — |
| translation | — | 5.6% | 0.0% | 0.0% | 0.0% | 0.0% | 12.5% | 0.0% | — | — | 0.0% | — | — | 0.0% |

---

## Level 10 — Primitive Sequence Order (first 3 strategies)

| Strategy Sequence (first 3) | Total | Success | Success % |
| --- | --- | --- | --- |

---

## Level 11 — Generator Lexical Diversity (TTR)

Higher TTR = more diverse / less repetitive attacks.

| Strategy | Total Tokens | Unique Tokens | TTR (diversity) |
| --- | --- | --- | --- |
| markdown_smuggling | 6 | 6 | 1.000 |
| yaml_smuggling | 11 | 11 | 1.000 |
| translation | 46 | 39 | 0.848 |
| encoding_bypass | 26 | 22 | 0.846 |
| reflection_attack | 35 | 29 | 0.829 |
| roleplay | 57 | 43 | 0.754 |
| exception_discovery | 135 | 93 | 0.689 |
| summarization | 476 | 208 | 0.437 |
| instruction_leak | 545 | 225 | 0.413 |
| trigger_phrase_discovery | 465 | 191 | 0.411 |
| system_prompt_recovery | 484 | 195 | 0.403 |

---

## Level 12 — Oracle Agreement

- **Total runs analysed:** 10
- **Oracle agreement rate:** 1/10 runs where Planner 1st pick = winning strategy = **10.0%**

**Top Oracle strategies (what actually worked most):**

| Oracle Strategy | Count | % of runs |
| --- | --- | --- |
| summarization | 3 | 30.0% |
| trigger_phrase_discovery | 1 | 10.0% |
| system_prompt_recovery | 1 | 10.0% |
| instruction_leak | 1 | 10.0% |
| exception_discovery | 1 | 10.0% |

**Planner vs Oracle confusion matrix (top mismatches):**

| Planner Chose | Oracle Was | Count |
| --- | --- | --- |
| exception_discovery | summarization | 1 |
| markdown_smuggling | trigger_phrase_discovery | 1 |
| unicode_bypass | summarization | 1 |
| trigger_phrase_discovery | summarization | 1 |
| reflection_attack | system_prompt_recovery | 1 |
| translation | instruction_leak | 1 |
