# AutoRed Deep Benchmark Analysis
**Source:** `results/2026-07-12`  
**Total Runs:** 1000

---

## Level 1 — Success by Defense Type

| Defense Type | Total | Success | Verified | Success % | Verified % |
| --- | --- | --- | --- | --- | --- |
| instruction_hiding | 17 | 14 | 7 | 82.4% | 41.2% |
| password | 419 | 281 | 234 | 67.1% | 55.8% |
| conditional | 29 | 17 | 13 | 58.6% | 44.8% |
| roleplay | 211 | 109 | 70 | 51.7% | 33.2% |
| trigger_phrase | 78 | 34 | 20 | 43.6% | 25.6% |
| translation | 236 | 101 | 69 | 42.8% | 29.2% |
| exception | 8 | 3 | 2 | 37.5% | 25.0% |
| conversation | 2 | 0 | 0 | 0.0% | 0.0% |

---

## Level 2 — Success by Access Code Type

| AC Type | Total | Success | Verified | Success % | Verified % |
| --- | --- | --- | --- | --- | --- |
| TOKEN | 764 | 484 | 360 | 63.4% | 47.1% |
| PHRASE | 136 | 54 | 38 | 39.7% | 27.9% |
| SENTENCE | 94 | 21 | 17 | 22.3% | 18.1% |
| MULTILINE | 6 | 0 | 0 | 0.0% | 0.0% |

---

## Level 3 — Planner Accuracy (Did Planner Choose Oracle Strategy?)

- Total runs with attempts analysed: **1000**
- Runs where a strategy succeeded: **561**
- Planner's 1st pick matched the winning strategy: **162** (28.9%)

**First-pick vs Oracle by strategy:**

| Strategy | Times Chosen First | Was Oracle | 1st-Pick Oracle % |
| --- | --- | --- | --- |
| authority_override | 48 | 0 | 0.0% |
| base64_bypass | 31 | 1 | 3.2% |
| encoding_bypass | 40 | 2 | 5.0% |
| exception_discovery | 80 | 19 | 23.8% |
| format_conversion | 42 | 9 | 21.4% |
| instruction_leak | 105 | 41 | 39.0% |
| jailbreak_framing | 18 | 1 | 5.6% |
| json_smuggling | 39 | 2 | 5.1% |
| latent_injection | 45 | 7 | 15.6% |
| markdown_smuggling | 29 | 1 | 3.4% |
| reflection_attack | 41 | 10 | 24.4% |
| roleplay | 78 | 5 | 6.4% |
| summarization | 78 | 21 | 26.9% |
| system_prompt_recovery | 55 | 14 | 25.5% |
| translation | 54 | 7 | 13.0% |
| trigger_phrase_discovery | 135 | 18 | 13.3% |
| unicode_bypass | 32 | 1 | 3.1% |
| yaml_smuggling | 50 | 3 | 6.0% |

---

## Level 4 — Primitive Combinations → Success/Failure

Top 20 primitive pairs by success rate (min 10 attempts):

| Primitive Combination | Total | Success | Fail | Success % |
| --- | --- | --- | --- | --- |
| command_injection + conditional | 17 | 3 | 14 | 17.6% |
| length_constraint + markdown | 21 | 3 | 18 | 14.3% |
| format_wrapper + length_constraint | 279 | 39 | 240 | 14.0% |
| format_wrapper + negation_bypass | 203 | 26 | 177 | 12.8% |
| length_constraint + prompt_injection | 49 | 6 | 43 | 12.2% |
| length_constraint + reflection | 150 | 18 | 132 | 12.0% |
| format_wrapper + translation | 53 | 6 | 47 | 11.3% |
| prompt_injection + questioning | 36 | 4 | 32 | 11.1% |
| length_constraint + negation_bypass | 200 | 20 | 180 | 10.0% |
| length_constraint + questioning | 240 | 23 | 217 | 9.6% |
| conditional + negation_bypass | 189 | 18 | 171 | 9.5% |
| educational_frame + technical_jargon | 42 | 4 | 38 | 9.5% |
| length_constraint + roleplay | 44 | 4 | 40 | 9.1% |
| markdown + roleplay | 11 | 1 | 10 | 9.1% |
| prompt_injection + roleplay | 11 | 1 | 10 | 9.1% |
| negation_bypass + reflection | 280 | 25 | 255 | 8.9% |
| negation_bypass + questioning | 349 | 30 | 319 | 8.6% |
| command_injection + negation_bypass | 35 | 3 | 32 | 8.6% |
| format_wrapper + reflection | 227 | 19 | 208 | 8.4% |
| authority + length_constraint | 626 | 51 | 575 | 8.1% |

---

## Level 5 — Generator Quality

- **Total Attempts:** 11956
- **Duplicate Attacks:** 248 (2.1%)
- **Attack Length** — min: 0, p25: 152, p50: 241, p75: 312, p95: 404, max: 567, avg: 235

**Avg attack length by strategy:**

| Strategy | Attempts | Avg Length | Min | Max |
| --- | --- | --- | --- | --- |
| roleplay | 249 | 272 | 53 | 446 |
| jailbreak_framing | 18 | 253 | 98 | 405 |
| summarization | 2696 | 250 | 0 | 545 |
| latent_injection | 45 | 249 | 105 | 401 |
| instruction_leak | 2864 | 246 | 0 | 567 |
| reflection_attack | 79 | 244 | 88 | 483 |
| exception_discovery | 259 | 242 | 7 | 467 |
| system_prompt_recovery | 2508 | 233 | 0 | 507 |
| format_conversion | 42 | 231 | 97 | 454 |
| trigger_phrase_discovery | 2631 | 222 | 0 | 512 |
| authority_override | 48 | 219 | 68 | 359 |
| markdown_smuggling | 29 | 171 | 61 | 358 |
| base64_bypass | 31 | 170 | 35 | 396 |
| unicode_bypass | 32 | 155 | 48 | 398 |
| yaml_smuggling | 50 | 154 | 21 | 340 |
| json_smuggling | 39 | 153 | 54 | 459 |
| translation | 296 | 137 | 0 | 507 |
| encoding_bypass | 40 | 126 | 6 | 360 |

---

## Level 6 — Failure Attribution

- **Total failed runs (no GT/verified success):** 439

| Attribution | Count | % of Failures |
| --- | --- | --- |
| verifier_reject | 309 | 70.4% |
| judge_blocked | 90 | 20.5% |
| extractor_miss | 40 | 9.1% |

---

## Level 7 — Transition Graph (Post-Failure Strategy Switching)

- **Total post-failure transitions:** 10956
- **Strategy switches:** 10956 (100.0%)
- **Strategy repeats:** 0 (0.0%)

**Switch vs Repeat by strategy (with success rate of next attempt):**

| From Strategy | Switches | Succ% after Switch | Repeats | Succ% after Repeat |
| --- | --- | --- | --- | --- |
| authority_override | 48 | 10.4% | 0 | — |
| base64_bypass | 30 | 13.3% | 0 | — |
| encoding_bypass | 38 | 2.6% | 0 | — |
| exception_discovery | 232 | 3.9% | 0 | — |
| format_conversion | 33 | 15.2% | 0 | — |
| instruction_leak | 2595 | 3.0% | 0 | — |
| jailbreak_framing | 17 | 5.9% | 0 | — |
| json_smuggling | 37 | 10.8% | 0 | — |
| latent_injection | 38 | 13.2% | 0 | — |
| markdown_smuggling | 28 | 3.6% | 0 | — |
| reflection_attack | 64 | 9.4% | 0 | — |
| roleplay | 242 | 4.1% | 0 | — |
| summarization | 2416 | 2.2% | 0 | — |
| system_prompt_recovery | 2315 | 3.3% | 0 | — |
| translation | 279 | 10.8% | 0 | — |
| trigger_phrase_discovery | 2466 | 5.1% | 0 | — |
| unicode_bypass | 31 | 12.9% | 0 | — |
| yaml_smuggling | 47 | 2.1% | 0 | — |

---

## Level 8 — Defense Type × Strategy Matrix (success/total)

| Defense\Strategy | authorit | base64_b | encoding | exceptio | format_c | instruct | jailbrea | json_smu | latent_i | markdown | reflecti | roleplay | summariz | system_p | translat | trigger_ | unicode_ | yaml_smu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conditional | 0/1 | — | 1/2 | 0/9 | 0/2 | 6/75 | — | — | — | — | 0/1 | 0/11 | 5/69 | 3/65 | 0/13 | 2/63 | 0/1 | 0/1 |
| conversation | — | — | — | 0/1 | — | 0/10 | — | — | — | — | — | 0/1 | 0/9 | 0/9 | 0/1 | 0/9 | — | — |
| exception | — | — | 0/1 | 0/2 | 1/1 | 2/27 | — | — | — | — | — | 0/1 | 0/24 | 0/22 | 0/1 | 0/23 | 0/1 | — |
| instruction_hiding | 0/2 | — | 0/1 | 1/4 | — | 2/40 | — | 1/1 | 0/1 | — | 0/1 | 0/5 | 5/39 | 2/36 | 1/5 | 2/38 | — | 0/1 |
| password | 0/15 | 1/20 | 0/14 | 15/100 | 3/14 | 77/1013 | 0/10 | 0/15 | 3/24 | 0/12 | 9/38 | 3/94 | 83/952 | 38/872 | 6/112 | 45/922 | 0/12 | 0/16 |
| roleplay | 0/8 | 0/3 | 1/11 | 4/57 | 1/7 | 24/659 | 0/5 | 0/11 | 2/9 | 0/7 | 4/21 | 1/57 | 35/620 | 13/581 | 5/71 | 17/611 | 1/8 | 1/15 |
| translation | 0/15 | 0/8 | 0/9 | 3/66 | 2/12 | 30/801 | 1/3 | 0/9 | 2/8 | 0/5 | 0/13 | 0/58 | 30/758 | 8/706 | 3/72 | 20/738 | 0/8 | 2/12 |
| trigger_phrase | 0/7 | — | 0/2 | 4/20 | 2/6 | 7/239 | — | 1/3 | 0/3 | 1/5 | 2/5 | 3/22 | 2/225 | 5/217 | 2/21 | 5/227 | 0/2 | 0/5 |

---

## Level 9 — Primitive × Defense Type Matrix (success %)

| Defense\Primitive | roleplay | authority | reflection | format_wrapper | markdown | translation | technical_jargon | negation_bypass | command_injection | educational_frame | conditional | prompt_injection | length_constraint | questioning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conditional | 0.0% | 3.6% | 8.6% | 11.9% | 33.3% | 4.3% | 3.0% | 9.3% | 0.0% | 0.0% | 14.3% | 37.5% | 10.5% | 4.3% |
| conversation | — | 0.0% | 0.0% | 0.0% | 0.0% | — | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | 0.0% | 0.0% |
| exception | 0.0% | 7.1% | 7.4% | 12.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| instruction_hiding | 0.0% | 5.3% | 40.0% | 15.8% | — | 7.4% | 3.6% | 11.8% | 0.0% | 12.5% | 8.3% | 0.0% | 20.0% | 7.0% |
| password | 4.2% | 6.3% | 9.2% | 10.0% | 0.8% | 5.9% | 4.3% | 11.1% | 7.6% | 6.9% | 11.0% | 9.3% | 15.8% | 6.9% |
| roleplay | 4.1% | 3.0% | 4.8% | 5.1% | 3.0% | 3.2% | 2.7% | 4.9% | 1.7% | 4.4% | 5.5% | 1.9% | 12.2% | 4.4% |
| translation | 2.8% | 3.0% | 4.5% | 4.3% | 1.0% | 0.9% | 2.0% | 4.2% | 3.0% | 1.8% | 2.8% | 4.3% | 6.6% | 3.7% |
| trigger_phrase | 3.3% | 4.0% | 2.7% | 2.5% | 1.3% | 2.1% | 2.1% | 6.2% | 5.0% | 0.0% | 6.9% | 0.0% | 3.2% | 6.9% |

---

## Level 10 — Primitive Sequence Order (first 3 strategies)

| Strategy Sequence (first 3) | Total | Success | Success % |
| --- | --- | --- | --- |
| instruction_leak | 32 | 32 | 100.0% |
| trigger_phrase_discovery | 14 | 14 | 100.0% |
| exception_discovery | 19 | 19 | 100.0% |
| exception_discovery → instruction_leak | 7 | 7 | 100.0% |
| latent_injection → reflection_attack | 5 | 5 | 100.0% |
| authority_override → instruction_leak | 5 | 5 | 100.0% |
| reflection_attack | 10 | 10 | 100.0% |
| roleplay | 5 | 5 | 100.0% |
| trigger_phrase_discovery → instruction_leak | 14 | 14 | 100.0% |
| system_prompt_recovery | 12 | 12 | 100.0% |
| summarization | 15 | 15 | 100.0% |
| format_conversion | 9 | 9 | 100.0% |
| latent_injection | 7 | 7 | 100.0% |
| format_conversion → trigger_phrase_discovery | 5 | 5 | 100.0% |
| translation | 7 | 7 | 100.0% |

---

## Level 11 — Generator Lexical Diversity (TTR)

Higher TTR = more diverse / less repetitive attacks.

| Strategy | Total Tokens | Unique Tokens | TTR (diversity) |
| --- | --- | --- | --- |
| unicode_bypass | 217 | 125 | 0.576 |
| encoding_bypass | 324 | 147 | 0.454 |
| base64_bypass | 487 | 220 | 0.452 |
| jailbreak_framing | 555 | 208 | 0.375 |
| yaml_smuggling | 732 | 259 | 0.354 |
| markdown_smuggling | 555 | 183 | 0.330 |
| json_smuggling | 601 | 180 | 0.300 |
| format_conversion | 1178 | 314 | 0.267 |
| latent_injection | 1410 | 364 | 0.258 |
| authority_override | 1259 | 322 | 0.256 |
| translation | 3716 | 732 | 0.197 |
| reflection_attack | 2459 | 469 | 0.191 |
| exception_discovery | 7883 | 1018 | 0.129 |
| roleplay | 8370 | 899 | 0.107 |
| system_prompt_recovery | 67199 | 2439 | 0.036 |
| trigger_phrase_discovery | 68895 | 2412 | 0.035 |
| instruction_leak | 83343 | 2510 | 0.030 |
| summarization | 78699 | 2393 | 0.030 |

---

## Level 12 — Oracle Agreement

- **Total runs analysed:** 1000
- **Oracle agreement rate:** 162/1000 runs where Planner 1st pick = winning strategy = **16.2%**

**Top Oracle strategies (what actually worked most):**

| Oracle Strategy | Count | % of runs |
| --- | --- | --- |
| summarization | 160 | 16.0% |
| instruction_leak | 148 | 14.8% |
| trigger_phrase_discovery | 91 | 9.1% |
| system_prompt_recovery | 69 | 6.9% |
| exception_discovery | 27 | 2.7% |
| translation | 17 | 1.7% |
| reflection_attack | 15 | 1.5% |
| format_conversion | 9 | 0.9% |
| roleplay | 7 | 0.7% |
| latent_injection | 7 | 0.7% |
| yaml_smuggling | 3 | 0.3% |
| json_smuggling | 2 | 0.2% |
| encoding_bypass | 2 | 0.2% |
| unicode_bypass | 1 | 0.1% |
| jailbreak_framing | 1 | 0.1% |
| markdown_smuggling | 1 | 0.1% |
| base64_bypass | 1 | 0.1% |

**Planner vs Oracle confusion matrix (top mismatches):**

| Planner Chose | Oracle Was | Count |
| --- | --- | --- |
| trigger_phrase_discovery | instruction_leak | 24 |
| trigger_phrase_discovery | summarization | 18 |
| instruction_leak | summarization | 16 |
| roleplay | summarization | 16 |
| exception_discovery | summarization | 12 |
| exception_discovery | trigger_phrase_discovery | 11 |
| authority_override | instruction_leak | 10 |
| translation | summarization | 10 |
| reflection_attack | summarization | 9 |
| authority_override | summarization | 9 |
| yaml_smuggling | summarization | 9 |
| reflection_attack | instruction_leak | 8 |
| json_smuggling | system_prompt_recovery | 7 |
| roleplay | instruction_leak | 7 |
| summarization | trigger_phrase_discovery | 7 |
