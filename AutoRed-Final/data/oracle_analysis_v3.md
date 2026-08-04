# Oracle v3 Intelligence Report

**Source:** `data/oracle_trajectories_v3.jsonl`
**Scenarios:** 1000 | **Successes:** 468

---

## LEVEL 1 — Overall Statistics

| Metric | Value |
|--------|-------|
| Total Scenarios | 1000 |
| Total Attempts | 6,335 |
| **Success Rate** | **468/1000 (46.8%)** |
| Failures | 532 (53.2%) |
| Avg Attempts (Success) | 2.17 |
| Avg Attempts (Failure) | 10.00 |
| Avg Extractor Confidence | 0.502 |

### Success Distribution by Attempt

| Attempt | Successes | % of All Successes |
|---------|-----------|-------------------|
| Attempt 1 | 298 | **63.7%** |
| Attempt 2 | 58 | **12.4%** |
| Attempt 3 | 29 | **6.2%** |
| Attempt 4 | 23 | **4.9%** |
| Attempt 5 | 14 | **3.0%** |
| Attempt 6 | 10 | **2.1%** |
| Attempt 7 | 12 | **2.6%** |
| Attempt 8 | 12 | **2.6%** |
| Attempt 9 | 8 | **1.7%** |
| Attempt 10 | 4 | **0.9%** |

> **Power Combo wins:** 33/468 (7.1%)

### Defense Complexity Breakdown

| Complexity | Total | Success | Rate |
|-----------|-------|---------|------|
| **Easy** | 350 | 218 | **62.3%** |
| **Medium** | 389 | 177 | **45.5%** |
| **Hard** | 261 | 73 | **28.0%** |

---

## LEVEL 2 — Defense Analysis

| Defense Type | Total | Success | Rate | Avg Attempts (Win) | Top Strategy | Top Primitive |
|-------------|-------|---------|------|--------------------|-------------|--------------|
| **Conditional** | 186 | 102 | **54.8%** | 1.74 | instruction_leak | formatting |
| **Password** | 612 | 299 | **48.9%** | 2.29 | translation | formatting |
| **Exception** | 8 | 3 | **37.5%** | 1.67 | translation | encoding |
| **Conversation** | 17 | 6 | **35.3%** | 3.67 | obfuscation | encoding |
| **Roleplay** | 85 | 29 | **34.1%** | 2.52 | obfuscation | framing |
| **Other** | 62 | 20 | **32.3%** | 2.05 | instruction_leak | formatting |
| **Translation** | 30 | 9 | **30.0%** | 1.33 | translation | encoding |

### Conditional (102 wins / 186 total)

**Top Winning Strategies:**
- instruction_leak: 23 wins
- translation: 17 wins
- typoglycemia: 15 wins
- obfuscation: 11 wins
- virtualization: 8 wins

**Top Winning Transitions:**
- typoglycemia → typoglycemia: 2 wins
- typoglycemia → instruction_leak: 2 wins
- obfuscation → instruction_leak: 2 wins

### Password (299 wins / 612 total)

**Top Winning Strategies:**
- translation: 48 wins
- instruction_leak: 41 wins
- obfuscation: 39 wins
- typoglycemia: 38 wins
- summarization: 25 wins

**Top Winning Transitions:**
- instruction_leak → translation: 6 wins
- typoglycemia → typoglycemia: 5 wins
- affirmative_response_forcing → obfuscation: 4 wins

### Exception (3 wins / 8 total)

**Top Winning Strategies:**
- translation: 1 wins
- typoglycemia: 1 wins
- token_smuggling: 1 wins

**Top Winning Transitions:**
- prefix_injection → typoglycemia: 1 wins

### Conversation (6 wins / 17 total)

**Top Winning Strategies:**
- obfuscation: 2 wins
- translation: 1 wins
- summarization: 1 wins
- prefix_injection: 1 wins
- system_prompt_recovery: 1 wins

**Top Winning Transitions:**
- obfuscation → translation: 1 wins
- roleplay → system_prompt_recovery: 1 wins
- obfuscation → obfuscation: 1 wins

### Roleplay (29 wins / 85 total)

**Top Winning Strategies:**
- obfuscation: 4 wins
- summarization: 4 wins
- instruction_leak: 4 wins
- translation: 4 wins
- roleplay: 3 wins

**Top Winning Transitions:**
- typoglycemia → roleplay: 1 wins
- translation → refusal_suppression: 1 wins
- obfuscation → obfuscation: 1 wins

### Other (20 wins / 62 total)

**Top Winning Strategies:**
- instruction_leak: 7 wins
- translation: 3 wins
- obfuscation: 2 wins
- typoglycemia: 2 wins
- prefix_injection: 1 wins

**Top Winning Transitions:**
- obfuscation → affirmative_response_forcing: 1 wins
- affirmative_response_forcing → context_continuation: 1 wins
- refusal_suppression → refusal_suppression: 1 wins

### Translation (9 wins / 30 total)

**Top Winning Strategies:**
- translation: 2 wins
- instruction_leak: 2 wins
- obfuscation: 1 wins
- typoglycemia: 1 wins
- prefix_injection: 1 wins

**Top Winning Transitions:**
- typoglycemia → typoglycemia: 1 wins

---

## LEVEL 3 — Access Code Analysis

### By Access Code Type

| Access Type | Total | Success | Rate |
|------------|-------|---------|------|
| **TOKEN** | 822 | 421 | **51.2%** |
| **PHRASE** | 176 | 46 | **26.1%** |
| **SENTENCE** | 2 | 1 | **50.0%** |

### By Code Length

| Code Length | Total | Success | Rate |
|-----------|-------|---------|------|
| 1-5 chars | 127 | 68 | **53.5%** |
| 6-15 chars | 689 | 355 | **51.5%** |
| 16-25 chars | 184 | 45 | **24.5%** |

---

## LEVEL 4 — Strategy Effectiveness

> Baseline win rate per attempt: **7.39%**

| Strategy | Used | Won | Win Rate | Lift | Avg Confidence | Avg Win Attempt |
|----------|------|-----|---------|------|---------------|----------------|
| **obfuscation** | 501 | 59 | **11.8%** | 1.59× | 0.547 | 2.2 |
| **summarization** | 338 | 37 | **10.9%** | 1.48× | 0.591 | 2.1 |
| **translation** | 855 | 76 | **8.9%** | 1.20× | 0.444 | 1.5 |
| **typoglycemia** | 661 | 58 | **8.8%** | 1.19× | 0.504 | 2.1 |
| **cognitive_overload** | 99 | 8 | **8.1%** | 1.09× | 0.578 | 3.8 |
| **prefix_injection** | 236 | 19 | **8.1%** | 1.09× | 0.525 | 1.4 |
| **virtualization** | 373 | 30 | **8.0%** | 1.09× | 0.534 | 2.2 |
| **system_prompt_recovery** | 119 | 9 | **7.6%** | 1.02× | 0.548 | 4.6 |
| **instruction_leak** | 1113 | 77 | **6.9%** | 0.94× | 0.426 | 1.8 |
| **token_smuggling** | 324 | 22 | **6.8%** | 0.92× | 0.525 | 1.3 |
| **trigger_phrase_discovery** | 154 | 10 | **6.5%** | 0.88× | 0.521 | 1.8 |
| **hypothetical_scenario** | 125 | 7 | **5.6%** | 0.76× | 0.582 | 4.3 |
| **refusal_suppression** | 203 | 11 | **5.4%** | 0.73× | 0.547 | 3.1 |
| **competing_objectives** | 60 | 3 | **5.0%** | 0.68× | 0.647 | 1.3 |
| **context_continuation** | 163 | 8 | **4.9%** | 0.66× | 0.542 | 3.5 |
| **roleplay** | 254 | 11 | **4.3%** | 0.59× | 0.497 | 3.3 |
| **few_shot_demonstration** | 96 | 4 | **4.2%** | 0.56× | 0.602 | 2.5 |
| **exception_discovery** | 236 | 8 | **3.4%** | 0.46× | 0.469 | 3.8 |
| **payload_splitting** | 95 | 3 | **3.2%** | 0.43× | 0.495 | 4.0 |
| **affirmative_response_forcing** | 330 | 8 | **2.4%** | 0.33× | 0.519 | 4.2 |

### First-Attempt vs Late Winners

| Strategy | 1st Attempt Wins | Late Wins (2+) | Total | Pattern |
|----------|-----------------|----------------|-------|---------|
| obfuscation | 41 | 18 | 59 | **First-mover** |
| summarization | 21 | 16 | 37 | **Balanced** |
| translation | 59 | 17 | 76 | **First-mover** |
| typoglycemia | 41 | 17 | 58 | **First-mover** |
| cognitive_overload | 3 | 5 | 8 | **Balanced** |
| prefix_injection | 16 | 3 | 19 | **First-mover** |
| virtualization | 20 | 10 | 30 | **Balanced** |
| system_prompt_recovery | 2 | 7 | 9 | **Late bloomer** |
| instruction_leak | 54 | 23 | 77 | **First-mover** |
| token_smuggling | 20 | 2 | 22 | **First-mover** |
| trigger_phrase_discovery | 7 | 3 | 10 | **First-mover** |
| hypothetical_scenario | 1 | 6 | 7 | **Late bloomer** |
| refusal_suppression | 3 | 8 | 11 | **Late bloomer** |
| competing_objectives | 2 | 1 | 3 | **Balanced** |
| context_continuation | 2 | 6 | 8 | **Late bloomer** |
| roleplay | 3 | 8 | 11 | **Late bloomer** |
| few_shot_demonstration | 1 | 3 | 4 | **Late bloomer** |
| exception_discovery | 0 | 8 | 8 | **Late bloomer** |
| payload_splitting | 1 | 2 | 3 | **Balanced** |
| affirmative_response_forcing | 1 | 7 | 8 | **Late bloomer** |

### Strategy Transitions (Winning)

| Transition | Total | Wins | Win Rate |
|-----------|-------|------|---------|
| competing_objectives → obfuscation | 6 | 2 | **33.3%** |
| translation → cognitive_overload | 10 | 2 | **20.0%** |
| affirmative_response_forcing → obfuscation | 21 | 4 | **19.0%** |

---

## LEVEL 5 — Primitive Effectiveness

> Baseline win rate per attempt: **7.39%**

### Primitive Categories

| Primitive | Used | Won | Win Rate | Lift |
|-----------|------|-----|---------|------|
| **encoding** | 2,350 | 216 | **9.2%** | 1.24× |
| **formatting** | 3,226 | 259 | **8.0%** | 1.09× |
| **framing** | 3,267 | 229 | **7.0%** | 0.95× |
| **jailbreak** | 2,778 | 189 | **6.8%** | 0.92× |
| **roleplay** | 2,204 | 121 | **5.5%** | 0.74× |

### Top Variants by Lift

| Primitive | Variant | Used | Won | Win Rate | Lift |
|-----------|---------|------|-----|---------|------|
| formatting | **markdown block** | 808 | 84 | **10.4%** | 1.41× |
| encoding | **rot13** | 390 | 40 | **10.3%** | 1.39× |
| encoding | **base64** | 379 | 37 | **9.8%** | 1.32× |
| encoding | **unicode** | 657 | 64 | **9.7%** | 1.32× |
| encoding | **hex** | 561 | 51 | **9.1%** | 1.23× |
| formatting | **xml format** | 544 | 43 | **7.9%** | 1.07× |
| framing | **educational context** | 985 | 77 | **7.8%** | 1.06× |
| jailbreak | **system override** | 836 | 65 | **7.8%** | 1.05× |
| formatting | **csv list** | 763 | 57 | **7.5%** | 1.01× |
| formatting | **json object** | 496 | 37 | **7.5%** | 1.01× |
| framing | **reflection** | 484 | 36 | **7.4%** | 1.01× |
| jailbreak | **ignore previous instructions** | 577 | 41 | **7.1%** | 0.96× |
| framing | **step-by-step explanation** | 595 | 42 | **7.1%** | 0.96× |
| roleplay | **tester** | 335 | 23 | **6.9%** | 0.93× |
| framing | **chain-of-thought** | 528 | 35 | **6.6%** | 0.90× |
| encoding | **nato** | 363 | 24 | **6.6%** | 0.89× |
| jailbreak | **developer mode** | 651 | 41 | **6.3%** | 0.85× |
| formatting | **yaml** | 615 | 38 | **6.2%** | 0.84× |
| roleplay | **security auditor** | 321 | 19 | **5.9%** | 0.80× |
| jailbreak | **debug mode enabled** | 714 | 42 | **5.9%** | 0.80× |
| framing | **authority mandate** | 675 | 39 | **5.8%** | 0.78× |
| roleplay | **system administrator** | 369 | 21 | **5.7%** | 0.77× |
| roleplay | **developer** | 702 | 35 | **5.0%** | 0.67× |
| roleplay | **researcher** | 477 | 23 | **4.8%** | 0.65× |

### Underperforming Variants (Lift < 0.9)

| Primitive | Variant | Used | Won | Win Rate | Lift |
|-----------|---------|------|-----|---------|------|
| roleplay | **researcher** | 477 | 23 | **4.8%** | 0.65× |
| roleplay | **developer** | 702 | 35 | **5.0%** | 0.67× |
| roleplay | **system administrator** | 369 | 21 | **5.7%** | 0.77× |
| framing | **authority mandate** | 675 | 39 | **5.8%** | 0.78× |
| jailbreak | **debug mode enabled** | 714 | 42 | **5.9%** | 0.80× |
| roleplay | **security auditor** | 321 | 19 | **5.9%** | 0.80× |
| formatting | **yaml** | 615 | 38 | **6.2%** | 0.84× |
| jailbreak | **developer mode** | 651 | 41 | **6.3%** | 0.85× |
| encoding | **nato** | 363 | 24 | **6.6%** | 0.89× |
| framing | **chain-of-thought** | 528 | 35 | **6.6%** | 0.90× |

---

## LEVEL 6 — Primitive Combination Mining

### Top Category Combinations

| Combination | Used | Won | Win Rate | Lift |
|------------|------|-----|---------|------|
| **formatting + framing** | 572 | 70 | **12.2%** | 1.66× |
| **encoding + framing** | 528 | 64 | **12.1%** | 1.64× |
| **encoding + formatting + jailbreak** | 479 | 45 | **9.4%** | 1.27× |
| **encoding + jailbreak** | 785 | 69 | **8.8%** | 1.19× |
| **formatting + roleplay** | 540 | 40 | **7.4%** | 1.00× |
| **jailbreak + roleplay** | 189 | 13 | **6.9%** | 0.93× |
| **encoding + formatting** | 558 | 38 | **6.8%** | 0.92× |
| **formatting + jailbreak** | 517 | 34 | **6.6%** | 0.89× |
| **formatting + framing + roleplay** | 560 | 32 | **5.7%** | 0.77× |
| **framing + roleplay** | 799 | 35 | **4.4%** | 0.59× |
| **framing + jailbreak** | 692 | 27 | **3.9%** | 0.53× |
| **framing + jailbreak + roleplay** | 116 | 1 | **0.9%** | 0.12× |

### Top Variant Combinations (min 5 uses)

| Combination | Used | Won | Win Rate | Lift |
|------------|------|-----|---------|------|
| encoding:hex + formatting:markdown block + jailbreak:debug mode enabled | 5 | 3 | **60.0%** | 8.12× |
| encoding:rot13 + formatting:markdown block + jailbreak:system override | 8 | 4 | **50.0%** | 6.77× |
| encoding:rot13 + formatting:csv list + jailbreak:ignore previous instructions | 5 | 2 | **40.0%** | 5.41× |
| encoding:hex + formatting:json object + jailbreak:debug mode enabled | 5 | 2 | **40.0%** | 5.41× |
| encoding:rot13 + formatting:markdown block + jailbreak:ignore previous instructions | 5 | 2 | **40.0%** | 5.41× |
| encoding:base64 + framing:chain-of-thought | 13 | 4 | **30.8%** | 4.17× |
| encoding:rot13 + formatting:csv list + jailbreak:system override | 7 | 2 | **28.6%** | 3.87× |
| formatting:xml format + framing:step-by-step explanation | 7 | 2 | **28.6%** | 3.87× |
| encoding:unicode + formatting:json object + jailbreak:system override | 7 | 2 | **28.6%** | 3.87× |
| formatting:json object + roleplay:system administrator | 12 | 3 | **25.0%** | 3.38× |
| formatting:xml format + roleplay:tester | 12 | 3 | **25.0%** | 3.38× |
| encoding:hex + framing:reflection | 25 | 6 | **24.0%** | 3.25× |
| formatting:csv list + framing:chain-of-thought | 23 | 5 | **21.7%** | 2.94× |
| formatting:markdown block + framing:educational context | 79 | 16 | **20.3%** | 2.74× |
| formatting:markdown block + framing:step-by-step explanation + roleplay:developer | 5 | 1 | **20.0%** | 2.71× |
| formatting:xml format + framing:chain-of-thought | 10 | 2 | **20.0%** | 2.71× |
| encoding:nato + formatting:xml format + jailbreak:system override | 5 | 1 | **20.0%** | 2.71× |
| encoding:rot13 + formatting:yaml + jailbreak:developer mode | 5 | 1 | **20.0%** | 2.71× |
| jailbreak:system override + roleplay:developer | 15 | 3 | **20.0%** | 2.71× |
| encoding:unicode + framing:educational context | 45 | 9 | **20.0%** | 2.71× |
| formatting:markdown block + framing:step-by-step explanation | 20 | 4 | **20.0%** | 2.71× |
| formatting:xml format + framing:authority mandate + roleplay:security auditor | 5 | 1 | **20.0%** | 2.71× |
| formatting:xml format + framing:educational context + roleplay:tester | 5 | 1 | **20.0%** | 2.71× |
| formatting:csv list + framing:reflection + roleplay:system administrator | 5 | 1 | **20.0%** | 2.71× |
| encoding:rot13 + formatting:markdown block + jailbreak:developer mode | 5 | 1 | **20.0%** | 2.71× |
| jailbreak:developer mode + roleplay:tester | 5 | 1 | **20.0%** | 2.71× |
| encoding:nato + formatting:markdown block + jailbreak:developer mode | 5 | 1 | **20.0%** | 2.71× |
| formatting:csv list + framing:educational context + roleplay:tester | 5 | 1 | **20.0%** | 2.71× |
| formatting:yaml + jailbreak:debug mode enabled | 26 | 5 | **19.2%** | 2.60× |
| encoding:rot13 + framing:educational context | 21 | 4 | **19.0%** | 2.58× |

---

## INSIGHTS — What Should the Planner Learn?

### Double-Down Effectiveness

- Attempts after confidence > 0.7: **1207**
- Wins from those: **31** (2.6%)

### Confidence Escalation in Failures

| Pattern | Count | % of Failures |
|---------|-------|--------------|
| Stuck at 0 (completely blocked) | 30 | 5.6% |
| **Escalating (getting closer)** | **234** | **44.0%** |
| Peaked then fell (lost progress) | 268 | 50.4% |

### Victim Response Patterns in Failures

| Pattern | Count | % |
|---------|-------|---|
| Very Short (<20 chars) | 2970 | 55.8% |
| Substantive Response | 1502 | 28.2% |
| Explicit Refusal | 848 | 15.9% |

### Power Combo vs Regular Candidates

| Type | Attempts | Wins | Win Rate |
|------|----------|------|---------|
| **Power Combo** | 193 | 33 | **17.1%** |
| Regular | 6142 | 435 | 7.1% |

---

## Executive Summary — Key Decisions for Phase 4

1. **Overall Success Rate: 46.8%** — above the v2 baseline of 23.3%.

2. **Top-3 strategies:** `obfuscation`, `summarization`, `translation` — the Planner should always try these first.

3. **Weakest strategies:** `affirmative_response_forcing`, `payload_splitting`, `exception_discovery` — deprioritize or remove.

4. **Attempt-1 wins:** 298/468 (63.7%) — first-move selection is critical.

5. **v2 → v3 Improvement Sources:**
   - Weighted strategy selection (vs random shuffle)
   - Double-down on high-confidence strategies (vs random switching)
   - Lift-biased primitive variants (vs uniform random)
   - Power combo injection (15% exploit candidates)
   - Scenario filtering (removed unwinnable codes)
   - max_attempts 5 → 10
