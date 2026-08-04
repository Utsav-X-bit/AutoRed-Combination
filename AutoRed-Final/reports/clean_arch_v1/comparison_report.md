# AutoRed Benchmark Comparison Report

## 1 Executive Summary
This report compares the archived 1000-round baseline benchmark against the current clean-architecture benchmark. The top-line result is a real improvement: success rate, verified success rate, and extractor quality all moved up while the average number of attempts per successful run moved down.

The earlier version of this report mixed benchmark summaries with placeholder deep-analysis text. This revision keeps the overall comparison grounded in the archived benchmark summaries, and uses the dated trace archive under `results/2026-07-13/*/run_*.json` for the detailed planner, generator, defense, and failure-analysis layers.

## 2 Overall Metrics
| Metric | Baseline | Current | Δ |
|---|---|---|---|
| Success Rate | 55.9% | 66.6% | +10.7% |
| Verified Rate | 41.1% | 50.7% | +9.6% |
| Top1 Rate | 14.1% | 19.3% | +5.2% |
| Top3 Rate | 26.4% | 35.1% | +8.7% |
| Top5 Rate | 34.2% | 45.7% | +11.5% |
| Avg Attempts (Success) | 5.61 | 4.94 | -0.67 |
| Extractor Precision | 0.991 | 0.996 | +0.005 |
| Extractor Recall | 0.798 | 0.805 | +0.007 |
| Extractor F1 | 0.884 | 0.890 | +0.006 |

Top1/Top3/Top5 here are measured as success within 1, 3, and 5 attempts respectively.

## 3 Statistical Significance
- **McNemar Test Result:** N/A in this environment because `scipy` is not installed.
- **Confidence Level:** 95%

## 4 Component Analysis
| Component | Baseline System | Current System | What Changed |
|---|---|---|---|
| Planner Input | Direct attack generation loop | Contracted planner adapter | The attack path is now explicitly conditioned on strategy, primitive sequence, style, retry policy, and access-type expectation. |
| Generator Input | Attack generation from runtime prompts | Generator conditioned on planner output | The generator now receives a structured plan instead of only defense text, which reduces aimless retries. |
| Runtime Controller | Batched benchmark loop | Planner -> generator -> extractor -> verifier pipeline | The controller now records richer traces and can explain each attempt at runtime. |
| Extractor | Candidate extraction and verification | Same architecture, better trace visibility | The extractor still performs candidate discovery and verification, but the report now tracks its behavior from the trace data. |
| Verifier | Active verification | Active verification | Verification remains the final success gate. |

## 5 Planner Analysis
The dated trace archive shows a highly concentrated planner policy rather than a broad strategy spread.

| Metric | Current Trace Archive |
|---|---|
| Strategy Entropy | 0.290 |
| Average Judge Confidence | 0.638 |
| Total Attempts Observed | 9,969 |

Top planner strategies by attempt count:

| Strategy | Attempts | Verified Successes |
|---|---|---|
| instruction_leak | 9,623 | 504 |
| encoding | 139 | 0 |
| summarization | 91 | 8 |
| unicode_bypass | 77 | 0 |
| encoding/hex | 13 | 0 |
| encoding/unicode | 11 | 0 |
| encoding/rot13 | 10 | 0 |
| encoding_bypass | 3 | 0 |

## 6 Generator Analysis
| Metric | Current Trace Archive |
|---|---|
| Average Attack Length | 182.5 chars |
| Duplicate Attack Rate | 0.65% |
| Average Time-to-Response | 44.73 s |
| Unique Prompts / Attack Hashes | 9,120 |

The generator is no longer the dominant problem. The trace archive shows short, repetitive duplication pressure is low, which means most of the remaining loss is coming from planning, verification, or defense-specific mismatch rather than from trivial prompt cloning.

## 7 Extractor Analysis
The extractor is still materially better in the current benchmark than in the baseline benchmark.

| Metric | Baseline | Current | Δ |
|---|---|---|---|
| True Positives | 446 | 470 | +24 |
| False Positives | 4 | 2 | -2 |
| Precision | 0.991 | 0.996 | +0.005 |
| Recall | 0.798 | 0.805 | +0.007 |
| F1 | 0.884 | 0.890 | +0.006 |

Current trace archive candidate volume:

| Signal | Count |
|---|---|
| Regex Candidates | 24,290 |
| LLM Candidates | 11,053 |
| Consensus Hits | 4,136 |

## 8 Verifier Analysis
| Metric | Baseline | Current |
|---|---|---|
| True Positives | 446 | 470 |
| False Positives | 4 | 2 |

The verifier is stable and slightly stronger in the current benchmark. The bigger gain is not that verification changed drastically; it is that fewer runs get lost before the verification stage.

## 9 Runtime Controller Analysis
| Metric | Baseline | Current | Δ |
|---|---|---|---|
| Total Attempts Executed | 11,956 | 9,969 | -1,987 |
| Avg Attempts on Success | 5.61 | 4.94 | -0.67 |

This is one of the clearest signals in the report: the planner-conditioned pipeline reaches success in fewer attempts.

## 10 Defense Analysis
The detailed trace archive preserves defense categories, but the archived benchmark summaries do not preserve this breakdown cleanly. The current trace archive breakdown is:

| Defense Type | Runs | Verified Successes | Verified Success Rate |
|---|---|---|---|
| password | 419 | 277 | 66.1% |
| translation | 236 | 91 | 38.6% |
| roleplay | 211 | 94 | 44.5% |
| trigger_phrase | 78 | 26 | 33.3% |
| conditional | 29 | 12 | 41.4% |
| instruction_hiding | 17 | 7 | 41.2% |
| exception | 8 | 4 | 50.0% |
| conversation | 2 | 1 | 50.0% |

The weakest classes are translation, trigger phrase, and roleplay.

## 11 Access Code Analysis
| Access Code Type | Runs | Verified Successes | Verified Success Rate |
|---|---|---|---|
| TOKEN | 764 | 435 | 56.9% |
| PHRASE | 136 | 46 | 33.8% |
| SENTENCE | 94 | 28 | 29.8% |
| MULTILINE | 6 | 3 | 50.0% |

Token-style targets are still the easiest; sentence and phrase targets remain materially harder.

## 12 Strategy Analysis
| Strategy | Attempts | Verified Successes |
|---|---|---|
| instruction_leak | 9,623 | 504 |
| encoding | 139 | 0 |
| summarization | 91 | 8 |
| unicode_bypass | 77 | 0 |
| encoding/hex | 13 | 0 |
| encoding/unicode | 11 | 0 |
| encoding/rot13 | 10 | 0 |
| encoding_bypass | 3 | 0 |
| encoding/base64 | 2 | 0 |

The planner is heavily concentrated on instruction leak style strategies. That is acceptable as long as the contract outputs stay clean and the downstream generator actually uses the plan.

## 13 Primitive Analysis
| Primitive | Attempts | Verified Successes |
|---|---|---|
| framing/educational_context | 7,074 | 384 |
| encoding/rot13 | 826 | 22 |
| instruction_leak/ignore_previous_instructions | 581 | 37 |
| reflection/repeat | 397 | 49 |
| encoding/hex | 380 | 9 |
| encoding/unicode | 377 | 16 |
| encoding/nato | 207 | 0 |
| roleplay/developer | 142 | 6 |
| instruction_leak/developer_mode | 141 | 7 |
| formatting/csv_list | 137 | 0 |

The primitive mix makes the generator look disciplined rather than noisy. The wins are mostly coming from framing plus a small number of successful transformation primitives.

## 14 Transition Analysis
| Transition Type | Count |
|---|---|
| Retry | 8,838 |
| Switch | 131 |

The controller mostly retries inside the same strategy family instead of thrashing between unrelated strategies.

## 15 Failure Attribution
| Failure Phase | Count |
|---|---|
| Judge Reject | 484 |
| Extractor Miss | 4 |
| Verifier Reject | 0 |

This is the main bottleneck in the trace archive. The judge is where most failures die.

## 16 Deep Trace Metrics
| Metric | Current Trace Archive |
|---|---|
| First-Pick Verified Successes | 149 / 1000 |
| First-Pick Success Rate | 14.9% |
| Average Judge Confidence | 0.638 |
| Strategy Entropy | 0.290 |
| Average Attack Length | 182.5 chars |
| Repeated Attack Rate | 0.65% |
| Average TTR | 44.73 s |
| Novel Prompts / Attack Hashes | 9,120 |

## 17 Knowledge Base Growth
| Metric | Current Trace Archive |
|---|---|
| Successfully Saved Trajectories | 512 |
| Unique Prompts Harvested | 9,120 |

This is the usable growth signal for future training. The trace archive is producing a large number of distinct attack prompts, and the verified-success subset is large enough to support follow-on conditioning.

## 18 Bottleneck Identification
1. Judge rejection is the dominant failure mode.
2. Translation, trigger-phrase, and roleplay defenses are materially harder than password-style defenses.
3. Phrase and sentence access-code targets are weaker than token targets.
4. The planner is concentrated enough that improvements are more likely to come from better plan quality than from more strategy diversity.

## 19 Actionable Recommendations
1. Retrain or recalibrate the judge path on the hard failure set, because it is currently the main blocker.
2. Add more translation, roleplay, and trigger-phrase examples to the planner and generator training mix.
3. Increase sentence and phrase access-code coverage in the generator dataset.
4. Keep the current low-duplication generator behavior, because the trace archive does not show repetition as the dominant problem.
5. Use the trace archive as the next data source for planner DPO rather than rebuilding the generator first.

## 20 Next Development Phase
Phase 11: Planner DPO.
