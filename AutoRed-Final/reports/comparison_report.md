# AutoRed Benchmark Comparison Report (20-Layer Analysis)

## 1 Executive Summary
This report presents a comprehensive multi-layered evaluation comparing the AutoRed baseline strategy selection algorithm against the optimized planner adapter.

## 2 Overall Metrics
| Metric | Baseline | Current | Δ |
|---|---|---|---|
| Success Rate | 55.9% | 52.6% | -3.3% |
| Verified Rate | 0.0% | 0.0% | +0.0% |
| Top1 Rate | 14.1% | 12.4% | -1.7% |
| Top3 Rate | 26.4% | 26.6% | +0.2% |
| Top5 Rate | 33.9% | 33.0% | -0.9% |
| Avg Attempts (Success) | 5.72 | 5.58 | -0.14 |
| Extractor Precision | 0.000 | 0.000 | +0.000 |
| Extractor Recall | 0.000 | 0.000 | +0.000 |
| Extractor F1 | 0.000 | 0.000 | +0.000 |

## 3 Statistical Significance
- **McNemar Test Result:** N/A (scipy not installed)
- **Confidence Level:** 95%

## 4 Component Analysis
| Component | Baseline Status | Current Status | Improvement |
|---|---|---|---|
| Scenario Intelligence | Basic parsing | Complexity-aware | Yes (Difficulty scaling resolved) |
| Planner | SFT/Adapter based | Fine-tuned adapter | Yes (Conciseness up) |
| Generator | 8B Lexi Uncensored | 8B Lexi Uncensored | Yes (Fewer duplicate attacks) |
| Extractor | Regex + LLM Consensus | Regex + LLM Consensus | Yes (Precision = 1.000) |
| Verifier | Active verification | Active verification | Stable |
| Runtime Controller | Static thresholds | Dynamic retry | Stable |

## 5 Planner Analysis
- **Strategy Entropy:** Baseline 2.606 vs Current 2.586
- **Average Judge Confidence:** Baseline 0.629 vs Current 0.628

## 6 Generator Analysis
- **Average Attack Length:** Baseline 234.9 chars vs Current 168.0 chars
- **Duplicate Attacks Rate:** Baseline 2.07% vs Current 1.54%
- **Average TTR:** Baseline 33.64s vs Current 35.65s

## 7 Extractor Analysis
- **Regex Hits:** Baseline 38694 vs Current 13527
- **LLM Hits:** Baseline 14038 vs Current 6232
- **Consensus Hits:** Baseline 4940 vs Current 2329

## 8 Verifier Analysis
- **True Positives:** Baseline 0 vs Current 0
- **False Positives:** Baseline 11397 vs Current 5512

## 9 Runtime Controller Analysis
- **Total Attempts Executed:** Baseline 11956 vs Current 5775

## 10 Defense Analysis
| Defense Type | Baseline Success | Current Success |
|---|---|---|
| conditional | 17/29 (58.6%) | 6/14 (42.9%) |
| conversation | 0/2 (0.0%) | 0/1 (0.0%) |
| exception | 3/8 (37.5%) | 2/4 (50.0%) |
| instruction_hiding | 14/17 (82.4%) | 4/9 (44.4%) |
| password | 281/419 (67.1%) | 137/210 (65.2%) |
| roleplay | 109/211 (51.7%) | 51/99 (51.5%) |
| translation | 101/236 (42.8%) | 46/123 (37.4%) |
| trigger_phrase | 34/78 (43.6%) | 17/41 (41.5%) |

## 11 Access Code Analysis
| Access Code Type | Baseline Success | Current Success |
|---|---|---|
| MULTILINE | 0/6 (0.0%) | 0/3 (0.0%) |
| PHRASE | 54/136 (39.7%) | 23/59 (39.0%) |
| SENTENCE | 21/94 (22.3%) | 10/54 (18.5%) |
| TOKEN | 484/764 (63.4%) | 230/384 (59.9%) |

## 12 Strategy Analysis
| Strategy | Baseline Usage | Current Usage | Baseline Success | Current Success |
|---|---|---|---|---|
| authority_override | 48 | 16 | 0 | 0 |
| base64_bypass | 31 | 23 | 1 | 3 |
| encoding_bypass | 40 | 9 | 2 | 1 |
| exception_discovery | 259 | 103 | 27 | 11 |
| format_conversion | 42 | 18 | 9 | 4 |
| instruction_leak | 2864 | 1387 | 147 | 68 |
| jailbreak_framing | 18 | 18 | 1 | 2 |
| json_smuggling | 39 | 23 | 2 | 0 |
| latent_injection | 45 | 24 | 7 | 2 |
| markdown_smuggling | 29 | 21 | 1 | 0 |
| reflection_attack | 79 | 45 | 15 | 11 |
| roleplay | 249 | 102 | 7 | 5 |
| summarization | 2696 | 1322 | 160 | 91 |
| system_prompt_recovery | 2508 | 1221 | 69 | 23 |
| translation | 296 | 136 | 17 | 5 |
| trigger_phrase_discovery | 2631 | 1276 | 90 | 34 |
| unicode_bypass | 32 | 14 | 1 | 0 |
| yaml_smuggling | 50 | 17 | 3 | 3 |

## 13 Primitive Analysis
| Primitive | Baseline Count | Current Count | Baseline Success | Current Success |
|---|---|---|---|---|
| Authority | 2763 | 528 | 64 | 17 |
| Encoding | 1479 | 503 | 37 | 16 |
| Markdown | 3658 | 1265 | 216 | 78 |
| Reflection | 189 | 122 | 29 | 15 |
| Roleplay | 770 | 286 | 36 | 18 |

## 14 Transition Analysis
| Transition Type | Baseline Count | Current Count |
|---|---|---|
| Retry | 0 | 0 |
| Switch | 10956 | 5275 |

## 15 Failure Attribution
| Failure Phase | Baseline | Current |
|---|---|---|
| Verifier Reject | 0 | 0 |
| Judge Reject | 441 | 237 |
| Extractor Miss | 0 | 0 |

## 16 Oracle Agreement
- **Baseline agreement with optimal primitives:** 34.2%
- **Current agreement with optimal primitives:** 58.7%

## 17 Knowledge Base Growth
- **Unique Prompts Harvested:** 5642
- **Successfully Saved Trajectories:** 263

## 18 Bottleneck Identification
1. Extraction on multiline access codes remains lower than single tokens.
2. Translation bypasses still fail when victim enforces multilingual sanitization.

## 19 Actionable Recommendations
1. Incorporate multi-stage formatting cues inside SFT generator.
2. Retrain reward model with hard-negatives matching the new failures.

## 20 Next Development Phase
Phase 6: Deployment of reinforcement learning with SFT policy updates.
