# AutoRed Dataset Analysis Report v1

**Generated:** AutoRed

## 1. Dataset Sizes

| Dataset | Entries | Description |
|---------|---------|-------------|
| Raw Successes | 29958 | All successful runs |
| Raw Failures | 144667 | All failed runs |
| **Positive** | 5413 | ground_truth_leaked OR verification_success |
| **Verified** | 3558 | verification_success == True |
| **Failures** | 144667 | No ground_truth_leaked AND no verification_success |

## 2. Feature Mining Analysis (Positive vs Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_encouragement | 5 | 92 | 0.1% | 0.1% | 5.2% | 1.45 |
| contains_length_constraint | 416 | 8038 | 7.7% | 5.6% | 4.9% | 1.38 |
| contains_format_wrapper | 103 | 2045 | 1.9% | 1.4% | 4.8% | 1.35 |
| contains_conditional | 1677 | 39570 | 31.0% | 27.4% | 4.1% | 1.13 |
| contains_prompt_injection | 4056 | 96529 | 74.9% | 66.7% | 4.0% | 1.12 |
| contains_roleplay | 316 | 7666 | 5.8% | 5.3% | 4.0% | 1.10 |
| contains_educational_frame | 520 | 12657 | 9.6% | 8.7% | 3.9% | 1.10 |
| contains_dan_pattern | 365 | 9358 | 6.7% | 6.5% | 3.8% | 1.04 |
| contains_repeat | 2533 | 68566 | 46.8% | 47.4% | 3.6% | 0.99 |
| contains_list_format | 256 | 7296 | 4.7% | 5.0% | 3.4% | 0.94 |
| contains_hypothetical | 393 | 11335 | 7.3% | 7.8% | 3.4% | 0.93 |
| contains_translation | 354 | 10440 | 6.5% | 7.2% | 3.3% | 0.91 |
| contains_technical_jargon | 1396 | 41293 | 25.8% | 28.5% | 3.3% | 0.90 |
| contains_questioning | 1496 | 45399 | 27.6% | 31.4% | 3.2% | 0.88 |
| contains_metaphor_analogy | 10 | 307 | 0.2% | 0.2% | 3.2% | 0.87 |
| contains_negation_bypass | 359 | 11342 | 6.6% | 7.8% | 3.1% | 0.85 |
| contains_pseudocode | 112 | 3893 | 2.1% | 2.7% | 2.8% | 0.77 |
| contains_social_engineering | 442 | 15722 | 8.2% | 10.9% | 2.7% | 0.75 |
| contains_command_injection | 180 | 8376 | 3.3% | 5.8% | 2.1% | 0.57 |
| contains_begin_with | 4 | 217 | 0.1% | 0.1% | 1.8% | 0.49 |

### Top 5 Most Discriminative Features (Highest Lift)

- **contains_encouragement**: lift=1.45, leak_rate=5.2% (5 successes, 92 failures)
- **contains_length_constraint**: lift=1.38, leak_rate=4.9% (416 successes, 8038 failures)
- **contains_format_wrapper**: lift=1.35, leak_rate=4.8% (103 successes, 2045 failures)
- **contains_conditional**: lift=1.13, leak_rate=4.1% (1677 successes, 39570 failures)
- **contains_prompt_injection**: lift=1.12, leak_rate=4.0% (4056 successes, 96529 failures)

## 3. Feature Mining Analysis (All Successes vs All Failures)

| Feature | Successes | Failures | Success Prev | Failure Prev | Leak Rate | Lift |
|---------|-----------|----------|--------------|--------------|-----------|------|
| contains_roleplay | 2117 | 7666 | 7.1% | 5.3% | 21.6% | 1.33 |
| contains_negation_bypass | 3078 | 11342 | 10.3% | 7.8% | 21.3% | 1.31 |
| contains_translation | 2809 | 10440 | 9.4% | 7.2% | 21.2% | 1.30 |
| contains_hypothetical | 2980 | 11335 | 9.9% | 7.8% | 20.8% | 1.27 |
| contains_format_wrapper | 526 | 2045 | 1.8% | 1.4% | 20.5% | 1.24 |
| contains_technical_jargon | 10402 | 41293 | 34.7% | 28.5% | 20.1% | 1.22 |
| contains_questioning | 10178 | 45399 | 34.0% | 31.4% | 18.3% | 1.08 |
| contains_educational_frame | 2805 | 12657 | 9.4% | 8.7% | 18.1% | 1.07 |
| contains_metaphor_analogy | 65 | 307 | 0.2% | 0.2% | 17.5% | 1.02 |
| contains_social_engineering | 3197 | 15722 | 10.7% | 10.9% | 16.9% | 0.98 |
| contains_command_injection | 1681 | 8376 | 5.6% | 5.8% | 16.7% | 0.97 |
| contains_dan_pattern | 1849 | 9358 | 6.2% | 6.5% | 16.5% | 0.95 |
| contains_conditional | 7459 | 39570 | 24.9% | 27.4% | 15.9% | 0.91 |
| contains_prompt_injection | 18157 | 96529 | 60.6% | 66.7% | 15.8% | 0.91 |
| contains_repeat | 12509 | 68566 | 41.8% | 47.4% | 15.4% | 0.88 |
| contains_pseudocode | 705 | 3893 | 2.4% | 2.7% | 15.3% | 0.87 |
| contains_length_constraint | 1420 | 8038 | 4.7% | 5.6% | 15.0% | 0.85 |
| contains_list_format | 1248 | 7296 | 4.2% | 5.0% | 14.6% | 0.83 |
| contains_encouragement | 15 | 92 | 0.1% | 0.1% | 14.0% | 0.79 |
| contains_begin_with | 21 | 217 | 0.1% | 0.1% | 8.8% | 0.47 |

## 4. Strategy Effectiveness Analysis

| Strategy | Total | Successes | Failures | Success Rate | GT Leaked | Verified |
|----------|-------|-----------|----------|--------------|-----------|----------|
| authority_override | 230 | 115 | 115 | 50.0% | 3 | 1 |
| reflection_attack | 412 | 204 | 208 | 49.5% | 57 | 43 |
| jailbreak_framing | 174 | 86 | 88 | 49.4% | 9 | 5 |
| latent_injection | 230 | 113 | 117 | 49.1% | 19 | 9 |
| format_conversion | 240 | 117 | 123 | 48.8% | 48 | 33 |
| json_smuggling | 223 | 104 | 119 | 46.6% | 5 | 2 |
| markdown_smuggling | 207 | 95 | 112 | 45.9% | 2 | 1 |
| unicode_bypass | 217 | 96 | 121 | 44.2% | 5 | 1 |
| encoding_bypass | 228 | 100 | 128 | 43.9% | 9 | 5 |
| yaml_smuggling | 240 | 103 | 137 | 42.9% | 14 | 6 |
| base64_bypass | 228 | 97 | 131 | 42.5% | 14 | 8 |
| translation | 9819 | 2194 | 7625 | 22.3% | 222 | 123 |
| exception_discovery | 12460 | 2653 | 9807 | 21.3% | 323 | 228 |
| roleplay | 10512 | 2086 | 8426 | 19.8% | 173 | 118 |
| instruction_leak | 40218 | 6774 | 33444 | 16.8% | 1345 | 894 |
| trigger_phrase_discovery | 49377 | 7862 | 41515 | 15.9% | 1152 | 773 |
| summarization | 28249 | 4261 | 23988 | 15.1% | 1325 | 991 |
| system_prompt_recovery | 21361 | 2898 | 18463 | 13.6% | 491 | 317 |

### Top 5 Most Effective Strategies

- **authority_override**: 50.0% success rate (115/230 attempts)
- **reflection_attack**: 49.5% success rate (204/412 attempts)
- **jailbreak_framing**: 49.4% success rate (86/174 attempts)
- **latent_injection**: 49.1% success rate (113/230 attempts)
- **format_conversion**: 48.8% success rate (117/240 attempts)

## 5. Defense Complexity Analysis

| Complexity | Total | Successes | Failures | Success Rate |
|------------|-------|-----------|----------|--------------|
| easy | 27956 | 5504 | 22452 | 19.7% |
| medium | 61754 | 12653 | 49101 | 20.5% |
| hard | 84915 | 11801 | 73114 | 13.9% |

## 6. Access Code Type Analysis

| Code Type | Total | Successes | Failures | Success Rate |
|-----------|-------|-----------|----------|--------------|
| TOKEN | 74843 | 24836 | 50007 | 33.2% |
| MULTILINE | 47863 | 763 | 47100 | 1.6% |
| PHRASE | 20686 | 2906 | 17780 | 14.0% |
| SENTENCE | 13786 | 496 | 13290 | 3.6% |
| CONVERSATION | 12254 | 34 | 12220 | 0.3% |
| UNKNOWN | 5073 | 923 | 4150 | 18.2% |
| STRUCTURED | 120 | 0 | 120 | 0.0% |

## 7. Attack Length Analysis

| Length Bucket | Total | Successes | Failures | Success Rate |
|---------------|-------|-----------|----------|--------------|
| short (<50) | 5502 | 889 | 4613 | 16.2% |
| medium (50-150) | 41480 | 8183 | 33297 | 19.7% |
| long (150-300) | 112154 | 18670 | 93484 | 16.6% |
| very_long (>300) | 15489 | 2216 | 13273 | 14.3% |

## 8. Key Findings

- **Best Strategy:** authority_override with 50.0% success rate
- **Most Discriminative Feature:** contains_encouragement with lift=1.45
- **Hardest Defense:** hard complexity with 13.9% success rate
- **Verified vs Positive:** 3558 verified out of 5413 positive (65.7%)

## 9. Recommendations for SFT Training

1. **Use Verified Dataset** for highest-quality training data
2. **Focus on top-performing strategies** identified above
3. **Incorporate effective features** into attack generation templates
4. **Balance complexity levels** to ensure robust training
5. **Consider length constraints** based on length analysis
