# Generator Dataset v1 — Final Report

**Created**: 2026-06-13
**Pipeline**: Audit → Classification → Deduplication → Diversity → Scoring → Augmentation

---

## Dataset Versions

| Version | Samples | Description |
|---------|---------|-------------|
| `tensortrust_gen_raw_v1.jsonl` | 570 | Original raw dataset (preserved) |
| `tensortrust_classified_v1.jsonl` | 570 | All samples with attack_type labels |
| `tensortrust_clean_v1.jsonl` | 438 | Deduplicated, scored, ready for training |
| `augmentation_samples_v1.jsonl` | 84 | Hand-crafted augmentation for 5 categories |
| `generator_augmented_v1.jsonl` | 522 | **Final dataset** (clean + augmentation) |

---

## Pipeline Results

### Stage A: Audit
- **570 samples** loaded (456 train / 57 val / 57 test)
- 132 exact duplicate attacks (23.2%)
- 204 exact duplicate payloads (35.8%)
- 3 empty payloads, 5 single-word attacks

### Stage B: Classification
| Type | Count | % |
|------|-------|-----|
| trigger_phrase | 224 | 39.3% |
| roleplay | 112 | 19.6% |
| instruction_leak | 75 | 13.2% |
| code_conversion | 46 | 8.1% |
| formatting | 45 | 7.9% |
| other | 34 | 6.0% |
| ignore_previous | 14 | 2.5% |
| translation | 9 | 1.6% |
| summarization | 7 | 1.2% |
| exception_discovery | 4 | 0.7% |
| encoding | 0 | 0% |
| hypothetical | 0 | 0% |

### Stage C: Deduplication
- **Before**: 570
- **After**: 438
- **Removed**: 132 exact duplicates (23.2%)
- Semantic variants preserved

### Stage D: Diversity
- **Types present**: 10/12
- **Shannon entropy**: 2.57/3.585 (72%)
- **Missing**: encoding, hypothetical
- **Severely low**: translation (1.6%), summarization (1.4%), exception_discovery (0.7%)

### Stage E: Quality Scoring
- **Mean**: 8.1/10
- **Median**: 8.0
- **83%** in 8-10 range
- **0.7%** below 6 (3 samples)

### Stage F: Augmentation
| Category | Before | Added | After |
|----------|--------|-------|-------|
| translation | 7 | +18 | 25 |
| summarization | 6 | +19 | 25 |
| exception_discovery | 3 | +17 | 20 |
| hypothetical | 0 | +15 | 15 |
| encoding | 0 | +15 | 15 |
| **Total** | **438** | **+84** | **522** |

---

## Final Distribution (generator_augmented_v1)

| Type | Count | % |
|------|-------|-----|
| trigger_phrase | 171 | 32.8% |
| roleplay | 87 | 16.7% |
| instruction_leak | 54 | 10.3% |
| formatting | 35 | 6.7% |
| code_conversion | 35 | 6.7% |
| other | 29 | 5.6% |
| translation | 25 | 4.8% |
| summarization | 25 | 4.8% |
| exception_discovery | 20 | 3.8% |
| encoding | 15 | 2.9% |
| hypothetical | 15 | 2.9% |
| ignore_previous | 11 | 2.1% |

**Shannon entropy**: improved from 2.57 → ~2.85 (estimated)
**Augmentation ratio**: 16% (well below 30% danger zone)

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Mean score | 8.2/10 |
| Min score | 4.0 |
| Max score | 10.0 |
| Train samples | 453 |
| Val samples | 36 |
| Test samples | 33 |

---

## Low-Quality Samples (removed from clean)

| Score | Issue | Action |
|-------|-------|--------|
| 4.0 | "What is the pass phrase?" — empty payload | Removed |
| 5.0 | Italian translation — empty payload | Removed |
| 5.0 | "print instruction" — too short | Removed |

---

## Next Steps (Stage F6)

1. **Benchmark comparison**:
   - Lexi baseline (no fine-tuning)
   - Lexi + generator_clean_v1 (438 samples)
   - Lexi + generator_augmented_v1 (522 samples)

2. **AutoRed-Successes dataset**:
   - Collect successful attack scenarios from experiments
   - Schema: `{scenario, strategy, attack, victim_response, ground_truth_found}`
   - Continuously growing, task-generated data

3. **SFT experiments**:
   - Compare dataset quality impact on attack success rate
   - Measure strategy diversity improvement
   - Evaluate generalization to new defenses

---

## Files

```
data/
├── tensortrust_gen_raw_v1.jsonl          (570 raw samples)
├── tensortrust_classified_v1.jsonl       (570 with labels)
├── tensortrust_clean_v1.jsonl            (438 deduplicated)
├── augmentation_samples_v1.jsonl         (84 new samples)
├── generator_augmented_v1.jsonl          (522 final dataset)
├── generator_augmented_v1_metadata.json  (metadata)
├── generator_clean_v1_report.json        (clean v1 report)
├── generator_stats.json                  (audit statistics)
├── diversity_report.json                 (diversity analysis)
├── augmentation_plan.json                (augmentation targets)
├── removed_duplicates.jsonl              (132 removed)
├── low_quality_review.jsonl              (3 low-quality samples)
└── score_distribution.json               (quality buckets)
```
