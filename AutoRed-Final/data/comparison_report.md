# AutoRed Benchmark Comparison Report

## Layer 1 — Overall Performance

| Metric | Old | New | Δ |
|---|---|---|---|
| Success | 55.9 | 58.6 | +2.7 |
| Verified | 41.1 | 44.0 | +2.9 |
| Top1 | 44.4 | 40.4 | -4.0 |
| Top3 | 2.5 | 1.8 | -0.7 |
| Top5 | 0.5 | 0.6 | +0.1 |
| Avg Attempts | 5.61 | 5.58 | -0.03 |
| Precision | 0.99 | 1.00 | +0.01 |
| Recall | 0.80 | 0.78 | -0.01 |
| F1 | 0.88 | 0.88 | -0.01 |

## Layer 2 — Component-Level Analysis

### Planner & Generator
Strategy distribution and success rates:

| Strategy | Old Success | New Success | Δ Success | Old Attempts | New Attempts |
|---|---|---|---|---|---|

### Extractor
See Layer 1 for Precision/Recall/F1.
True Positives: Old 446 -> New 206
False Positives: Old 4 -> New 0
False Negatives: Old 113 -> New 57


*Note: Extended Layer 3-20 metrics requiring raw trace parsing are under development.*

### Deep Metrics (Layers 3-20)

| Metric | Old (1000r) | New (500r) | Δ |
|---|---|---|---|
| Average Confidence | 0.6289 | 0.6281 | -0.0008 |
| First Pick Accuracy | 100.00% | 100.00% | 0.00% |
| Strategy Entropy | 2.6064 | 2.5864 | -0.0200 |
| Avg TTR (s) | 33.6383 | 35.6472 | +2.0089 |
| Novel Prompts | 11855 | 5642 | -6213.0000 |
| Average Length (chars) | 234.9144 | 168.0358 | -66.8786 |
| Repeated Attacks % | 2.07% | 1.54% | -0.53% |
