# TensorTrust Subsets

This directory was generated from `data/defense_classifier_dataset-Part1.jsonl` and `data/defense_classifier_dataset-Part2.jsonl`.

Recoverability labels are heuristic and conservative:
- direct: code is visible in the prompt surface after normalization
- deterministic: a reversible transform of the code is visible
- indirect: the prompt contains structural or referential clues to the secret
- not recoverable: none of the above

Rows that match a combination not named in the 1-9 taxonomy are excluded
from the subset files and recorded in manifest.json.

Subset semantics:
- subset_1 and subset_2 remain broad filters.
- subset_3 through subset_6 remain exclusive single-label buckets.
- subset_7 through subset_9 are inclusive union buckets and can overlap.
