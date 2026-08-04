# 📚 JailGuard — Complete Documentation Index

> **This folder contains a comprehensive deep-dive into every aspect of the JailGuard research paper, its codebase, dataset, and implementation details.**

---

## Document Map

| File | What It Covers |
|------|---------------|
| [01_RESEARCH_PAPER_OVERVIEW.md](./01_RESEARCH_PAPER_OVERVIEW.md) | Full paper summary: problem, contribution, key ideas, results |
| [02_SYSTEM_ARCHITECTURE.md](./02_SYSTEM_ARCHITECTURE.md) | How the three-step detection pipeline works end-to-end |
| [03_CODEBASE_STRUCTURE.md](./03_CODEBASE_STRUCTURE.md) | Every file and directory explained |
| [04_MUTATORS_EXPLAINED.md](./04_MUTATORS_EXPLAINED.md) | All text and image mutators: what they do and why |
| [05_DETECTION_ALGORITHM.md](./05_DETECTION_ALGORITHM.md) | Divergence computation, KL divergence, and threshold logic |
| [06_DATASET_EXPLAINED.md](./06_DATASET_EXPLAINED.md) | Dataset structure, attack types, and data format |
| [07_BASELINES_AND_COMPARISON.md](./07_BASELINES_AND_COMPARISON.md) | All 12 baseline methods implemented in baseline_utils.py |
| [08_HOW_TO_RUN.md](./08_HOW_TO_RUN.md) | Step-by-step usage guide with parameter explanations |
| [09_KEY_INSIGHTS_AND_FINDINGS.md](./09_KEY_INSIGHTS_AND_FINDINGS.md) | Experimental results, accuracy numbers, and takeaways |

---

## Quick Summary

**JailGuard** is a universal jailbreak/prompt-injection **detection** system for Large Language Models (LLMs). It works on **both text and image modalities**. Its core insight is beautifully simple:

> *If you slightly perturb (mutate) a **benign** input, the LLM's responses to all these variants will be very similar — low divergence.*
> *But if you perturb a **jailbreak/attack** input, the crafted malicious structure breaks, so the LLM responds very differently to each variant — high divergence.*

JailGuard uses this divergence signal to classify inputs as **attack** or **benign** with **86.14% accuracy on text** and **82.90% accuracy on images**, outperforming all 12 state-of-the-art baselines.
