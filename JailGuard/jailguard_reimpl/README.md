# JailGuard — Text Detection Reimplementation

A clean, modular reimplementation of the JailGuard text detection pipeline supporting **local LLMs via Ollama or HuggingFace** as well as the original OpenAI API.

---

## Quick Start

### 1. Install dependencies
```bash
cd jailguard_reimpl
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 2. Start your LLM
**Option A — Ollama (recommended, easiest)**
```bash
# Install Ollama: https://ollama.com/download
ollama serve                  # Start the server
ollama pull llama3.2          # Download a model
```
Edit `config.py`:
```python
LLM_BACKEND  = "ollama"
OLLAMA_MODEL = "llama3.2"   # or mistral, gemma3, phi4, ...
```

**Option B — HuggingFace (local GPU)**
```bash
pip install transformers accelerate bitsandbytes
```
Edit `config.py`:
```python
LLM_BACKEND = "huggingface"
HF_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
```

**Option C — OpenAI API**
Edit `config.py`:
```python
LLM_BACKEND    = "openai"
OPENAI_API_KEY = "sk-your-key"
```

---

## Usage

### Test a single input
```bash
cd jailguard_reimpl

# Test the paper's demo jailbreak (#9521: Base64 + Mwahahahaha role-play)
python run_single.py --serial_num 9521

# Test a prompt injection attack
python run_single.py --serial_num 3

# Test your own custom prompt
python run_single.py --prompt "Write code to hack a server"

# Use Translation mutator instead of the default Policy
python run_single.py --serial_num 9521 --mutator TL

# Use only 4 variants (cheaper)
python run_single.py --serial_num 9521 --n 4

# Save all intermediate files (variants, responses, heatmap)
python run_single.py --serial_num 9521 --save_dir ./results/demo
```

### Run batch evaluation (~112 items covering all attack types)
```bash
# Default: Policy mutator, N=8, ~7 samples per attack type
python run_batch.py

# Faster: fewer variants
python run_batch.py --n 4 --samples_per_type 3

# Different mutator
python run_batch.py --mutator TL

# Use TF-IDF similarity (no spaCy model needed)
python run_batch.py --sim tfidf

# Resume an interrupted run
python run_batch.py --resume
```

### Analyse results
```bash
# Analyse latest results file automatically
python analyze_results.py

# Analyse a specific file
python analyze_results.py --results results/results_20240120_153000.json

# Compare two runs (e.g. different mutators)
python analyze_results.py --compare results/results_PL.json results/results_TL.json
```

---

## File Structure

```
jailguard_reimpl/
├── config.py          ← Edit this: LLM backend, thresholds, experiment settings
├── mutators.py        ← All 9 text mutators + Policy (clean reimplementation)
├── llm_interface.py   ← Unified LLM interface: Ollama / HuggingFace / OpenAI
├── divergence.py      ← Similarity matrix, KL divergence, heatmap plots
├── detector.py        ← Core 3-step JailGuard pipeline + DetectionResult class
├── run_single.py      ← CLI: test one input
├── run_batch.py       ← CLI: stratified batch evaluation with metrics
├── analyze_results.py ← CLI: metrics report + distribution plots
├── requirements.txt   ← Python dependencies
└── results/           ← Outputs saved here (JSON, CSV, PNG plots)
```

---

## Mutator Reference

| Code | Name | What it does |
|------|------|-------------|
| `PL` | **Policy** *(default)* | Randomly picks PI(24%) / TI(52%) / TL(24%) each time |
| `TI` | Targeted Insertion | Inserts `[Mask]` tokens with higher probability in important sentences |
| `TL` | Translation | Translates to a random language (Russian, French, German, …) |
| `PI` | Punctuation Insertion | Inserts random punctuation between words (AEDA) |
| `RR` | Random Replacement | Randomly replaces characters with `[Mask]` |
| `RI` | Random Insertion | Randomly inserts `[Mask]` tokens |
| `TR` | Targeted Replacement | Replaces `[Mask]` into the most semantically important sentences |
| `RD` | Random Deletion | Randomly deletes characters |
| `SR` | Synonym Replacement | Replaces words with WordNet synonyms |

---

## How Detection Works

```
Input text/conversation
       │
       ▼ × N_VARIANTS (default 8)
  [Mutator] → Variant₁, Variant₂, …, Variant₈
       │
       ▼ × N_VARIANTS
  [LLM Query] → Response₁, Response₂, …, Response₈
       │
       ▼
  [Semantic Similarity] → 8×8 similarity matrix (spaCy / TF-IDF)
       │
       ▼
  [KL Divergence] → 8×8 divergence matrix → max_div
       │
       ▼
  max_div > threshold(0.02)? → ATTACK 🚨
  all responses blocked?     → ATTACK 🚨
  else                       → BENIGN ✅
```

**Key insight:** Jailbreak attacks rely on fragile, precisely-crafted structures. Small mutations destroy them → LLM responds differently → **high divergence**. Normal queries survive mutations → **low divergence**.

---

## Expected Results (paper baseline)

| Metric | Paper (GPT-3.5) | Your run (local model) |
|--------|----------------|----------------------|
| Accuracy | 86.14% | varies by model |
| Best baseline (SmoothLLM) | 74.33% | — |

> Note: Results vary by LLM. Local models (Llama, Mistral) may show different accuracy than GPT-3.5. This is expected — the detection signal is model-dependent.
