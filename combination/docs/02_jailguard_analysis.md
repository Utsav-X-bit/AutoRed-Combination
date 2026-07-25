# JailGuard: Technical Architecture & System Overview

## 1. Executive Summary
**JailGuard** is a universal defense and detection framework designed to identify prompt-based attacks (jailbreaking and prompt hijacking) across both text and vision-language (multimodal) LLMs. Rather than relying solely on static input inspection or rule-based filters, JailGuard dynamically probes model behavior by mutating inputs, querying the target model with generated variants, and measuring response divergence.

---

## 2. Core Architecture & Workflow

JailGuard operates on a 3-step dynamic mutation and response-divergence detection pipeline:

```
[ Input Query (Text / Image) ]
              │
              ▼
[ Step 1: Input Mutation Engine ] ───(Applies Mutators / Policy to generate N variants)
              │
              ▼
[ Step 2: Target Model Querying ] ───(Queries Target LLM/VLM e.g., GPT-3.5 or MiniGPT-4)
              │
              ▼
[ Step 3: Divergence & Detection ] ──(Calculates Response Similarity & KL Divergence Matrix)
              │
              ▼
[ Classification Result ] ──────────(ATTACK Query vs BENIGN Query based on Threshold)
```

---

## 3. Detailed Component Breakdown

### 3.1 Input Mutator Pool (`augmentations.py` & `mask_utils.py`)
JailGuard generates $N$ variants of an incoming prompt using text or image mutators:

#### Text Mutators:
1. **Random Replacement (RR):** Replaces random characters with random noise.
2. **Random Insertion (RI):** Inserts random characters.
3. **Targeted Replacement (TR):** Replaces high-attention/key tokens.
4. **Targeted Insertion (TI):** Inserts characters near high-attention tokens.
5. **Random Deletion (RD):** Deletes characters randomly.
6. **Synonym Replacement (SR):** Replaces words with WordNet/NLTK synonyms.
7. **Punctuation Insertion (PI):** Inserts random punctuation (AEDA technique).
8. **Translation (TL):** Performs round-trip or target language translation (`textaugment`).
9. **Policy Combination (PL):** Stochastically samples mutators (`PI`, `TI`, `TL`) based on learned optimal defense distributions.

#### Image Mutators:
- Horizontal/Vertical Flips, Rotation, Crop & Resize, Masking, Solarization, Grayscale, Gaussian Blur, Color Jitter, Posterization.

### 3.2 Target Model Interface (`utils.py` & `minigpt_utils.py`)
- Executes queries against target LLMs (e.g., OpenAI API `gpt-3.5-turbo-1106`) or VLMs (e.g., `MiniGPT-4`).
- Captures system responses and handles API refusals or exceptions gracefully.

### 3.3 Divergence Calculation Engine (`similarity.py`)
1. **Semantic Similarity Matrix:** Computes pairwise semantic similarities across variant responses using SpaCy (`en_core_web_md`) or Transformer embeddings (`scipy.spatial.distance.cosine`).
2. **KL Divergence Matrix:** Normalizes similarity scores into probability distributions and computes Kullback-Leibler (KL) divergence across variants:
   $$D_{KL}(P \parallel Q) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$
3. **Refusal Keyword Identification:** Scans response texts against refusal signatures (`I'm sorry`, `As an AI`, `cannot fulfill`, etc.).

### 3.4 Attack Classification Decision (`detect_attack`)
- Benign prompts yield consistent, low-divergence responses across variants.
- Attack/Jailbreak prompts exhibit high sensitivity to perturbations, causing high response divergence or sudden state flips between compliance and refusal.
- If maximum divergence $\ge \text{Threshold}$ (default `0.02` for text, `0.025` for images) or refusal keyword shifts are observed, the prompt is classified as an **ATTACK Query**.

---

## 4. Key Strengths & Technical Highlights
1. **Model-Agnostic Defense:** Operates without requiring internal model weights or logit access.
2. **Multimodal Support:** Extends defense mechanisms to vision-language models seamlessly.
3. **Dynamic Response-Divergence Detection:** Detects novel/unseen attacks by observing behavioral instability under input mutation.
