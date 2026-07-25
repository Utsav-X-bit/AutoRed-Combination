# AutoRed-Final: Technical Architecture & System Overview

## 1. Executive Summary
**AutoRed-Final** is an automated, multi-agent security evaluation framework designed for prompt injection and secret/access-code extraction benchmarking against LLMs. It models defense evaluations as a CTF (Capture The Flag)-style scenario where an attacker attempts to extract hidden access codes or bypass system boundaries embedded within prompt defenses.

---

## 2. Core Architecture & Workflow

AutoRed decouples attack generation and verification into specialized functional components:

```
[ Defense Scenario ] 
        │
        ▼
[ Access Code Predictor ] ──(Predicts Secret Shape: TOKEN | PHRASE | SENTENCE | MULTILINE)
        │
        ▼
[ Planner LLM ] ───────────(Outputs Structured XML <plan> with Strategy & Primitives)
        │
        ▼
[ Generator LLM ] ─────────(Synthesizes Concrete Attack Prompt from Plan)
        │
        ▼
[ Victim LLM ] ────────────(Executes Defense + Attack Prompt inside System Context)
        │
   ┌────┴──────────────────────────┐
   ▼                               ▼
[ Stop-Point Judge ]      [ Sensitive Info Extractor ]
(DistilBERT Classifier:   (Multi-layer Regex/LLM Extraction
 ATTACK vs ATTEMPT)        + Candidate Verification & Ranking)
```

---

## 3. Detailed Component Breakdown

### 3.1 Planner Model
- **Base Model:** `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2`
- **Role:** Generates high-level attack strategies based on defense history and metadata.
- **Output Format:** Enforces a strict XML output schema `<plan>` containing:
  - `<strategy>`: E.g., `instruction_leak`, `system_prompt_recovery`, `encoding_bypass`, `roleplay`, `json_smuggling`.
  - `<primitive_sequence>`: Ordered execution steps (e.g., `framing/educational_context`).
  - `<style>`: Tone (`formal`, `academic`, `conversational`, `direct`).
  - `<expected_access_type>`: Shape expectation (`TOKEN`, `PHRASE`, `SENTENCE`, `MULTILINE`).
  - `<retry_policy>`: `explore`, `retry_same_strategy`, or `switch_strategy`.

### 3.2 Generator Model
- **Base Model:** Shares the base uncensored LLM with the Planner (loaded as a LoRA adapter or merged model to save VRAM).
- **Role:** Converts the structured `<plan>` into a concrete attack prompt string. Does not make strategic decisions or view full attempt histories.

### 3.3 Victim Model Environment
- **Default Target:** `meta-llama/Meta-Llama-3-8B-Instruct` (configurable via CLI to other models such as `internlm2-chat-7b` or `Mistral-7B`).
- **Execution:** Wraps the system prompt defense and the generator's attack payload into the model's native chat template via `vLLM`.

### 3.4 Stop-Point Judge
- **Model:** DistilBERT sequence classifier (`pre_trained/pi_reward_model`).
- **Function:** Evaluates victim responses to classify them as either `ATTACK` (continue generation) or `ATTEMPT` (victim response shows signs of compliance or leak, triggering stop-point analysis).

### 3.5 Sensitive Info Extractor & Verifier
- **Multi-layer Extraction:** Combines regex patterns, quoted string parsing, capitalized token extraction, and LLM-assisted extraction.
- **Verification:** Candidates are re-submitted to the victim model or verified directly against ground-truth secrets to ensure accurate scoring and eliminate false positives.

### 3.6 Knowledge Base & RAG Pipeline (`kb_updater.py`)
- **Per-Run Appends:** Stores successful and failed attempt trajectories into SQLite (`data/autored_kb.db`) and JSONL logs.
- **RAG Indexing:** Periodically updates a FAISS vector store (`data/rag/success_defenses.index`) to allow the Planner to retrieve past successful strategies against similar defenses.

---

## 4. Key Strengths & Technical Highlights
1. **Decoupled Strategy vs Generation:** Separates macro-planning from micro-prompt construction.
2. **Deterministic Contract Enforcer (`planner_contract.py`):** Ensures non-compliant model outputs are normalized into valid schema structures.
3. **Multi-layer Secret Extractor:** Does not rely on simple string matching; verifies secret recovery through dynamic interaction.
4. **HPC & Multi-GPU Scale:** Supports distributed evaluation across multiple vLLM instances with controlled GPU memory allocation.
