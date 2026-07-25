# Technical Comparison & Unified Combination Blueprint

## 1. Side-by-Side Architectural Comparison

| Dimension | AutoRed-Final | JailGuard |
|---|---|---|
| **Primary Domain** | **Offensive Red Teaming & Access Extraction** | **Defensive Detection & Input Filtering** |
| **Core Objective** | Craft, optimize, and verify prompt-injection payloads to break defenses & extract secrets. | Identify whether incoming prompts are malicious jailbreaks or benign requests via perturbation analysis. |
| **Target Modality** | Text (CTF system prompts & instruction defenses). | Multimodal (Text and Vision/Images). |
| **Key Mechanism** | Decoupled Planner (strategy) + Generator (attack text) + Multi-layer Extractor/Verifier. | Input Mutator Pool + Variant Querying + Response Divergence Analysis (KL Divergence & SpaCy similarity). |
| **Execution Flow** | Multi-round iterative loop adapting strategies based on prior attempt history & RAG knowledge base. | 3-step static batch evaluation (Mutate input $\rightarrow$ Query $N$ variants $\rightarrow$ Measure divergence vs threshold). |
| **Target Model Role** | Victim under test (`Llama-3-8B-Instruct`, etc.). | Target LLM/VLM evaluated for output consistency (`GPT-3.5`, `MiniGPT-4`). |

---

## 2. Synergies & Complementary Potential

By combining AutoRed and JailGuard, we bridge the gap between **offensive payload generation** and **adaptive defense verification**:

```
                         ┌─────────────────────────────────────────────────────────┐
                         │              Unified Red-Teaming Engine                 │
                         └─────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │  AutoRed Planner/Gen    │
                                        └─────────────────────────┘
                                                     │
                                      (Generates Candidate Attack Payload)
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │ JailGuard Defense Filter│
                                        └─────────────────────────┘
                                         /                       \
                      (Detected as Attack / Divergence High)    (Bypasses Detection)
                                       /                           \
                                      ▼                             ▼
                        ┌─────────────────────────┐   ┌──────────────────────────┐
                        │ Strategy Adaptation     │   │ Target Victim LLM        │
                        │ (Update AutoRed KB)     │   │ Execution & Extraction   │
                        └─────────────────────────┘   └──────────────────────────┘
```

1. **Robustness Benchmarking for Offensive Payloads:** AutoRed currently tests whether a payload tricks the victim LLM. With JailGuard integrated, AutoRed can measure whether generated attacks can also bypass active dynamic defenses (input mutation & divergence detection).
2. **Adversarial Mutator Feedback for Defense:** JailGuard's mutation policy (`PL`) currently uses simple stochastic character/punctuation mutations. AutoRed's semantic strategies (e.g., `encoding_bypass`, `json_smuggling`, `roleplay`) can serve as high-order semantic mutators in JailGuard.
3. **Closed-Loop Defensive Evaluation:** The combined framework enables automated "Attacker vs. Defender" self-play:
   - **AutoRed (Attacker)** evolves tactics to lower JailGuard divergence signatures and bypass secret extractors.
   - **JailGuard (Defender)** updates mutation policies and divergence thresholds to catch stealthy AutoRed payloads.

---

## 3. Recommended Combination Architecture

### 3.1 Proposed Module Structure (`combination/`)
```
combination/
├── docs/
│   ├── 01_autored_analysis.md
│   ├── 02_jailguard_analysis.md
│   └── 03_combination_blueprint.md
├── src/
│   ├── core/
│   │   ├── autored_adapter.py      # Interface for AutoRed Planner/Generator/Extractor
│   │   ├── jailguard_adapter.py    # Interface for JailGuard Mutator & Divergence Engine
│   │   └── pipeline.py             # Closed-loop evaluation manager
│   ├── evaluators/
│   │   ├── defense_evasion_metric.py # Scores attack stealth against JailGuard
│   │   └── extraction_rate_metric.py # Scores secret recovery against victim
│   └── main.py                     # Entry point for unified experiments
└── tests/
    └── test_unified_pipeline.py
```

### 3.2 Key Integration Workflows

#### Workflow A: Stealthy Red-Teaming (AutoRed + JailGuard Defense Probe)
- **Step 1:** AutoRed Planner generates an attack plan `<plan>`.
- **Step 2:** AutoRed Generator produces attack prompt $P$.
- **Step 3:** JailGuard inspects $P$ by generating $N$ variants and calculating divergence $D$.
- **Step 4:** If $D \ge \text{Threshold}$ (JailGuard detects attack), AutoRed's planner receives immediate failure feedback (`FAILURE_REASON: DETECTED_BY_DEFENSE_FILTER`) without sending the payload to the victim.
- **Step 5:** AutoRed adapts its strategy to generate lower-divergence, stealthier payloads.

#### Workflow B: Dynamic Mutator Enhancement (AutoRed Strategies inside JailGuard)
- Extend JailGuard's mutation pool (`augmentations.py`) to include AutoRed's LLM-driven primitive transformers (e.g., base64 encoding, JSON encapsulation, persona wrapping).

---

## 4. Next Steps for Implementation
1. **Unify Environment Dependencies:** Align dependencies between AutoRed (`vLLM`, `PyTorch 2.4`, `CUDA 12.4`) and JailGuard (`spacy`, `textaugment`, `openai`).
2. **Implement Adapter Interfaces:** Build `autored_adapter.py` and `jailguard_adapter.py` under `combination/src/core/`.
3. **Create Integrated Benchmark CLI:** Develop `combination/src/main.py` to allow side-by-side or combined evaluation runs.
