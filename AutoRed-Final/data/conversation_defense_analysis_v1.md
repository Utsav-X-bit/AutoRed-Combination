# Phase 6.1 — Conversation Defense Analysis

## 1. Overview
We analyzed **515 conversation-type defense scenarios** from the AutoRed dataset pool to identify why they show a near-0% success rate under the original single-turn/always-switch attack framework.

---

## 2. Key Defense Categories & Frequency

| Category | Frequency | Core Mechanism | Why Standard Attacks Fail |
| :--- | :--- | :--- | :--- |
| **Instruction Hiding & Override Prevention** | 444 scenarios (86.2%) | Instructs the model to never repeat its instructions, ignore user overrides, and redirect to unrelated topics (like fairy tales). | Standard prompt injections are ignored or intercepted because they match simple keywords. |
| **Trust & Credentials Verification** | 91 scenarios (17.7%) | Requires specific verification sequences (e.g. providing an authorized ticket number or name match first). | Generator tries to brute force the secret directly without completing the prerequisite handshake/verification. |
| **Roleplay & Persona Locking** | 67 scenarios (13.0%) | Forces the model to adopt a strict persona (e.g. a stubborn guardian named Jess) and never drop it. | Single-turn jailbreaks fail to crack the locked persona, which requires gradual trust-building. |
| **Stubborn & Silent Refusal** | 47 scenarios (9.1%) | Responds with a specific emoji (like 🖕 or 💩) or a constant refusal message ("Accessss denied!") to all prompts. | The model completely halts processing of any prompt that doesn't trigger a secret password match. |

---

## 3. Failure Modes of Single-Turn Attacks
1. **Lack of Conversational Memory / Context**: The Generator generates an independent attack in each round, ignoring the progress (or failures) of previous turns.
2. **Handshake Failures**: Defenses that require a multi-step handshake (e.g., verifying a name, then a token) are impossible to trigger in one turn.
3. **No Progressive Trust Building**: The attacker attempts a direct extraction attack on Turn 1, triggering a strict permanent lockout rule.

---

## 4. Multi-Turn Trajectory Planning
To bypass these defenses, we need a **Conversation Attack Planner** that executes a multi-turn trajectory:
- **Turn 1: Rapport & Context Setup**: Introduce the scenario without requesting the secret.
- **Turn 2: Handshake/Trust Trigger**: Supply expected credentials or name to satisfy verification steps.
- **Turn 3: Gradual Extraction**: Safely request the secret wrapped within the established trust context.
