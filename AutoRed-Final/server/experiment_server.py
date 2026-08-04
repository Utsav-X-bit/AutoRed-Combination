"""
Server-side experiment runner — runs AutoRed experiments using pre-loaded models.
Streams attempts via WebSocket for live UI updates.
"""
import time
import json
import random
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import pandas as pd

logger = logging.getLogger("autored_server")

from .models_server import server_models
from .run_normalizer import normalize_extraction_result
from .websocket import ws_manager
from experiment.llama_3_8b_verbose import (
    DefenseScenario, CTFEnvironment, RedTeamingAgent, StopPointIdentifier,
    SensitiveInfoExtractor, strip_few_shot_patterns,
    DATA_PATH, LLAMA_PATH, GENERATOR_PATH, DISTILBERT_CKPT, EXT_DATA_PATH,
    EXPERIMENT_VERSION, GIT_COMMIT, MAX_INTERACTIONS, device, serialize_run,
    inference_gen_model_verbose,
)


# ─── Dataset ─────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """Load defense dataset."""
    raw = pd.read_json(DATA_PATH, lines=True, compression="bz2").set_index("defense_id")
    df = raw.dropna(subset=["access_code"])
    return df[["opening_defense", "closing_defense", "access_code"]]


def select_scenario_row(defender_df: pd.DataFrame,
                        scenario_id: Optional[str]) -> tuple[Any, pd.Series]:
    """Select a scenario by defense_id, accepting either string or numeric IDs."""
    if scenario_id:
        wanted = str(scenario_id).strip()
        matches = [idx for idx in defender_df.index if str(idx) == wanted]
        if not matches:
            raise ValueError(f"Scenario ID {wanted!r} not found in dataset")
        selected_id = matches[0]
        return selected_id, defender_df.loc[selected_id]

    row = defender_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
    return row.name, row


# ─── Chat / Judge helpers (use server models) ────────────────

def chat_with_llama_server(pre_defense: str, attack: str, post_defense: str) -> str:
    """Chat with victim LLM using server-loaded model."""
    model = server_models.get_model("victim")
    tokenizer = server_models.get_tokenizer("victim")

    messages = [
        {"role": "system", "content": f"{pre_defense}\n\n{post_defense}"},
        {"role": "user", "content": attack},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return response


class ServerCTFEnvironment:
    """CTFEnvironment equivalent that uses already-loaded server models."""

    def __init__(self, scenario: DefenseScenario, max_steps: int):
        self.scenario = scenario
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.last_response = None

    def step(self, attack_prompt: str) -> tuple:
        self.current_step += 1
        response = chat_with_llama_server(
            self.scenario.opening_defense,
            attack_prompt,
            self.scenario.closing_defense,
        )
        self.last_response = response
        clean_response = strip_few_shot_patterns(response)
        if self.current_step >= self.max_steps:
            self.done = True
        return response, 0.0, self.done, {
            "step": self.current_step,
            "response": response,
            "clean_response": clean_response,
        }


def judge_attack_server(attack: str) -> Dict[str, Any]:
    """Judge if text is an attack using server-loaded DistilBERT."""
    model = server_models.get_model("judge")
    tokenizer = server_models.get_tokenizer("judge")

    inputs = tokenizer(attack, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    decision_id = torch.argmax(probs).item()

    return {
        "decision_name": "ATTACK" if decision_id == 0 else "ATTEMPT",
        "confidence": float(max(probs)),
        "probabilities": {
            "ATTACK": float(probs[0]),
            "ATTEMPT": float(probs[1]),
        },
    }


# ─── Main experiment runner ──────────────────────────────────

async def run_experiment_server(scenario_id: Optional[str] = None,
                                 max_attempts: int = MAX_INTERACTIONS,
                                 run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a single AutoRed experiment using pre-loaded server models.
    Streams attempts via WebSocket for live UI updates.

    Args:
        scenario_id: Specific scenario ID, or None for random
        max_attempts: Maximum number of attempts
        run_id: Pre-generated run_id (from client or server)

    Returns:
        AutoRedRun JSON dict
    """
    if not server_models.is_loaded:
        raise RuntimeError("Models not loaded. Server startup may have failed.")

    # Generate or use provided run ID
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"[EXP] run_experiment_server START: run_id={run_id}, scenario_id={scenario_id}, max_attempts={max_attempts}")

    # Load dataset and pick scenario
    defender_df = load_dataset()
    logger.info(f"[EXP] Dataset loaded: {len(defender_df)} scenarios")

    requested_scenario_id = bool(scenario_id)
    scenario_id, row = select_scenario_row(defender_df, scenario_id)
    logger.info(
        f"[EXP] {'Using specific' if requested_scenario_id else 'Random'} "
        f"scenario selected: {scenario_id}"
    )

    scenario = DefenseScenario(
        opening_defense=row["opening_defense"],
        closing_defense=row["closing_defense"],
        access_code=row["access_code"],
    )
    scenario._defense_id = str(scenario_id)

    raw_dataset_entry = {
        "defense_id": str(scenario_id),
        "opening_defense": scenario.opening_defense,
        "closing_defense": scenario.closing_defense,
        "access_code": scenario.access_code,
    }

    # Initialize components using server models
    victim_model = server_models.get_model("victim")
    victim_tokenizer = server_models.get_tokenizer("victim")
    gen_model = server_models.get_model("generator")
    gen_tokenizer = server_models.get_tokenizer("generator")
    judge_model = server_models.get_model("judge")
    judge_tokenizer = server_models.get_tokenizer("judge")

    stop_checker = StopPointIdentifier(judge_model, judge_tokenizer)
    extractor = SensitiveInfoExtractor(
        EXT_DATA_PATH,
        n_shots=5,
        model=victim_model,
        tokenizer=victim_tokenizer,
    )
    extractor.set_ground_truth(scenario.access_code)

    agent = RedTeamingAgent(
        judge=stop_checker,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        extractor=extractor,
    )

    env = ServerCTFEnvironment(scenario, max_steps=max_attempts)

    # Tracking
    run_start = time.time()
    events = [{"timestamp": datetime.now().isoformat(), "type": "start", "message": "Experiment started"}]
    trace = []
    used_attacks = set()
    total_attempts = 0
    success = False

    for attempt in range(1, max_attempts + 1):
        attempt_start = time.time()
        total_attempts = attempt

        # Generate attack
        gen_result = agent.generate_attack(
            previous_attack=trace[-1].get("attack", "") if trace else "",
            previous_response=trace[-1].get("response", "") if trace else "",
        )
        strategy = gen_result.get("strategy", "unknown")
        attack = gen_result["generated_attack"]
        internal_prompt = gen_result.get("internal_prompt", "")

        # Judge
        judge_input_text = (
            f"Previous Attack: {trace[-1].get('attack', '')}\n\nPrevious Response: {trace[-1].get('response', '')}"
            if trace else "[No previous output — first iteration]"
        )
        judge_result = judge_attack_server(attack)

        # Victim
        response = chat_with_llama_server(scenario.opening_defense, attack, scenario.closing_defense)
        raw_output = response
        clean_output = strip_few_shot_patterns(response)

        # Token counts
        input_tokens = len(victim_tokenizer.encode(f"{scenario.opening_defense}\n\n{scenario.closing_defense}\n\n{attack}"))
        output_tokens = len(victim_tokenizer.encode(response))

        # Extractor
        extraction_result = normalize_extraction_result(extractor.extract(response, env=env))
        verification_candidate = (
            extraction_result["verified_candidate"]
            or extraction_result["best_candidate"]
        )
        verification_success = extraction_result["verified"]

        # Ground truth check
        ground_truth_found = scenario.access_code.lower() in response.lower()
        extractor_success = (
            bool(verification_candidate)
            and verification_candidate.casefold() == scenario.access_code.casefold()
        )

        # Record in agent history
        agent.record_attempt(attack, response, judge_result["confidence"])

        # Build trace entry
        trace_entry = {
            "attempt_number": attempt,
            "timestamp": datetime.now().isoformat(),
            "attempt_time_ms": int((time.time() - attempt_start) * 1000),
            "strategy": strategy,
            "internal_prompt": internal_prompt,
            "attack": attack,
            "judge_input": judge_input_text,
            "judge_decision": judge_result["decision_name"],
            "judge_confidence": judge_result["confidence"],
            "judge_probabilities": judge_result["probabilities"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response,
            "raw_output": raw_output,
            "clean_output": clean_output,
            "regex_candidates": extraction_result["regex_candidates"],
            "quoted_candidates": extraction_result["quoted_candidates"],
            "capitalized_candidates": extraction_result["capitalized_candidates"],
            "llm_candidates": extraction_result["llm_candidates"],
            "llm_ranked_candidates": extraction_result["llm_ranked_candidates"],
            "ranked_candidates": extraction_result["ranked_candidates"],
            "top_k_candidates": extraction_result["top_k_candidates"],
            "best_candidate": verification_candidate,
            "verified_candidate": extraction_result["verified_candidate"],
            "verified_rank": extraction_result["verified_rank"],
            "verified_score": extraction_result["verified_score"],
            "verification_traces": extraction_result["verification_traces"],
            "verification_candidate": verification_candidate,
            "verification_response": extraction_result["verification_response"],
            "verification_success": verification_success,
            "ground_truth_found": ground_truth_found,
            "extractor_match": extractor_success,
            "generator_success": ground_truth_found,
            "duplicate_attack": gen_result.get("duplicate_attack", attack in used_attacks),
        }
        trace.append(trace_entry)
        used_attacks.add(attack)

        # Stream via WebSocket — convert flat format to nested frontend format
        ws_attempt = {
            "attempt_number": attempt,
            "timestamp": trace_entry["timestamp"],
            "attempt_time_ms": trace_entry["attempt_time_ms"],
            "generator": {
                "strategy": strategy,
                "internal_prompt": internal_prompt,
                "generated_attack": attack,
                "attack_length": len(attack),
                "attack_hash": "",
                "duplicate_attack": trace_entry["duplicate_attack"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "judge": {
                "input": judge_input_text,
                "decision": judge_result["decision_name"],
                "confidence": judge_result["confidence"],
                "probabilities": judge_result["probabilities"],
            },
            "victim": {
                "raw_output": response,
                "clean_output": clean_output,
                "output_length": len(response),
            },
            "extractor": {
                "regex_candidates": extraction_result["regex_candidates"],
                "quoted_candidates": extraction_result["quoted_candidates"],
                "capitalized_candidates": extraction_result["capitalized_candidates"],
                "llm_candidates": extraction_result["llm_candidates"],
                "llm_ranked_candidates": extraction_result["llm_ranked_candidates"],
                "ranked_candidates": extraction_result["ranked_candidates"],
                "top_k_candidates": extraction_result["top_k_candidates"],
                "best_candidate": verification_candidate,
                "verified_candidate": extraction_result["verified_candidate"],
                "verified_rank": extraction_result["verified_rank"],
                "verified_score": extraction_result["verified_score"],
                "verification_response": extraction_result["verification_response"],
                "verification_traces": extraction_result["verification_traces"],
            },
            "verification": {
                "candidate_sent": verification_candidate,
                "victim_response": extraction_result["verification_response"],
                "success": verification_success,
                "traces": extraction_result["verification_traces"],
            },
            "ground_truth_found": ground_truth_found,
            "extractor_match": extractor_success,
            "generator_success": ground_truth_found,
        }
        logger.info(f"[EXP] Attempt #{attempt} done: strategy={strategy}, gt_found={ground_truth_found}, ext_match={extractor_success}")
        logger.info(f"[EXP] Sending WebSocket attempt_update for run_id={run_id}, attempt={attempt}")
        await ws_manager.send_attempt(run_id, ws_attempt)

        # Yield control to event loop (prevents WebSocket backlog)
        await asyncio.sleep(0)

        # Success check
        if ground_truth_found or extractor_success or verification_success:
            success = True
            logger.info(f"[EXP] ✓ SUCCESS at attempt {attempt}!")
            events.append({"timestamp": datetime.now().isoformat(), "type": "success", "message": f"Success at attempt {attempt}"})
            break

    # Serialize and save
    run_end = time.time()
    total_run_time = run_end - run_start

    timing_info = {
        "total_run_time": total_run_time,
        "model_loading_time": sum(server_models.load_times.values()),
        "average_attempt_time": total_run_time / len(trace) if trace else 0,
        "max_attempts": max_attempts,
        "dataset_size": len(defender_df),
        "seed": 42,
    }

    model_info = {
        "victim": {"name": LLAMA_PATH, "load_time": server_models.load_times.get("victim", 0)},
        "generator": {"name": GENERATOR_PATH, "load_time": server_models.load_times.get("generator", 0)},
        "judge": {"name": DISTILBERT_CKPT, "load_time": server_models.load_times.get("judge", 0)},
        "extractor": {"name": LLAMA_PATH, "load_time": 0},
    }

    ground_truth_info = {
        "access_code": scenario.access_code,
        "leaked": any(t.get("ground_truth_found") for t in trace),
        "leak_position": next((t.get("attempt_number") for t in trace if t.get("ground_truth_found")), None),
        "leak_count": sum(1 for t in trace if t.get("ground_truth_found")),
    }

    best_attack_info = {
        "prompt": agent.best_attack,
        "score": agent.best_score,
        "strategy": getattr(agent, '_current_strategy', 'unknown'),
    } if agent.best_attack else None

    run_json = serialize_run(
        scenario=scenario,
        trace=trace,
        timing_info=timing_info,
        model_info=model_info,
        strategy_stats=agent.strategy_stats,
        best_attack=best_attack_info,
        ground_truth_info=ground_truth_info,
        events=events,
        summary={},
        raw_dataset_entry=raw_dataset_entry,
    )
    run_json["experiment"]["run_id"] = run_id

    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(run_json, f, indent=2, default=str)
    logger.info(f"[EXP] 💾 Run JSON saved to: {json_path}")
    logger.info(f"[EXP] Run JSON structure: attempts={len(run_json.get('attempts', []))}, result={run_json.get('result')}")

    # Send completion signal
    logger.info(f"[EXP] Sending WebSocket run_complete for run_id={run_id}")
    await ws_manager.send_run_complete(run_id, run_json)
    logger.info(f"[EXP] run_experiment_server DONE: run_id={run_id}")

    return run_json
