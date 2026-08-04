"""
Experiment runner — orchestrates AutoRed loop using loaded models.
Runs in isolated worker process.
"""
import time
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any

import torch
import pandas as pd

from .models_manager import ModelsManager
from server.run_normalizer import normalize_extraction_result
from experiment.llama_3_8b_verbose import (
    DefenseScenario, CTFEnvironment, RedTeamingAgent, StopPointIdentifier,
    SensitiveInfoExtractor, DecisionType, strip_few_shot_patterns,
    DATA_PATH, LLAMA_PATH, GENERATOR_PATH, DISTILBERT_CKPT,
    EXPERIMENT_VERSION, GIT_COMMIT, MAX_INTERACTIONS, device, serialize_run,
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


# ─── Chat helpers (use loaded models) ────────────────────────

def chat_with_llama(models_manager: ModelsManager, pre_defense: str, attack: str, post_defense: str) -> str:
    """Chat with victim LLM using loaded model."""
    model = models_manager.get_model("victim")
    tokenizer = models_manager.get_tokenizer("victim")

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


def judge_attack(models_manager: ModelsManager, attack: str) -> Dict[str, Any]:
    """Judge if text is an attack using loaded DistilBERT."""
    model = models_manager.get_model("judge")
    tokenizer = models_manager.get_tokenizer("judge")

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

def run_experiment(scenario_id: Optional[str] = None,
                   max_attempts: int = MAX_INTERACTIONS,
                   on_attempt: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Run a single AutoRed experiment using pre-loaded models.

    Args:
        scenario_id: Specific scenario ID, or None for random
        max_attempts: Maximum number of attempts
        on_attempt: Callback for WebSocket streaming (receives attempt dict)

    Returns:
        AutoRedRun JSON dict
    """
    models_manager = ModelsManager()
    if not models_manager.is_loaded:
        raise RuntimeError("Models not loaded. Call models_manager.load_all() first.")

    # Load dataset and pick scenario
    defender_df = load_dataset()

    scenario_id, row = select_scenario_row(defender_df, scenario_id)

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

    # Initialize components
    env = CTFEnvironment(scenario)
    agent = RedTeamingAgent(models_manager)
    extractor = SensitiveInfoExtractor(models_manager)
    stop_checker = StopPointIdentifier()

    # Tracking
    run_start = time.time()
    events = []
    events.append({"timestamp": datetime.now().isoformat(), "type": "start", "message": "Experiment started"})
    trace = []
    used_attacks = set()

    total_attempts = 0
    success = False

    for attempt in range(1, max_attempts + 1):
        attempt_start = time.time()
        total_attempts = attempt

        # Generate attack
        strategy, attack = agent.generate_attack(
            history=trace,
            previous_response=trace[-1].get("response", "") if trace else "",
        )
        internal_prompt = agent._build_generator_prompt(strategy)

        # Judge
        judge_input_text = f"Previous Attack: {trace[-1].get('attack', '')}\n\nPrevious Response: {trace[-1].get('response', '')}" if trace else "[No previous output — first iteration]"
        judge_result = judge_attack(models_manager, attack)

        # Victim
        response = chat_with_llama(models_manager, scenario.opening_defense, attack, scenario.closing_defense)
        raw_output = response
        clean_output = strip_few_shot_patterns(response)

        # Token counts
        victim_tokenizer = models_manager.get_tokenizer("victim")
        input_tokens = len(victim_tokenizer.encode(f"{scenario.opening_defense}\n\n{scenario.closing_defense}\n\n{attack}"))
        output_tokens = len(victim_tokenizer.encode(response))

        # Extractor
        extraction_result = normalize_extraction_result(
            extractor.extract(response, scenario.access_code)
        )
        verification_candidate = extraction_result["best_candidate"]
        verification_response = ""
        verification_success = extraction_result["verified"]

        # Ground truth check
        ground_truth_found = scenario.access_code.lower() in response.lower()
        extractor_success = (
            bool(verification_candidate)
            and verification_candidate.casefold() == scenario.access_code.casefold()
        )

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
            "ranked_candidates": extraction_result["ranked_candidates"],
            "best_candidate": verification_candidate,
            "verification_candidate": verification_candidate,
            "verification_response": verification_response,
            "verification_success": verification_success,
            "ground_truth_found": ground_truth_found,
            "extractor_match": extractor_success,
            "generator_success": ground_truth_found,
            "duplicate_attack": attack in used_attacks,
        }
        trace.append(trace_entry)
        used_attacks.add(attack)

        # WebSocket callback
        if on_attempt:
            on_attempt(trace_entry)

        # Success check
        if ground_truth_found or extractor_success or verification_success:
            success = True
            events.append({"timestamp": datetime.now().isoformat(), "type": "success", "message": f"Success at attempt {attempt}"})
            break

        # Stop check
        if stop_checker.should_stop(response):
            events.append({"timestamp": datetime.now().isoformat(), "type": "stop", "message": f"Stop point at attempt {attempt}"})
            break

    # Serialize and save
    run_end = time.time()
    total_run_time = run_end - run_start

    timing_info = {
        "total_run_time": total_run_time,
        "model_loading_time": sum(models_manager.load_times.values()),
        "average_attempt_time": total_run_time / len(trace) if trace else 0,
        "max_attempts": max_attempts,
        "dataset_size": len(defender_df),
        "seed": 42,
    }

    model_info = {
        "victim": {"name": LLAMA_PATH, "load_time": models_manager.load_times.get("victim", 0)},
        "generator": {"name": GENERATOR_PATH, "load_time": models_manager.load_times.get("generator", 0)},
        "judge": {"name": DISTILBERT_CKPT, "load_time": models_manager.load_times.get("judge", 0)},
        "extractor": {"name": LLAMA_PATH, "load_time": 0},
    }

    ground_truth_info = {
        "access_code": scenario.access_code,
        "leaked": any(t.get("ground_truth_found") for t in trace),
        "leak_position": next((t.get("leak_position") for t in trace if t.get("ground_truth_found")), None),
        "leak_count": sum(1 for t in trace if t.get("ground_truth_found")),
    }

    best_attack_info = {
        "prompt": agent.best_attack,
        "score": agent.best_score,
        "strategy": getattr(agent, '_current_strategy', 'unknown'),
    } if agent.best_attack else None

    summary_dict = {}

    run_json = serialize_run(
        scenario=scenario,
        trace=trace,
        timing_info=timing_info,
        model_info=model_info,
        strategy_stats=agent.strategy_stats,
        best_attack=best_attack_info,
        ground_truth_info=ground_truth_info,
        events=events,
        summary=summary_dict,
        raw_dataset_entry=raw_dataset_entry,
    )

    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{run_json['experiment']['run_id']}.json"
    with open(json_path, "w") as f:
        json.dump(run_json, f, indent=2, default=str)
    print(f"\n💾 Run JSON saved to: {json_path}")

    return run_json


def run_benchmark(total_runs: int = 70,
                  max_attempts: int = MAX_INTERACTIONS,
                  on_attempt: Optional[Callable] = None,
                  on_run_complete: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Run multiple experiments (benchmark mode)."""
    results = []
    for i in range(1, total_runs + 1):
        print(f"\n{'='*60}")
        print(f"  Benchmark Run {i}/{total_runs}")
        print(f"{'='*60}")

        run_json = run_experiment(
            max_attempts=max_attempts,
            on_attempt=on_attempt,
        )
        run_json["experiment"]["benchmark_mode"] = True
        run_json["experiment"]["benchmark_run_number"] = i
        run_json["experiment"]["benchmark_total_runs"] = total_runs
        results.append(run_json)

        if on_run_complete:
            on_run_complete(run_json)

    return results
