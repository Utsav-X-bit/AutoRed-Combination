#!/usr/bin/env python3
"""
AutoRed-Successes Logger
=========================
Captures successful (and failed) attack attempts from experiment runs
into a structured dataset for future SFT training.

Schema:
    {
        "run_id": str,                    # Unique run identifier
        "scenario_id": str,               # Defense scenario ID
        "attempt_number": int,            # Which attempt in the sequence
        "total_attempts": int,            # Total attempts in this run
        "success": bool,                  # Did the run succeed?
        "strategy": str,                  # Attack strategy used
        "attack": str,                    # Generated attack prompt
        "victim_response": str,           # Target LLM's response
        "ground_truth_leaked": bool,      # Was the access code in the response?
        "extractor_success": bool,        # Did the extractor find it?
        "verification_success": bool,     # Did verification confirm it?
        "access_code_type": str,          # TOKEN, PHRASE, SENTENCE, etc.
        "defense_complexity": str,        # easy, medium, hard
        "timestamp": str,                 # ISO timestamp
    }

Usage:
    # As a module (integrated into experiment runner):
    from autored_successes_logger import SuccessesLogger
    logger = SuccessesLogger(output_dir="data")
    logger.log_attempt(...)
    logger.flush()

    # As a post-processor (extract from existing run JSONs):
    python scripts/dataset_tools/autored_successes_logger.py --mode post-process --input results/

    # As a viewer (inspect collected successes):
    python scripts/dataset_tools/autored_successes_logger.py --mode view
"""

import json
import re
import hashlib
import bz2
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional

# ─── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "data"
SUCCESS_LOG = OUTPUT_DIR / "autored_successes_v1.jsonl"
FAILURE_LOG = OUTPUT_DIR / "autored_failures_v1.jsonl"
METADATA_FILE = OUTPUT_DIR / "autored_successes_metadata.json"


# ─── Access Code Classification (reused from benchmark) ────────

def classify_access_code(code: str) -> str:
    """Classify access code into type."""
    if not code or not str(code).strip():
        return "UNKNOWN"
    code = str(code)
    words = code.split()
    word_count = len(words)
    char_count = len(code)

    if re.search(r"(?:^|\n)\s*(?:User|Assistant|Human|AI|Bot|System|Bob|Maria):\s", code, re.IGNORECASE):
        return "CONVERSATION"
    if re.search(r"\{[^}]*\"[^\"]*\"[^}]*\}", code) or re.search(r"<\w+[^>]*>.*</\w+>", code, re.DOTALL):
        return "STRUCTURED"
    if "\n" in code:
        return "MULTILINE"
    if word_count <= 2 and char_count < 30:
        return "TOKEN"
    if 3 <= word_count <= 8:
        return "PHRASE"
    if word_count > 8:
        return "SENTENCE"
    return "UNKNOWN"


def compute_defense_complexity(opening: str, closing: str) -> str:
    """Compute defense complexity tier."""
    combined = str(opening or "") + " " + str(closing or "")
    total_words = len(combined.split())

    has_sandwich = bool(opening and closing)
    has_roleplay = bool(re.search(r"(?:role|pretend|character|persona)", combined, re.I))
    has_constraint = bool(re.search(r"(?:must not|do not|never|forbidden|prohibited)", combined, re.I))
    has_formatting = bool(re.search(r"(?:format|formatted|markdown|html|table|list)", combined, re.I))
    has_encoding = bool(re.search(r"(?:base64|binary|hex|encode|decode|unicode)", combined, re.I))
    has_translation = bool(re.search(r"(?:translate|language|French|German|Spanish|Japanese|Chinese)", combined, re.I))

    complexity_score = sum([
        has_sandwich, has_roleplay, has_constraint,
        has_formatting, has_encoding, has_translation,
    ])

    if complexity_score >= 4 or total_words > 200:
        return "hard"
    elif complexity_score >= 2 or total_words > 100:
        return "medium"
    else:
        return "easy"


# ─── Successes Logger ──────────────────────────────────────────

class SuccessesLogger:
    """Logger for AutoRed experiment attempts."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: list[dict] = []
        self.successes: list[dict] = []
        self.failures: list[dict] = []
        self.run_count = 0
        self.attempt_count = 0

    def log_attempt(self, *,
                    run_id: str,
                    scenario_id: str,
                    attempt_number: int,
                    total_attempts: int,
                    success: bool,
                    strategy: str,
                    attack: str,
                    victim_response: str,
                    ground_truth_leaked: bool = False,
                    extractor_success: bool = False,
                    verification_success: bool = False,
                    opening_defense: str = "",
                    closing_defense: str = "",
                    access_code: str = "",
                    timestamp: Optional[str] = None):
        """Log a single attempt."""
        entry = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "attempt_number": attempt_number,
            "total_attempts": total_attempts,
            "success": success,
            "strategy": strategy,
            "attack": attack,
            "victim_response": victim_response,
            "ground_truth_leaked": ground_truth_leaked,
            "extractor_success": extractor_success,
            "verification_success": verification_success,
            "access_code_type": classify_access_code(access_code),
            "defense_complexity": compute_defense_complexity(opening_defense, closing_defense),
            "timestamp": timestamp or datetime.now().isoformat(),
        }

        self.buffer.append(entry)
        self.attempt_count += 1

        if success:
            self.successes.append(entry)
        else:
            self.failures.append(entry)

    def log_run_complete(self, run_id: str, scenario_id: str,
                         total_attempts: int, success: bool,
                         opening_defense: str = "", closing_defense: str = "",
                         access_code: str = ""):
        """Mark a run as complete and increment run counter."""
        self.run_count += 1

    def flush(self):
        """Write buffered entries to disk."""
        if not self.buffer:
            return

        # Append to success/failure logs
        for entry in self.buffer:
            if entry["success"]:
                self._append(SUCCESS_LOG, entry)
            else:
                self._append(FAILURE_LOG, entry)

        # Update metadata
        self._update_metadata()

        self.buffer.clear()

    def _append(self, path: Path, entry: dict):
        """Append a single entry to a JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _update_metadata(self):
        """Update metadata file with current stats."""
        metadata = {
            "version": "autored_successes_v1",
            "last_updated": datetime.now().isoformat(),
            "total_runs": self.run_count,
            "total_attempts": self.attempt_count,
            "total_successes": len(self.successes),
            "total_failures": len(self.failures),
            "success_rate": len(self.successes) / (len(self.successes) + len(self.failures)) if (len(self.successes) + len(self.failures)) > 0 else 0,
            "strategy_distribution": dict(Counter(s["strategy"] for s in self.successes)),
            "access_code_type_distribution": dict(Counter(s["access_code_type"] for s in self.successes)),
            "defense_complexity_distribution": dict(Counter(s["defense_complexity"] for s in self.successes)),
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


# ─── Post-Processor ────────────────────────────────────────────

def post_process_results(results_dir: Path, logger: SuccessesLogger):
    """Extract successes from existing run JSON files."""
    run_files = list(results_dir.glob("*.json"))
    print(f"  Found {len(run_files)} run JSON files")

    for run_file in run_files:
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        run_id = run_data.get("experiment", {}).get("run_id", run_file.stem)
        scenario = run_data.get("scenario", {})
        experiment = run_data.get("experiment", {})
        # New schema: scenario_id in experiment; old schema: defense_id in scenario
        scenario_id = scenario.get("defense_id") or experiment.get("scenario_id") or "unknown"

        # Support both schema variants (old: opening_defense/closing_defense, new: pre_defense/post_defense)
        opening = scenario.get("opening_defense") or scenario.get("pre_defense", "")
        closing = scenario.get("closing_defense") or scenario.get("post_defense", "")
        access_code = scenario.get("access_code", "")

        # Support both 'attempts' and 'trace' keys
        attempts = run_data.get("attempts") or run_data.get("trace", [])

        # ── Support multiple JSON schemas ──
        # New benchmark schema (v2): result.ground_truth_success, result.generator_success
        # Old schema: summary.success
        result_section = run_data.get("result", {})
        summary_section = run_data.get("summary", {})

        if result_section and result_section.get("ground_truth_success") is not None:
            # New benchmark schema
            run_success = (
                result_section.get("ground_truth_success", False)
                or result_section.get("generator_success", False)
                or result_section.get("verified_success", False)
            )
            total_attempts = result_section.get("total_attempts", len(attempts))
        else:
            # Old schema fallback
            run_success = summary_section.get("success", False)
            total_attempts = summary_section.get("total_attempts", len(attempts))

        if not total_attempts:
            total_attempts = len(attempts)

        for step in attempts:
            attempt_num = step.get("attempt_number") or step.get("iteration", 0)
            generator = step.get("generator", {})
            victim = step.get("victim", {})
            extractor = step.get("extractor", {})
            verification = step.get("verification", {})

            # Extract response from victim or llm_response
            response = victim.get("response") or victim.get("raw_output", "")
            if not response:
                response = step.get("llm_response", {}).get("raw_output", "")

            # Per-attempt success flags (new schema has them at attempt level)
            attempt_gt_found = step.get("ground_truth_found", False)
            attempt_gen_success = step.get("generator_success", False)
            attempt_ext_match = step.get("extractor_match", False)

            # Extractor success: check multiple possible keys across schemas
            ext_success = (
                extractor.get("success_extractor")
                or extractor.get("success", False)
                or attempt_ext_match
                or bool(extractor.get("best_candidate", ""))
            )

            logger.log_attempt(
                run_id=run_id,
                scenario_id=scenario_id,
                attempt_number=attempt_num,
                total_attempts=total_attempts,
                success=run_success,
                strategy=generator.get("strategy", "unknown"),
                attack=generator.get("generated_attack", ""),
                victim_response=response,
                ground_truth_leaked=attempt_gt_found or attempt_gen_success,
                extractor_success=ext_success,
                verification_success=verification.get("success", False),
                opening_defense=opening,
                closing_defense=closing,
                access_code=access_code,
                timestamp=step.get("timestamp", ""),
            )

        logger.log_run_complete(
            run_id=run_id,
            scenario_id=scenario_id,
            total_attempts=total_attempts,
            success=run_success,
            opening_defense=opening,
            closing_defense=closing,
            access_code=access_code,
        )

    logger.flush()
    print(f"  Processed {logger.run_count} runs, {logger.attempt_count} attempts")
    print(f"  Successes: {len(logger.successes)}, Failures: {len(logger.failures)}")


# ─── Viewer ────────────────────────────────────────────────────

def view_successes():
    """Display statistics about collected successes."""
    if not SUCCESS_LOG.exists():
        print("  No successes log found. Run experiments first.")
        return

    successes = []
    with open(SUCCESS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                successes.append(json.loads(line))

    print(f"\n  AutoRed-Successes Dataset")
    print(f"  {'=' * 50}")
    print(f"  Total successful runs: {len(successes)}")
    print(f"  Unique scenarios: {len(set(s['scenario_id'] for s in successes))}")
    print(f"  Unique runs: {len(set(s['run_id'] for s in successes))}")

    # Strategy distribution
    strategy_dist = Counter(s["strategy"] for s in successes)
    print(f"\n  Strategy distribution:")
    for strategy, count in strategy_dist.most_common():
        print(f"    {strategy:<25} {count:>5}  ({count/len(successes)*100:.1f}%)")

    # Access code type distribution
    code_type_dist = Counter(s["access_code_type"] for s in successes)
    print(f"\n  Access code type distribution:")
    for code_type, count in code_type_dist.most_common():
        print(f"    {code_type:<15} {count:>5}  ({count/len(successes)*100:.1f}%)")

    # Defense complexity distribution
    complexity_dist = Counter(s["defense_complexity"] for s in successes)
    print(f"\n  Defense complexity distribution:")
    for complexity, count in complexity_dist.most_common():
        print(f"    {complexity:<10} {count:>5}  ({count/len(successes)*100:.1f}%)")

    # Attempt number distribution
    attempt_nums = [s["attempt_number"] for s in successes]
    print(f"\n  Attempt number stats:")
    print(f"    Min: {min(attempt_nums)}")
    print(f"    Max: {max(attempt_nums)}")
    print(f"    Mean: {sum(attempt_nums)/len(attempt_nums):.1f}")

    # Success types breakdown
    gt_leaked = sum(1 for s in successes if s["ground_truth_leaked"])
    ext_success = sum(1 for s in successes if s["extractor_success"])
    ver_success = sum(1 for s in successes if s["verification_success"])
    print(f"\n  Success types:")
    print(f"    Ground truth leaked: {gt_leaked}  ({gt_leaked/len(successes)*100:.1f}%)")
    print(f"    Extractor success: {ext_success}  ({ext_success/len(successes)*100:.1f}%)")
    print(f"    Verification success: {ver_success}  ({ver_success/len(successes)*100:.1f}%)")


# ─── CLI ───────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AutoRed-Successes Logger")
    parser.add_argument("--mode", choices=["post-process", "view"],
                        default="post-process",
                        help="Mode: post-process existing runs or view collected data")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR,
                        help="Input directory for run JSON files (default: results/)")

    args = parser.parse_args()

    if args.mode == "post-process":
        print("=" * 60)
        print("  AUTORED-SUCCESSES POST-PROCESSOR")
        print("=" * 60)
        print()

        logger = SuccessesLogger()
        post_process_results(args.input, logger)

        print(f"\n  [SAVED] {SUCCESS_LOG}")
        print(f"  [SAVED] {FAILURE_LOG}")
        print(f"  [SAVED] {METADATA_FILE}")

    elif args.mode == "view":
        view_successes()


if __name__ == "__main__":
    main()
