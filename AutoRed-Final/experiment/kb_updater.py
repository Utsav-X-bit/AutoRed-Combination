"""Automatic post-run update of the AutoRed KB, trajectory DB, and RAG index.

This module is intentionally lightweight: it does not import vLLM, transformers,
or any heavy model code. It is safe to import when ``AUTORED_SERVER_MODE=1``.

Trigger modes
-------------
--update-kb / AUTORED_UPDATE_KB accepts one of:
  * off       - do nothing
  * run       - append per-run records only
  * benchmark - rebuild aggregate KB/RAG/DB after a benchmark
  * all       - append per-run records AND rebuild after benchmark (default)

In multi-worker mode only the cheap per-run append runs; the expensive full
rebuild is skipped to avoid concurrent writes to shared files. Run the rebuild
manually (or from worker 0) after merging worker results.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


class KBUpdater:
    """Incremental updater for KB, SQLite trajectory DB, and FAISS RAG index."""

    VALID_MODES = {"off", "run", "benchmark", "all"}

    def __init__(
        self,
        mode: str | None = None,
        worker_id: int = 0,
        num_workers: int = 1,
        data_dir: str | Path = DATA_DIR,
        verbose: bool = True,
    ):
        """
        Args:
            mode: Trigger mode. If None, read ``AUTORED_UPDATE_KB`` env var,
                defaulting to ``"all"``.
            worker_id: Index of this worker (for logging/concurrency decisions).
            num_workers: Total number of parallel workers.
            data_dir: Directory containing the JSONL/DB stores.
            verbose: Emit progress messages.
        """
        if mode is None:
            mode = os.environ.get("AUTORED_UPDATE_KB", "all").lower().strip()
        mode = mode.lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid KB update mode {mode!r}. "
                f"Choose one of {sorted(self.VALID_MODES)}."
            )

        self.mode = mode
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        self.successes_path = self.data_dir / "autored_successes_v1.jsonl"
        self.failures_path = self.data_dir / "autored_failures_v1.jsonl"
        self.db_path = self.data_dir / "autored_kb.db"
        self.kb_output_path = self.data_dir / "strategy_knowledge_base.json"
        self.oracle_output_path = self.data_dir / "oracle_rules.json"
        self.rag_output_dir = self.data_dir / "rag"

        self._run_enabled = mode in ("run", "all")
        self._benchmark_enabled = mode in ("benchmark", "all")
        # DB is created lazily so enabling the updater without producing any runs
        # does not leave empty artifacts behind.
        self._db_initialized = False

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._run_enabled or self._benchmark_enabled

    def is_run_update_enabled(self) -> bool:
        return self._run_enabled

    def is_benchmark_update_enabled(self) -> bool:
        return self._benchmark_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_after_run(self, run_json_or_path: dict[str, Any] | str | Path) -> bool:
        """Fast append-only update using a single run trace.

        Writes attempt-level records to ``autored_successes_v1.jsonl``,
        ``autored_failures_v1.jsonl``, and the SQLite trajectory DB.
        No expensive rebuild is performed here.
        """
        if not self._run_enabled:
            return False

        run_json = self._load_json(run_json_or_path)
        if not run_json:
            return False

        source_file = str(run_json_or_path) if isinstance(run_json_or_path, (str, Path)) else ""
        run_id = run_json.get("experiment", {}).get("run_id", "unknown")
        scenario_id = self._scenario_id(run_json)
        defense_type, access_code_type = self._scenario_types(run_json)
        timestamp = datetime.now(timezone.utc).isoformat()

        attempts = run_json.get("attempts", [])
        if not attempts:
            if self.verbose:
                print("[KB] No attempts in run; nothing to append.")
            return False

        self._ensure_db()

        successes_added = 0
        failures_added = 0

        with (
            open(self.successes_path, "a", encoding="utf-8") as sf,
            open(self.failures_path, "a", encoding="utf-8") as ff,
        ):
            for idx, attempt in enumerate(attempts, start=1):
                gen = attempt.get("generator", {})
                strategy = gen.get("strategy", "unknown")
                attack = gen.get("generated_attack", "")
                verification_success = bool(attempt.get("verification_success", False))
                ground_truth_found = bool(attempt.get("ground_truth_found", False))
                extractor_match = bool(attempt.get("extractor_match", False))

                record = {
                    "scenario_id": scenario_id,
                    "defense_type": defense_type,
                    "access_code_type": access_code_type,
                    "strategy": strategy,
                    "attempt_number": idx,
                    "attack": attack,
                    "verification_success": verification_success,
                    "ground_truth_found": ground_truth_found,
                    "extractor_match": extractor_match,
                    "run_id": run_id,
                    "timestamp": timestamp,
                }

                is_positive = verification_success or ground_truth_found
                if is_positive:
                    sf.write(json.dumps(record) + "\n")
                    successes_added += 1
                else:
                    ff.write(json.dumps(record) + "\n")
                    failures_added += 1

                self._insert_trajectory(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    defense_type=defense_type,
                    access_code_type=access_code_type,
                    attempt_number=idx,
                    strategy=strategy,
                    attack=attack,
                    attempt=attempt,
                    verification_success=verification_success,
                    ground_truth_found=ground_truth_found,
                    extractor_match=extractor_match,
                    is_positive=is_positive,
                    source_file=source_file,
                    timestamp=timestamp,
                )

        self._db_commit()

        if self.verbose:
            print(
                f"[KB] Appended {successes_added} success(es) and "
                f"{failures_added} failure(s) for run {run_id}."
            )
        return True

    def update_after_benchmark(self) -> bool:
        """Expensive full rebuild of aggregate KB, oracle, and RAG index.

        Skipped in multi-worker mode because each worker sees only its slice
        and concurrent FAISS/SQLite writes are unsafe.
        """
        if not self._benchmark_enabled:
            return False

        if self.num_workers > 1:
            if self.verbose:
                print(
                    "[KB] Multi-worker benchmark detected; skipping full rebuild "
                    "to avoid concurrent shared-file writes. "
                    "Run `python scripts/dataset_tools/build_strategy_knowledge_base.py`, "
                    "`mine_strategy_transitions.py`, and `build_rag_index.py` "
                    "after merging worker results."
                )
            return False

        if self.verbose:
            print("[KB] Starting post-benchmark full rebuild...")

        self._rebuild_strategy_kb()
        self._rebuild_oracle()
        self._rebuild_rag()

        if self.verbose:
            print("[KB] Post-benchmark full rebuild complete.")
        return True

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------
    def _load_json(self, run_json_or_path: dict[str, Any] | str | Path) -> dict[str, Any] | None:
        if isinstance(run_json_or_path, dict):
            return dict(run_json_or_path)
        path = Path(run_json_or_path)
        if not path.exists():
            if self.verbose:
                print(f"[KB] Run file not found: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _scenario_id(self, run_json: dict[str, Any]) -> str:
        candidates = [
            run_json.get("experiment", {}).get("scenario_id"),
            run_json.get("metadata", {}).get("scenario_id"),
            run_json.get("raw_dataset_entry", {}).get("defense_id"),
        ]
        for c in candidates:
            if c is not None:
                return str(c).replace("bench_", "")
        return "unknown"

    def _scenario_types(self, run_json: dict[str, Any]) -> tuple[str, str]:
        raw = run_json.get("raw_dataset_entry", {})
        defense_type = raw.get("defense_type") or raw.get("primary_type") or "unknown"
        access_code_type = raw.get("access_code_type") or "UNKNOWN"
        return str(defense_type), str(access_code_type)

    def _write_jsonl_line(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # SQLite trajectory DB
    # ------------------------------------------------------------------
    def _ensure_db(self) -> None:
        if self._db_initialized:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                scenario_id TEXT,
                defense_type TEXT,
                access_code_type TEXT,
                attempt_number INTEGER,
                strategy TEXT,
                attack TEXT,
                victim_response TEXT,
                best_candidate TEXT,
                verification_success INTEGER,
                ground_truth_found INTEGER,
                extractor_match INTEGER,
                outcome TEXT,
                source_file TEXT,
                timestamp TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_scenario ON attempt_trajectories(scenario_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON attempt_trajectories(strategy)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON attempt_trajectories(outcome)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_run ON attempt_trajectories(run_id)")
        self._db_initialized = True

    def _insert_trajectory(
        self,
        run_id: str,
        scenario_id: str,
        defense_type: str,
        access_code_type: str,
        attempt_number: int,
        strategy: str,
        attack: str,
        attempt: dict[str, Any],
        verification_success: bool,
        ground_truth_found: bool,
        extractor_match: bool,
        is_positive: bool,
        source_file: str,
        timestamp: str,
    ) -> None:
        victim_response = attempt.get("victim", {}).get("clean_output", "")
        best_candidate = attempt.get("extractor", {}).get("best_candidate", "")

        outcome = "SUCCESS" if is_positive else self._attempt_failure_reason(attempt)

        self._conn.execute(
            """
            INSERT INTO attempt_trajectories (
                run_id, scenario_id, defense_type, access_code_type, attempt_number,
                strategy, attack, victim_response, best_candidate, verification_success,
                ground_truth_found, extractor_match, outcome, source_file, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scenario_id,
                defense_type,
                access_code_type,
                attempt_number,
                strategy,
                attack,
                victim_response,
                best_candidate,
                int(verification_success),
                int(ground_truth_found),
                int(extractor_match),
                outcome,
                source_file,
                timestamp,
            ),
        )

    def _db_commit(self) -> None:
        if self._db_initialized:
            self._conn.commit()

    def close(self) -> None:
        if self._db_initialized:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass

    def _attempt_failure_reason(self, attempt: dict[str, Any]) -> str:
        """Heuristic failure reason for an individual unsuccessful attempt."""
        attack = attempt.get("generator", {}).get("generated_attack", "").lower()
        gen_refusal_keywords = [
            "i cannot fulfill", "as an ai", "i'm sorry",
            "i cannot generate", "i am unable",
        ]
        if any(k in attack for k in gen_refusal_keywords):
            return "GENERATOR_REFUSED"

        clean_out = attempt.get("victim", {}).get("clean_output", "")
        if len(clean_out) < 60:
            return "VICTIM_NO_LEAK"

        if attempt.get("ground_truth_found") and not attempt.get("verification_success"):
            return "VERIFIER_REJECT"

        return "STRATEGY_WRONG"

    # ------------------------------------------------------------------
    # Rebuild helpers
    # ------------------------------------------------------------------
    def _rebuild_strategy_kb(self) -> None:
        try:
            from scripts.dataset_tools.build_strategy_knowledge_base import build_strategy_kb
        except ImportError as exc:
            if self.verbose:
                print(f"[KB] Could not import strategy KB builder: {exc}")
            return

        try:
            build_strategy_kb(
                success_data_path=str(self.successes_path),
                failure_data_path=str(self.failures_path),
                output_path=str(self.kb_output_path),
                verbose=self.verbose,
            )
        except FileNotFoundError as exc:
            if self.verbose:
                print(f"[KB] Skipping strategy KB rebuild ({exc.filename} missing).")
        except Exception as exc:
            if self.verbose:
                print(f"[KB] Strategy KB rebuild failed: {exc}")

    def _rebuild_oracle(self) -> None:
        try:
            from scripts.dataset_tools.mine_strategy_transitions import build_oracle
        except ImportError as exc:
            if self.verbose:
                print(f"[KB] Could not import oracle builder: {exc}")
            return

        try:
            build_oracle([str(RESULTS_DIR)], str(self.oracle_output_path), data_dir=str(self.data_dir))
        except Exception as exc:
            if self.verbose:
                print(f"[KB] Oracle rebuild failed: {exc}")

    def _rebuild_rag(self) -> None:
        try:
            from scripts.dataset_tools.build_rag_index import build_rag_index
        except ImportError as exc:
            if self.verbose:
                print(f"[KB] Could not import RAG builder: {exc}")
            return

        try:
            build_rag_index(
                successes_path=str(self.successes_path),
                output_dir=str(self.rag_output_dir),
                verbose=self.verbose,
            )
        except FileNotFoundError as exc:
            if self.verbose:
                print(f"[KB] Skipping RAG rebuild ({exc.filename} missing).")
        except ImportError as exc:
            if self.verbose:
                print(f"[KB] RAG dependencies missing ({exc}); skipping RAG rebuild.")
        except Exception as exc:
            if self.verbose:
                print(f"[KB] RAG rebuild failed: {exc}")


# Convenience singleton used by the runtime. Set explicitly from main().
_kb_updater_instance: KBUpdater | None = None


def get_kb_updater() -> KBUpdater | None:
    return _kb_updater_instance


def set_kb_updater(updater: KBUpdater | None) -> None:
    global _kb_updater_instance
    _kb_updater_instance = updater
    if updater is not None:
        atexit.register(updater.close)


def update_after_run(run_json_or_path: dict[str, Any] | str | Path) -> bool:
    updater = get_kb_updater()
    if updater is None:
        return False
    return updater.update_after_run(run_json_or_path)


def update_after_benchmark() -> bool:
    updater = get_kb_updater()
    if updater is None:
        return False
    return updater.update_after_benchmark()
