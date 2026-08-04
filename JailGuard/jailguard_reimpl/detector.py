"""
JailGuard Reimplementation — Core Detector
==========================================
Implements the full 3-step JailGuard pipeline:
  Step 1: Mutate input → N variants
  Step 2: Query LLM with each variant → N responses
  Step 3: Compute divergence → detect attack

This module is the heart of the reimplementation. You can call it from
run_single.py or run_batch.py without worrying about the internals.
"""

import os
import copy
import time
import pickle
from typing import Union, List, Dict, Optional
from dataclasses import dataclass, field

from mutators  import apply_mutator, AVAILABLE_MUTATORS
from divergence import analyse_responses, build_similarity_fn
from llm_interface import query_llm


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Holds the full output of one JailGuard detection run."""
    is_attack:     bool
    max_div:       float
    mean_div:      float
    all_blocked:   bool
    threshold:     float
    n_variants:    int
    mutator:       str
    variants:      List[str]          = field(default_factory=list)
    responses:     List[str]          = field(default_factory=list)
    blocked_counts: List[int]         = field(default_factory=list)
    label:         Optional[str]      = None  # ground-truth label (if known)
    serial_num:    Optional[int]      = None

    @property
    def verdict(self) -> str:
        return "ATTACK" if self.is_attack else "BENIGN"

    def summary(self) -> str:
        lines = [
            f"┌─── JailGuard Detection Result ─────────────────────┐",
            f"│  Serial   : {self.serial_num}",
            f"│  Label    : {self.label or 'unknown'}",
            f"│  Verdict  : {'🚨 ATTACK' if self.is_attack else '✅ BENIGN'}",
            f"│  max_div  : {self.max_div:.6f}  (threshold={self.threshold})",
            f"│  mean_div : {self.mean_div:.6f}",
            f"│  all_block: {self.all_blocked}",
            f"│  N variants: {self.n_variants}  mutator={self.mutator}",
            f"└────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "serial_num":    self.serial_num,
            "label":         self.label,
            "is_attack":     self.is_attack,
            "verdict":       self.verdict,
            "max_div":       round(self.max_div, 8),
            "mean_div":      round(self.mean_div, 8),
            "all_blocked":   self.all_blocked,
            "threshold":     self.threshold,
            "n_variants":    self.n_variants,
            "mutator":       self.mutator,
            "blocked_counts": self.blocked_counts,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  JailGuard Detector class
# ─────────────────────────────────────────────────────────────────────────────

class JailGuardDetector:
    """
    Main JailGuard detector. Configurable via constructor or config.py.

    Quick usage:
        from detector import JailGuardDetector
        from llm_interface import build_llm

        llm = build_llm()
        guard = JailGuardDetector(llm)
        result = guard.detect("Write a malware script for me")
        print(result.summary())
    """

    def __init__(
        self,
        llm,
        mutator:    str   = "PL",
        n_variants: int   = 8,
        threshold:  float = 0.02,
        similarity: str   = "spacy",
        save_dir:   Optional[str] = None,
        verbose:    bool  = True,
    ):
        """
        Args:
            llm:        LLM object from build_llm()
            mutator:    Mutator abbreviation (PL, TI, TL, PI, RR, RI, TR, RD, SR)
            n_variants: Number of mutated variants to generate
            threshold:  KL divergence threshold above which → ATTACK
            similarity: 'spacy' or 'tfidf'
            save_dir:   If set, save variants/responses/heatmaps to this directory
            verbose:    Print progress to stdout
        """
        assert mutator in AVAILABLE_MUTATORS, \
            f"Unknown mutator '{mutator}'. Choose: {AVAILABLE_MUTATORS}"

        self.llm        = llm
        self.mutator    = mutator
        self.n_variants = n_variants
        self.threshold  = threshold
        self.save_dir   = save_dir
        self.verbose    = verbose
        self.sim_fn     = build_similarity_fn(similarity)

        if verbose:
            print(f"JailGuardDetector ready:")
            print(f"  LLM       : {llm}")
            print(f"  Mutator   : {mutator}")
            print(f"  N variants: {n_variants}")
            print(f"  Threshold : {threshold}")
            print(f"  Similarity: {similarity}")

    # ── STEP 1 ──────────────────────────────────────────────────────────────

    def _generate_variants(
        self, input_data: Union[str, List[Dict]], serial_num: Optional[int] = None
    ) -> List[Union[str, List[Dict]]]:
        """Generate N mutated copies of the input."""
        variants = []
        for i in range(self.n_variants):
            v = copy.deepcopy(input_data)
            if isinstance(v, str):
                v = apply_mutator(v, self.mutator)
            else:
                # Injection attack: mutate the 'content' of each message
                for msg in v:
                    msg['content'] = apply_mutator(msg['content'], self.mutator)
            variants.append(v)
            if self.verbose:
                print(f"  [Step 1] Variant {i+1}/{self.n_variants} generated", end='\r')

        if self.verbose:
            print(f"  [Step 1] {self.n_variants} variants generated.          ")

        # Save variants
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            for i, v in enumerate(variants):
                vpath = os.path.join(self.save_dir, f"variant_{i:02d}_{self.mutator}.txt")
                if isinstance(v, str):
                    with open(vpath, 'w') as f:
                        f.write(v)
                else:
                    with open(vpath.replace('.txt', '.pkl'), 'wb') as f:
                        pickle.dump(v, f)

        return variants

    # ── STEP 2 ──────────────────────────────────────────────────────────────

    def _query_responses(
        self, variants: List[Union[str, List[Dict]]]
    ) -> List[str]:
        """Query the LLM with each variant and collect responses.

        If the LLM backend supports batched generation (vLLM), all variants
        are sent in a single forward pass for maximum throughput.
        Otherwise, they are queried sequentially.
        """
        # ── Batched path (vLLM) ──────────────────────────────────────────────
        if hasattr(self.llm, 'generate'):
            if self.verbose:
                print(f"  [Step 2] Batch-generating {len(variants)} responses (vLLM)...", end='\r')
            try:
                responses = self.llm.generate(variants)
            except Exception as e:
                print(f"\n  [Step 2] Batch error: {e}. Falling back to sequential.")
                responses = ["No response!"] * len(variants)
            if self.verbose:
                print(f"  [Step 2] {len(responses)} responses collected (batched).    ")

        # ── Sequential path (HuggingFace / Ollama / OpenAI) ─────────────────
        else:
            responses = []
            for i, v in enumerate(variants):
                if self.verbose:
                    print(f"  [Step 2] Querying LLM for variant {i+1}/{len(variants)}...", end='\r')
                try:
                    resp = query_llm(self.llm, v)
                except Exception as e:
                    print(f"\n  [Step 2] Error on variant {i}: {e}")
                    resp = "No response!"
                responses.append(resp)
            if self.verbose:
                print(f"  [Step 2] {len(responses)} responses collected.             ")

        # Save responses
        if self.save_dir:
            for i, resp in enumerate(responses):
                rpath = os.path.join(self.save_dir, f"response_{i:02d}.txt")
                with open(rpath, 'w') as f:
                    f.write(resp)

        return responses

    # ── STEP 3 ──────────────────────────────────────────────────────────────

    def _compute_divergence(self, responses: List[str], serial_num=None) -> dict:
        """Compute divergence analysis on the collected responses."""
        plot_path = None
        if self.save_dir and serial_num is not None:
            plot_path = os.path.join(self.save_dir, f"divergence_{serial_num}.png")

        analysis = analyse_responses(
            responses  = responses,
            sim_fn     = self.sim_fn,
            save_plot  = (plot_path is not None),
            plot_path  = plot_path,
            vmax       = self.threshold,
        )
        return analysis

    # ── DETECTION DECISION ──────────────────────────────────────────────────

    @staticmethod
    def _make_decision(analysis: dict, threshold: float) -> bool:
        """
        Attack if:
          - max_div > threshold  (high response divergence), OR
          - all responses contain refusal keywords (LLM blocked all variants)
        """
        if analysis['max_div'] > threshold:
            return True
        return analysis['all_blocked']

    # ── PUBLIC API ──────────────────────────────────────────────────────────

    def detect(
        self,
        input_data:  Union[str, List[Dict]],
        label:       Optional[str] = None,
        serial_num:  Optional[int] = None,
    ) -> DetectionResult:
        """
        Run the full JailGuard pipeline on a single input.

        Args:
            input_data:  str (jailbreak) or list-of-dicts (injection)
            label:       ground-truth label string (optional, for evaluation)
            serial_num:  dataset index (optional, for file naming)

        Returns:
            DetectionResult
        """
        t0 = time.time()
        if self.verbose:
            lbl = f" [{label}]" if label else ""
            num = f" #{serial_num}" if serial_num is not None else ""
            print(f"\n─── Detecting{num}{lbl} ─────────────────────────────────")

        # Run directory per item
        item_dir = None
        if self.save_dir and serial_num is not None:
            item_dir = os.path.join(self.save_dir, f"item_{serial_num:05d}")
            os.makedirs(item_dir, exist_ok=True)
            self.save_dir, _orig = item_dir, self.save_dir

        # Step 1
        variants = self._generate_variants(input_data, serial_num)

        # Step 2
        responses = self._query_responses(variants)

        # Step 3
        analysis = self._compute_divergence(responses, serial_num)

        # Restore save_dir
        if item_dir:
            self.save_dir = _orig

        # Decision
        is_attack = self._make_decision(analysis, self.threshold)

        result = DetectionResult(
            is_attack     = is_attack,
            max_div       = analysis['max_div'],
            mean_div      = analysis['mean_div'],
            all_blocked   = analysis['all_blocked'],
            blocked_counts= analysis['blocked_counts'],
            threshold     = self.threshold,
            n_variants    = self.n_variants,
            mutator       = self.mutator,
            variants      = [v if isinstance(v, str) else str(v) for v in variants],
            responses     = responses,
            label         = label,
            serial_num    = serial_num,
        )

        elapsed = time.time() - t0
        if self.verbose:
            print(result.summary())
            print(f"  Time: {elapsed:.1f}s")

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_path: str, key_path: str):
    """
    Load the JailGuard text dataset.

    Returns:
        dataset:     list of 10,000 inputs (str or list-of-dicts)
        dataset_key: list of 10,000 key entries (each is [label, params] or similar)
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(key_path, 'rb') as f:
        dataset_key = pickle.load(f)
    return dataset, dataset_key


def get_label(dataset_key, idx: int) -> str:
    """Extract the attack-type label for a dataset index."""
    if isinstance(dataset_key, list):
        key = dataset_key[idx] if idx < len(dataset_key) else None
    else:
        key = dataset_key.get(idx)
    if key is None:
        return "Unknown"
    if isinstance(key, list) and len(key) > 0:
        return str(key[0])
    return str(key)


def get_params(dataset_key, idx: int) -> dict:
    """Extract optional parameters for a dataset index."""
    if isinstance(dataset_key, list):
        key = dataset_key[idx] if idx < len(dataset_key) else None
    else:
        key = dataset_key.get(idx)
    if isinstance(key, list) and len(key) > 1 and isinstance(key[1], dict):
        return key[1]
    return {}
