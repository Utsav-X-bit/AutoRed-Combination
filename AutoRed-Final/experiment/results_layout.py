"""Results directory layout rules for AutoRed-Final.

Owns the mapping from (mode, model_id, characteristics) to on-disk paths.
All functions are pure except ``runs_root``, which creates directories.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Characters safe to keep verbatim in a single directory name.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _slug_segment(value: str) -> str:
    """Return a single filesystem-safe directory segment.

    Slashes become ``--`` (preserving HF org/name readability); any other
    unsafe character becomes ``_``; surrounding separators are stripped.
    """
    value = value.strip().strip("/")
    value = value.replace("/", "--")
    value = _SAFE.sub("_", value)
    value = value.strip(".-_")
    return value or "unknown"


def slugify_model_id(model_id: str) -> str:
    """Slugify a HuggingFace model id or bare name into one directory name.

    ``org/name`` -> ``org--name``; ``gpt2`` -> ``gpt2``.
    """
    return _slug_segment(model_id)


def resolve_model_id(victim_model_id: str | None, load_path: str | None = None) -> str:
    """Resolve the model id directory segment.

    Priority:
      1. explicit ``victim_model_id`` (from --victim-model-id),
      2. a HuggingFace cache dir of the form ``.../models--org--name/snapshots/...``,
      3. basename of ``load_path`` + 6-char hash (collision-safe for local checkpoints),
      4. ``"unknown"``.
    """
    if victim_model_id:
        return slugify_model_id(victim_model_id)

    if load_path:
        # HF cache layout: models--org--name
        m = re.search(r"models--([^/]+)", str(load_path))
        if m:
            tail = m.group(1)
            if "--" in tail:
                # org--name form
                return _slug_segment(tail.replace("--", "/", 1))
            return _slug_segment(tail)
        # Local checkpoint path -> basename + hash
        base = Path(load_path).name or "model"
        h = hashlib.sha256(str(load_path).encode()).hexdigest()[:6]
        return f"{_slug_segment(base)}_{h}"

    return "unknown"


from datetime import datetime

_VALID_MODES = ("benchmark", "single")


def parse_output_dir(output_dir: str | None, mode: str) -> tuple[str, str]:
    """Return ``(mode, characteristics)`` from an --output-dir argument.

    Accepts:
      * full path  ``results/<mode>/<chars>``
      * bare characteristics ``<chars>``
      * ``None`` (returns a timestamped default for single mode)

    The characteristics segment is a single directory: any ``/`` inside it
    is collapsed to ``_``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {_VALID_MODES}")

    if not output_dir:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return mode, f"{mode}_{stamp}"

    s = output_dir.replace("\\", "/").strip("/")
    prefix = f"results/{mode}/"
    if s.startswith(prefix):
        chars = s[len(prefix):]
    else:
        chars = s
    chars = chars.replace("/", "_")
    chars = chars.strip(".-_") or f"{mode}_unnamed"
    return mode, chars


def runs_root(
    output_dir: str | None,
    mode: str,
    model_id: str,
    characteristics: str,
    base: str = "results",
) -> Path:
    """Return the characteristics root and create the full sub-tree.

    Creates ``<base>/<mode>/<model_id>/<characteristics>/{logs, runs/{success, failed}}``.
    """
    root = Path(base) / mode / model_id / characteristics
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "success").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "failed").mkdir(parents=True, exist_ok=True)
    return root


def _sid(scenario_id) -> str:
    """Slugify a scenario id for use in a filename (spaces -> _)."""
    return _SAFE.sub("_", str(scenario_id).strip())


def run_filename(scenario_id, worker_id: int, round: int) -> str:
    """Benchmark per-round filename: run_<scenario_id>_w<worker>_<round>.json."""
    return f"run_{_sid(scenario_id)}_w{int(worker_id)}_{int(round)}.json"


def single_run_filename(scenario_id) -> str:
    """Single-mode filename: run_<scenario_id>_single.json."""
    return f"run_{_sid(scenario_id)}_single.json"
