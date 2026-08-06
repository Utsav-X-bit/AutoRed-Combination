import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .run_normalizer import normalize_run

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"        # legacy flat layout
BENCHMARK_DIR = RESULTS_DIR / "benchmark"          # new results_layout.py tree


def _overall_success(result: Dict[str, Any]) -> bool:
    return any(
        bool(result.get(key))
        for key in ("ground_truth_success", "extractor_success", "verified_success")
    )


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _is_benchmark_artifact(path: Path) -> bool:
    """True for per-round run JSONs that belong to a benchmark run.

    Excludes both the legacy ``results/benchmarks/`` tree (plural) and the new
    ``results/benchmark/`` tree (singular) so benchmark per-round JSONs are not
    surfaced as standalone single runs in ``/api/runs/all``.
    """
    return "benchmarks" in path.parts or "benchmark" in path.parts


def _archive_date_from_timestamp(timestamp: str) -> Optional[str]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def _trace_archives_for_timestamp(timestamp: str) -> List[Path]:
    date = _archive_date_from_timestamp(timestamp)
    if not date:
        return []
    day_root = RESULTS_DIR / date
    if not day_root.exists():
        return []
    return sorted(
        [p for p in day_root.iterdir() if p.is_dir() and list(p.glob("run_*.json"))],
        key=lambda p: p.stat().st_mtime,
    )


# ---------------------------------------------------------------------------
# Benchmark discovery — supports two on-disk layouts:
#   * legacy flat:   results/benchmarks/<name>/merged_summary.json
#   * new nested:    results/benchmark/<model_id>/<chars>/logs/merged_summary.json
#                    with runs under <chars>/runs/{success,failed}/run_*.json
# A nested benchmark_id is the slash-form "<model>/<chars>".
# ---------------------------------------------------------------------------


def _run_files_in(benchmark_dir: Path) -> List[Path]:
    """Per-round run JSONs for a benchmark dir, across layouts.

    New layout: <dir>/runs/{success,failed}/run_*.json
    Legacy layout: date-based archives under results/<YYYY-MM-DD>/... (no local runs)
    """
    runs_root = benchmark_dir / "runs"
    if runs_root.exists():
        files = list(runs_root.glob("success/run_*.json")) + list(runs_root.glob("failed/run_*.json"))
        return sorted(files)
    return []


def _discover_benchmark_dirs() -> List[Dict[str, Any]]:
    """Return one entry per benchmark across legacy + new layouts.

    Each entry: {benchmark_id, benchmark_dir, summary_file, benchmark_group, layout}
    """
    found: List[Dict[str, Any]] = []

    # Legacy flat layout.
    if BENCHMARKS_DIR.exists():
        for d in BENCHMARKS_DIR.iterdir():
            if not d.is_dir():
                continue
            summary = d / "merged_summary.json"
            if summary.exists():
                found.append({
                    "benchmark_id": d.name,
                    "benchmark_dir": d,
                    "summary_file": summary,
                    "benchmark_group": None,
                    "layout": "legacy",
                })

    # New nested layout: benchmark/<model>/<chars>/logs/merged_summary.json
    if BENCHMARK_DIR.exists():
        for model_dir in sorted(BENCHMARK_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            for chars_dir in sorted(model_dir.iterdir()):
                if not chars_dir.is_dir():
                    continue
                summary = chars_dir / "logs" / "merged_summary.json"
                if summary.exists():
                    found.append({
                        "benchmark_id": f"{model_dir.name}/{chars_dir.name}",
                        "benchmark_dir": chars_dir,
                        "summary_file": summary,
                        "benchmark_group": model_dir.name,
                        "layout": "nested",
                    })

    return found


def _summarize_run_file(path: Path) -> Dict[str, Any]:
    return _run_metadata_from_file(path)


def _run_metadata_from_file(path: Path) -> Dict[str, Any]:
    raw = _load_json(path) or {}
    normalized = normalize_run(raw, path.stem)
    return {
        "run_id": normalized.get("experiment", {}).get("run_id", path.stem),
        "file_path": str(path),
        "timestamp": normalized.get("experiment", {}).get("timestamp", ""),
        "scenario_id": normalized.get("experiment", {}).get("scenario_id", ""),
        "success": _overall_success(normalized.get("result", {})),
        "verified_success": bool(normalized.get("result", {}).get("verified_success")),
        "total_attempts": normalized.get("result", {}).get("total_attempts", 0),
        "access_code": normalized.get("scenario", {}).get("access_code", ""),
        "generator": normalized.get("models", {}).get("generator", {}).get("name", ""),
        "victim": normalized.get("models", {}).get("victim", {}).get("name", ""),
        "benchmark_mode": normalized.get("experiment", {}).get("benchmark_mode", False),
        "worker_id": normalized.get("experiment", {}).get("worker_id"),
        "attempt_count": len(normalized.get("attempts", [])),
    }


def _summarize_trace_archive(path: Path) -> Dict[str, Any]:
    runs = []
    total_successes = 0
    verified_successes = 0
    total_attempts = 0
    for run_file in sorted(path.glob("run_*.json")):
        meta = _run_metadata_from_file(run_file)
        runs.append(meta)
        total_successes += int(meta["success"])
        verified_successes += int(meta["verified_success"])
        total_attempts += int(meta["total_attempts"])

    run_count = len(runs)
    return {
        "archive_id": f"{path.parent.name}/{path.name}",
        "date": path.parent.name,
        "path": str(path),
        "timestamp": runs[0]["timestamp"] if runs else "",
        "run_count": run_count,
        "success_rate": (total_successes / run_count) if run_count else 0.0,
        "verified_rate": (verified_successes / run_count) if run_count else 0.0,
        "avg_attempts_on_success": (
            total_attempts / total_successes if total_successes else 0.0
        ),
        "runs": runs,
    }


def list_trace_archives() -> List[Dict[str, Any]]:
    """List dated trace archives under results/YYYY-MM-DD/*."""
    ensure_results_dir()
    archives: List[Dict[str, Any]] = []
    for day_dir in sorted(
        [p for p in RESULTS_DIR.iterdir() if p.is_dir() and len(p.name) == 10 and p.name[4] == "-" and p.name[7] == "-"]
    ):
        for archive_dir in sorted([p for p in day_dir.iterdir() if p.is_dir() and list(p.glob("run_*.json"))]):
            archives.append(_summarize_trace_archive(archive_dir))
    return archives


def _benchmark_summary_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Project the stable summary fields every benchmark list entry exposes."""
    return {
        "total_rounds": data.get("total_rounds", 0),
        "total_successes": data.get("total_successes", 0),
        "verified_success": data.get("verified_success", 0),
        "success_rate": data.get("success_rate", 0.0),
        "avg_attempts_on_success": data.get("avg_attempts_on_success", 0.0),
        "top1_success": data.get("top1_success", 0),
        "top3_success": data.get("top3_success", 0),
        "top5_success": data.get("top5_success", 0),
        "extractor_metrics": data.get("extractor_metrics", {}),
        "worker_summaries": data.get("worker_summaries", []),
    }


def list_benchmarks(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """List benchmark summaries across legacy (results/benchmarks) and new
    (results/benchmark/<model>/<chars>) layouts."""
    ensure_results_dir()

    benchmarks: List[Dict[str, Any]] = []
    for entry in _discover_benchmark_dirs():
        data = _load_json(entry["summary_file"])
        if not data:
            continue

        metadata = data.get("metadata", {})
        timestamp = metadata.get("timestamp", "")

        # New layout stores runs locally under runs/{success,failed}; legacy
        # benchmarks keep falling back to date-based trace archives.
        if entry["layout"] == "nested":
            local_runs = _run_files_in(entry["benchmark_dir"])
            trace_archives: List[Path] = []
            trace_archive_names = []
            run_count = len(local_runs)
        else:
            trace_archives = _trace_archives_for_timestamp(timestamp)
            trace_archive_names = [a.name for a in trace_archives]
            local_runs = []
            run_count = sum(len(list(a.glob("run_*.json"))) for a in trace_archives)

        benchmarks.append({
            "benchmark_id": entry["benchmark_id"],
            "benchmark_group": entry["benchmark_group"],
            "file_path": str(entry["summary_file"]),
            "timestamp": timestamp,
            **_benchmark_summary_fields(data),
            "metadata": metadata,
            "trace_archive_count": len(trace_archives) if entry["layout"] == "legacy" else run_count,
            "trace_archives": trace_archive_names,
            "layout": entry["layout"],
        })

    # Stable ordering: timestamp desc, then benchmark_id.
    benchmarks.sort(key=lambda b: (b.get("timestamp", ""), b.get("benchmark_id", "")), reverse=True)
    if offset:
        benchmarks = benchmarks[offset:]
    if limit is not None and limit >= 0:
        benchmarks = benchmarks[:limit]
    return benchmarks


def _resolve_benchmark_entry(benchmark_id: str) -> Optional[Dict[str, Any]]:
    """Resolve a benchmark_id to a discovery entry across layouts.

    Nested IDs are slash-form "<model>/<chars>"; legacy IDs are a single segment.
    A single-segment id is tried against both layouts (nested wins only if a
    matching chars dir exists under some model dir).
    """
    for entry in _discover_benchmark_dirs():
        if entry["benchmark_id"] == benchmark_id:
            return entry
    return None


def get_benchmark(benchmark_id: str) -> Optional[Dict[str, Any]]:
    """Load a benchmark summary and attach trace runs across layouts."""
    entry = _resolve_benchmark_entry(benchmark_id)
    if entry is None:
        return None

    data = _load_json(entry["summary_file"])
    if not data:
        return None

    metadata = data.get("metadata", {})
    timestamp = metadata.get("timestamp", "")

    if entry["layout"] == "nested":
        run_files = _run_files_in(entry["benchmark_dir"])
        trace_runs = [_run_metadata_from_file(p) for p in run_files]
        trace_archives: List[Dict[str, Any]] = []
        if trace_runs:
            # Present the local runs as a single pseudo-archive for the UI.
            trace_archives = [{
                "archive_id": f"{entry['benchmark_dir'].name}/runs",
                "date": _archive_date_from_timestamp(timestamp) or "",
                "path": str(entry["benchmark_dir"] / "runs"),
                "timestamp": trace_runs[0].get("timestamp", "") if trace_runs else "",
                "run_count": len(trace_runs),
                "success_rate": (
                    sum(int(r["success"]) for r in trace_runs) / len(trace_runs)
                    if trace_runs else 0.0
                ),
                "verified_rate": (
                    sum(int(r["verified_success"]) for r in trace_runs) / len(trace_runs)
                    if trace_runs else 0.0
                ),
                "avg_attempts_on_success": 0.0,
                "runs": trace_runs,
            }]
    else:
        trace_archives = [_summarize_trace_archive(p) for p in _trace_archives_for_timestamp(timestamp)]
        trace_runs = []
        for archive in trace_archives:
            trace_runs.extend(archive.get("runs", []))

    trace_runs.sort(key=lambda item: (item.get("timestamp", ""), item.get("run_id", "")))

    return {
        "benchmark_id": benchmark_id,
        "benchmark_group": entry["benchmark_group"],
        "layout": entry["layout"],
        "summary": data,
        "metadata": metadata,
        "worker_summaries": data.get("worker_summaries", []),
        "trace_archives": trace_archives,
        "trace_runs": trace_runs,
    }


def list_runs(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """List top-level run JSON files with metadata."""
    ensure_results_dir()
    runs: List[Dict[str, Any]] = []
    for f in sorted(RESULTS_DIR.glob("run_*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = normalize_run(json.load(fp), f.stem)
            runs.append({
                "run_id": data.get("experiment", {}).get("run_id", f.stem),
                "file_path": str(f),
                "timestamp": data.get("experiment", {}).get("timestamp", ""),
                "scenario_id": data.get("experiment", {}).get("scenario_id", ""),
                "success": _overall_success(data.get("result", {})),
                "total_attempts": data.get("result", {}).get("total_attempts", 0),
                "access_code": data.get("scenario", {}).get("access_code", ""),
                "generator": data.get("models", {}).get("generator", {}).get("name", ""),
                "victim": data.get("models", {}).get("victim", {}).get("name", ""),
                "benchmark_mode": data.get("experiment", {}).get("benchmark_mode", False),
            })
        except (json.JSONDecodeError, KeyError) as e:
            runs.append({
                "run_id": f.stem,
                "file_path": str(f),
                "scenario_id": "",
                "error": str(e),
                "benchmark_mode": False,
            })
    if offset:
        runs = runs[offset:]
    if limit is not None and limit >= 0:
        runs = runs[:limit]
    return runs


def list_all_runs_recursive() -> List[Dict[str, Any]]:
    """List every run JSON, including dated trace archives."""
    ensure_results_dir()
    runs: List[Dict[str, Any]] = []
    for f in sorted(RESULTS_DIR.rglob("run_*.json"), key=os.path.getmtime, reverse=True):
        if _is_benchmark_artifact(f):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = normalize_run(json.load(fp), f.stem)
            runs.append({
                "run_id": data.get("experiment", {}).get("run_id", f.stem),
                "file_path": str(f),
                "timestamp": data.get("experiment", {}).get("timestamp", ""),
                "scenario_id": data.get("experiment", {}).get("scenario_id", ""),
                "success": _overall_success(data.get("result", {})),
                "total_attempts": data.get("result", {}).get("total_attempts", 0),
                "access_code": data.get("scenario", {}).get("access_code", ""),
                "generator": data.get("models", {}).get("generator", {}).get("name", ""),
                "victim": data.get("models", {}).get("victim", {}).get("name", ""),
                "benchmark_mode": data.get("experiment", {}).get("benchmark_mode", False),
            })
        except (json.JSONDecodeError, KeyError) as e:
            runs.append({
                "run_id": f.stem,
                "file_path": str(f),
                "scenario_id": "",
                "error": str(e),
                "benchmark_mode": False,
            })
    return runs


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific run by run_id."""
    ensure_results_dir()
    candidates = list(RESULTS_DIR.glob("*.json")) + list(RESULTS_DIR.rglob("run_*.json"))
    seen: set[Path] = set()
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        if path in seen or _is_benchmark_artifact(path):
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = normalize_run(json.load(f), path.stem)
            if path.stem == run_id or data["experiment"]["run_id"] == run_id:
                return data
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return None


def upload_run(file_path: str) -> Dict[str, Any]:
    """Upload an external JSON file to results directory."""
    ensure_results_dir()
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(src, "r", encoding="utf-8") as f:
        data = normalize_run(json.load(f))

    run_id = data.get("experiment", {}).get("run_id", f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    safe_run_id = Path(run_id).name.replace(".json", "") or f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest = RESULTS_DIR / f"{safe_run_id}.json"

    if dest.exists():
        timestamp = datetime.now().strftime("%H%M%S%f")[:-3]
        dest = RESULTS_DIR / f"{safe_run_id}_{timestamp}.json"

    data["experiment"]["run_id"] = dest.stem
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {"run_id": dest.stem, "file_path": str(dest)}


def delete_run(run_id: str) -> bool:
    """Delete a top-level run JSON file."""
    for path in RESULTS_DIR.glob("*.json"):
        if path.stem == run_id:
            path.unlink()
            return True
    return False
