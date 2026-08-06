import os
# MUST set BEFORE any experiment imports to prevent double model loading
os.environ["AUTORED_SERVER_MODE"] = "1"
# Suppress transformer warnings that clutter logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import json
import tempfile
import asyncio
import logging
import html as html_lib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autored_server")

from .file_manager import (
    list_runs,
    list_all_runs_recursive,
    list_benchmarks,
    get_benchmark,
    list_trace_archives,
    get_run,
    upload_run,
    delete_run,
)
from .websocket import ws_manager

try:
    from .models_server import server_models
except ImportError as exc:
    logger.warning("[SERVER] ML dependencies unavailable: %s", exc)

    class UnavailableModelsManager:
        is_loaded = False
        load_error = str(exc)
        models: Dict[str, Any] = {}
        tokenizers: Dict[str, Any] = {}

        def load_all(self):
            raise RuntimeError(self.load_error)

        def get_status(self):
            return {
                "ready": False,
                "error": self.load_error,
                **{
                    name: {"loaded": False, "name": "unavailable", "load_time": 0}
                    for name in ("victim", "generator", "judge", "extractor")
                },
            }

    server_models = UnavailableModelsManager()


# ─── Startup / Shutdown ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, keep in memory across runs."""
    if os.environ.get("AUTORED_LOAD_MODELS", "1") == "1":
        print("[SERVER] Starting up — loading models...")
        try:
            load_times = server_models.load_all()
            total = sum(load_times.values())
            print(f"[SERVER] ✓ All models loaded in {total:.1f}s")
        except Exception as exc:
            logger.exception("[SERVER] Model loading failed; run history remains available")
            server_models.load_error = str(exc)
    else:
        print("[SERVER] Starting without models (AUTORED_LOAD_MODELS=0)")
    yield
    print("[SERVER] Shutting down — clearing models...")
    server_models.models.clear()
    server_models.tokenizers.clear()


app = FastAPI(title="AutoRed Web UI", version="1.0.0", lifespan=lifespan)

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track running state
_run_lock = asyncio.Lock()
_is_running = False


@app.get("/api/run/status")
def api_run_status():
    """Check if a run is currently in progress."""
    return {"running": _is_running}


# ─── Run Endpoints ──────────────────────────────────────────

@app.get("/api/runs")
def api_list_runs(limit: Optional[int] = Query(default=None, ge=0), offset: int = Query(default=0, ge=0)):
    """List top-level past runs with optional pagination."""
    return list_runs(limit=limit, offset=offset)


@app.get("/api/runs/all")
def api_list_all_runs():
    """List all run JSON files, including dated benchmark trace archives."""
    return list_all_runs_recursive()


@app.get("/api/run/{run_id}")
def api_get_run(run_id: str):
    """Get a specific run by ID."""
    logger.info(f"[API] GET /api/run/{run_id}")
    run = get_run(run_id)
    if not run:
        logger.warning(f"[API] Run {run_id} not found")
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    logger.info(f"[API] Returning run {run_id}: attempts={len(run.get('attempts', []))}, result={run.get('result')}")
    return run


@app.get("/api/benchmarks")
def api_list_benchmarks(limit: Optional[int] = Query(default=None, ge=0), offset: int = Query(default=0, ge=0)):
    """List benchmark summary folders with optional pagination."""
    return list_benchmarks(limit=limit, offset=offset)


@app.get("/api/benchmarks/{benchmark_id:path}")
def api_get_benchmark(benchmark_id: str):
    """Get benchmark summary plus associated trace runs.

    ``benchmark_id`` uses the path converter so nested IDs of the form
    "<model>/<chars>" (new results_layout.py tree) match across the slash.
    Legacy single-segment IDs match unchanged.
    """
    benchmark = get_benchmark(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")
    return benchmark


@app.get("/api/trace-archives")
def api_list_trace_archives():
    """List all dated trace archives."""
    return list_trace_archives()


@app.post("/api/runs/upload")
async def api_upload_run(file: UploadFile = File(...)):
    """Upload an external run JSON file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = upload_run(tmp_path)
    os.unlink(tmp_path)
    return result


@app.delete("/api/run/{run_id}")
def api_delete_run(run_id: str):
    """Delete a run."""
    if not delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"deleted": run_id}


# ─── Model Status Endpoints ─────────────────────────────────

@app.get("/api/models/status")
def api_model_status():
    """Get model load status from server."""
    return server_models.get_status()


# ─── Experiment Endpoints ───────────────────────────────────

@app.post("/api/run")
async def api_start_run(
    run_id: Optional[str] = Query(default=None, description="Client-generated run_id for WebSocket routing"),
    scenario_id: str = Query(default="", description="Specific scenario ID (optional)"),
    max_attempts: int = Query(default=20, description="Maximum number of attempts"),
):
    """Start a new experiment run with pre-loaded models and live WebSocket streaming.

    Client MUST connect WebSocket to /ws/run/{run_id} BEFORE calling this endpoint.
    If client provides run_id query param, server uses it (ensures WebSocket routing works).
    Returns immediately; experiment runs in background.
    """
    if not server_models.is_loaded:
        logger.warning("[SERVER] Models not loaded yet, rejecting run request")
        raise HTTPException(status_code=503, detail="Models not loaded yet. Server may be starting up.")

    global _is_running
    if _is_running or _run_lock.locked():
        raise HTTPException(status_code=409, detail="Another experiment is already running.")

    # Use client-provided run_id if available (critical for WebSocket routing!)
    if run_id and run_id.startswith("run_"):
        actual_run_id = run_id
        logger.info(f"[SERVER] Using client-provided run_id: {actual_run_id}")
    else:
        actual_run_id = f"run_{__import__('time').time():.0f}"
        logger.info(f"[SERVER] Generated server run_id: {actual_run_id}")

    scenario_id = scenario_id.strip()
    logger.info(f"[SERVER] Starting experiment run_id={actual_run_id}, scenario_id={scenario_id or 'random'}, max_attempts={max_attempts}")
    logger.info(f"[SERVER] WebSocket connections for {actual_run_id}: {len(ws_manager._connections.get(actual_run_id, set()))}")

    _is_running = True

    # Start experiment in background
    asyncio.create_task(
        _run_experiment_task(actual_run_id, scenario_id or None, max_attempts)
    )

    logger.info(f"[SERVER] Background task created for {actual_run_id}, returning immediately to client")
    return {
        "run_id": actual_run_id,
        "status": "started",
        "message": f"Experiment started. Connect WebSocket to /ws/run/{actual_run_id} for live updates.",
    }


async def _run_experiment_task(run_id: str, scenario_id: Optional[str], max_attempts: int):
    """Background task: run experiment, stream via WebSocket."""
    global _is_running
    logger.info(f"[SERVER] _run_experiment_task acquired lock for {run_id}")
    async with _run_lock:
        _is_running = True
        logger.info(f"[SERVER] Experiment {run_id} beginning execution (models are pre-loaded)")
        try:
            from .experiment_server import run_experiment_server
            result = await run_experiment_server(
                run_id=run_id,
                scenario_id=scenario_id,
                max_attempts=max_attempts,
            )
            total_attempts = result.get("result", {}).get("total_attempts", 0)
            result_info = result.get("result", {})
            success = any(
                bool(result_info.get(key))
                for key in ("ground_truth_success", "extractor_success", "verified_success")
            )
            logger.info(f"[SERVER] ✓ Experiment {run_id} COMPLETED: attempts={total_attempts}, success={success}")
        except Exception as e:
            logger.error(f"[SERVER] ✗ Experiment {run_id} FAILED with exception: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            await ws_manager.send_run_complete(run_id, {"error": str(e), "experiment": {"run_id": run_id}})
        finally:
            _is_running = False
            logger.info(f"[SERVER] _run_experiment_task finished for {run_id}, _is_running=False")


# ─── Export Endpoints ───────────────────────────────────────

@app.get("/api/export/{run_id}/json")
def api_export_json(run_id: str):
    """Export run as JSON."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.get("/api/export/{run_id}/csv")
def api_export_csv(run_id: str):
    """Export run attempts as CSV."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    import csv
    import io

    output = io.StringIO()
    if not run.get("attempts"):
        output.write("no_attempts\n")
    else:
        fieldnames = [
            "attempt", "strategy", "attack", "judge_decision",
            "ground_truth_found", "extractor_match", "best_candidate",
            "verification_success"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for a in run["attempts"]:
            row = {
                "attempt": a.get("attempt_number"),
                "strategy": a.get("generator", {}).get("strategy"),
                "attack": a.get("generator", {}).get("generated_attack", "")[:100],
                "judge_decision": a.get("judge", {}).get("decision"),
                "ground_truth_found": a.get("ground_truth_found"),
                "extractor_match": a.get("extractor_match"),
                "best_candidate": a.get("extractor", {}).get("best_candidate"),
                "verification_success": a.get("verification", {}).get("success"),
            }
            writer.writerow(row)

    from fastapi.responses import Response
    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={run_id}.csv"})


@app.get("/api/export/{run_id}/html")
def api_export_html(run_id: str):
    """Export run as styled HTML report."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    def esc(value: Any) -> str:
        return html_lib.escape(str(value if value is not None else ""))

    result = run.get("result", {})
    success = any(
        bool(result.get(key))
        for key in ("ground_truth_success", "extractor_success", "verified_success")
    )
    success_badge = f'<span class="badge {"badge-green" if success else "badge-red"}">{"✓" if success else "✗"}</span>'

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AutoRed Report: {esc(run_id)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.2rem; margin-top: 1.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #e2e8f0; padding: 0.5rem; text-align: left; font-size: 0.875rem; }}
th {{ background: #f1f5f9; }}
.badge {{ display: inline-block; padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }}
.badge-green {{ background: #dcfce7; color: #166534; }}
.badge-red {{ background: #fee2e2; color: #991b1b; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; }}
pre {{ background: #f1f5f9; padding: 0.75rem; border-radius: 0.375rem; overflow-x: auto; font-size: 0.8rem; }}
</style></head><body>
<h1>AutoRed Report: {esc(run_id)}</h1>
<p>Timestamp: {esc(run.get('experiment', {}).get('timestamp', 'N/A'))}</p>
<p>Scenario: {esc(run.get('experiment', {}).get('scenario_id', 'N/A'))}</p>
<p>Version: {esc(run.get('experiment', {}).get('experiment_version', 'N/A'))}</p>
<p>Git: {esc(run.get('experiment', {}).get('git_commit', 'N/A'))}</p>

<h2>Result</h2>
<div class="card">
<p>Overall: {success_badge}</p>
<p>Ground Truth: <span class="badge {'badge-green' if run.get('result', {}).get('ground_truth_success') else 'badge-red'}">{'✓' if run.get('result', {}).get('ground_truth_success') else '✗'}</span></p>
<p>Extractor: <span class="badge {'badge-green' if run.get('result', {}).get('extractor_success') else 'badge-red'}">{'✓' if run.get('result', {}).get('extractor_success') else '✗'}</span></p>
<p>Verifier: <span class="badge {'badge-green' if run.get('result', {}).get('verified_success') else 'badge-red'}">{'✓' if run.get('result', {}).get('verified_success') else '✗'}</span></p>
<p>Attempts: {esc(run.get('result', {}).get('total_attempts', 0))}</p>
</div>

<h2>Attempts</h2>
<table>
<tr><th>#</th><th>Strategy</th><th>Judge</th><th>GT Found</th><th>Extractor</th><th>Best Candidate</th></tr>
"""
    for a in run.get("attempts", []):
        gt = '<span class="badge badge-green">✓</span>' if a.get("ground_truth_found") else '<span class="badge badge-red">✗</span>'
        ext = '<span class="badge badge-green">✓</span>' if a.get("extractor_match") else '<span class="badge badge-red">✗</span>'
        html += (
            f'<tr><td>{esc(a.get("attempt_number"))}</td>'
            f'<td>{esc(a.get("generator", {}).get("strategy"))}</td>'
            f'<td>{esc(a.get("judge", {}).get("decision"))}</td>'
            f'<td>{gt}</td><td>{ext}</td>'
            f'<td>{esc(a.get("extractor", {}).get("best_candidate", "N/A"))}</td></tr>\n'
        )

    html += """</table></body></html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


# ─── WebSocket Endpoint ─────────────────────────────────────

@app.websocket("/ws/run/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    logger.info(f"[WS] Incoming WebSocket connection request for run_id={run_id}")
    await ws_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, websocket)
        logger.info(f"[WS] WebSocket disconnected for run_id={run_id}")
