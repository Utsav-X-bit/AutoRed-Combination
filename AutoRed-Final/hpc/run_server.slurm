#!/bin/bash
# =============================================================================
# AutoRed Backend Server — SLURM Launcher
# =============================================================================
# Usage: sbatch hpc/run_server.slurm
#
# Starts the FastAPI backend server with all models pre-loaded.
# The server listens on port 8001 and serves:
#   - REST API:  http://<node>:8001/api/*
#   - WebSocket: ws://<node>:8001/ws/run/{run_id}
#
# Port forwarding (run on your local machine BEFORE submitting this job):
#   ssh -L 8001:localhost:8001 username@nls.iastate.edu
# =============================================================================

#SBATCH --job-name=AutoRed_Server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err
#SBATCH --partition=gpu

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/nlsasfs/home/isea/isea13/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Offline mode (models pre-downloaded)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error

# Server configuration
export AUTORED_SERVER_MODE=1
export AUTORED_LOAD_MODELS=1
SERVER_PORT=8001
SERVER_HOST="0.0.0.0"

# =============================================================================
# Setup
# =============================================================================
mkdir -p logs results

echo "============================================="
echo "AutoRed Backend Server"
echo "============================================="
echo "Project Root : $PROJECT_ROOT"
echo "Host         : $SERVER_HOST"
echo "Port         : $SERVER_PORT"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================="

# =============================================================================
# Start Server
# =============================================================================
echo ""
echo "[SERVER] Starting FastAPI server on ${SERVER_HOST}:${SERVER_PORT}..."
echo "[SERVER] Models will be loaded on startup..."
echo ""

uvicorn server.main:app \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --log-level info \
    --access-log
