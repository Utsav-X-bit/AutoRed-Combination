#!/bin/bash
# =============================================================================
# AutoRed All-in-One — Backend + Frontend on HPC
# =============================================================================
# Usage: sbatch hpc/run_all.slurm
#
# Starts both the FastAPI backend (port 8001) and the Vite frontend (port 3000)
# on the same GPU node. Use this when you want everything running on the cluster.
#
# Port forwarding (run on your local machine BEFORE submitting this job):
#   ssh -L 8001:localhost:8001 -L 3000:localhost:3000 username@nls.iastate.edu
# =============================================================================

#SBATCH --job-name=AutoRed_All
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err
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
export AUTORED_SERVER_MODE=1
export AUTORED_LOAD_MODELS=1

# =============================================================================
# Setup
# =============================================================================
mkdir -p logs results

echo "============================================="
echo "AutoRed All-in-One (Backend + Frontend)"
echo "============================================="
echo "Project Root : $PROJECT_ROOT"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================="

# =============================================================================
# Start Backend Server
# =============================================================================
echo ""
echo "[START] Backend server on port 8001..."
uvicorn server.main:app --host 0.0.0.0 --port 8001 --log-level info &
BACKEND_PID=$!

# Wait for backend to initialize and load models
echo "[WAIT] Waiting for backend to start and load models..."
sleep 30

# Verify backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[OK] Backend server is running (PID: $BACKEND_PID)"
else
    echo "[ERROR] Backend server failed to start. Check logs."
    exit 1
fi

# =============================================================================
# Start Frontend Dev Server
# =============================================================================
echo ""
echo "[START] Frontend dev server on port 3000..."
cd ui

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "[INSTALL] Installing Node.js dependencies..."
    npm install
fi

npx vite --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

# Verify frontend is running
if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "[OK] Frontend server is running (PID: $FRONTEND_PID)"
else
    echo "[ERROR] Frontend server failed to start. Check logs."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "============================================="
echo "AutoRed All-in-One Running"
echo "============================================="
echo "Backend : http://0.0.0.0:8001"
echo "Frontend: http://0.0.0.0:3000"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Port forwarding (run on local machine):"
echo "  ssh -L 8001:localhost:8001 -L 3000:localhost:3000 username@nls.iastate.edu"
echo "============================================="

# =============================================================================
# Cleanup on Exit
# =============================================================================
cleanup() {
    echo ""
    echo "[STOP] Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "[STOP] Done."
}
trap cleanup EXIT INT TERM

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
