#!/usr/bin/env bash
# =============================================================================
#  RAG Performance Prediction Framework - Linux launcher
#  Starts the FastAPI backend (port 8000) and the Vite frontend (port 3000).
#  Runs entirely from the framework/ folder (no dependency on rq1/rq3/rq4).
#
#  Usage:
#    chmod +x run-linux.sh
#    ./run-linux.sh
# =============================================================================
set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$FRAMEWORK_DIR"

echo "========================================"
echo "  RAG Performance Prediction Framework"
echo "========================================"

# --- Locate a Python interpreter (prefer the project venv) -------------------
if [ -x "$FRAMEWORK_DIR/../.venv/bin/python" ]; then
  PYTHON="$FRAMEWORK_DIR/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi
echo "[python] $PYTHON"

# --- Ensure bundled data exists ----------------------------------------------
if [ ! -f "$FRAMEWORK_DIR/backend/data/rq3_models.json" ]; then
  echo "[setup] Bundled data not found. Running sync_data.py..."
  "$PYTHON" "$FRAMEWORK_DIR/sync_data.py"
fi

# --- Backend dependencies -----------------------------------------------------
echo "[1/3] Installing backend dependencies..."
"$PYTHON" -m pip install -r "$FRAMEWORK_DIR/backend/requirements.txt" -q

# --- Start backend ------------------------------------------------------------
echo "[2/3] Starting backend on http://localhost:8000 ..."
( cd "$FRAMEWORK_DIR/backend" && "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload ) &
BACKEND_PID=$!

# Stop the backend when this script exits (Ctrl+C included).
cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 3

# --- Frontend -----------------------------------------------------------------
cd "$FRAMEWORK_DIR/frontend"
if [ ! -d node_modules ]; then
  echo "[setup] Installing frontend dependencies..."
  npm install --silent
fi

echo "[3/3] Starting frontend on http://localhost:3000 ..."
echo ""
echo "  Backend:  http://localhost:8000  (docs at /docs)"
echo "  Frontend: http://localhost:3000"
echo "  Press Ctrl+C to stop."
echo ""

# Open the browser (best-effort, non-blocking).
( sleep 4 && command -v xdg-open >/dev/null 2>&1 && xdg-open "http://localhost:3000" >/dev/null 2>&1 ) &

npm run dev
