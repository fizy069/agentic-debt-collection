#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

PID_TEMPORAL=""
PID_WORKER=""
PID_API=""

SKIP_INSTALL=false
OPEN_UI=false
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=true ;;
    --open-ui)     OPEN_UI=true ;;
  esac
done

if ! command -v temporal &>/dev/null; then
  echo "ERROR: Temporal CLI not found in PATH."
  echo "Install with: winget install Temporal.TemporalCLI"
  exit 1
fi

VENV_DIR="$PROJECT_ROOT/.venv"
if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
  PYTHON="$VENV_DIR/Scripts/python.exe"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
  PYTHON="$VENV_DIR/bin/python"
else
  echo "Creating virtual environment in .venv..."
  python -m venv "$VENV_DIR"
  if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
    PYTHON="$VENV_DIR/Scripts/python.exe"
  else
    PYTHON="$VENV_DIR/bin/python"
  fi
fi

if [[ "$SKIP_INSTALL" == false ]]; then
  echo "Installing/updating dependencies..."
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -e .
fi

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Using environment file: .env"
else
  echo "WARNING: .env not found. Copy .env.example to .env before running."
fi

echo "Using python: $PYTHON"
echo ""

cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "$PID_TEMPORAL" "$PID_WORKER" "$PID_API"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "All processes stopped."
}
trap cleanup EXIT INT TERM

echo "Log files:"
echo "  Temporal: $LOG_DIR/temporal.log"
echo "  Worker:   $LOG_DIR/worker.log"
echo "  API:      $LOG_DIR/api.log"
echo ""

echo "Starting Temporal server..."
temporal server start-dev > >(tee -a "$LOG_DIR/temporal.log") 2>&1 &
PID_TEMPORAL=$!
sleep 3

echo "Starting worker..."
"$PYTHON" -m app.worker > >(tee -a "$LOG_DIR/worker.log") 2>&1 &
PID_WORKER=$!
sleep 2

echo "Starting API server..."
"$PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > >(tee -a "$LOG_DIR/api.log") 2>&1 &
PID_API=$!
sleep 1

if [[ "$OPEN_UI" == true ]]; then
  start "http://localhost:8233" 2>/dev/null || xdg-open "http://localhost:8233" 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  Dev stack running"
echo "  Temporal UI:  http://localhost:8233"
echo "  API docs:     http://127.0.0.1:8000/docs"
echo "  Test console: http://127.0.0.1:8000/test"
echo "  Press Ctrl+C to stop all processes"
echo "============================================"
echo ""

wait
