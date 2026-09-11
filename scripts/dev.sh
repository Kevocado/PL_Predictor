#!/usr/bin/env bash
# dev.sh — one command to run both halves of the local dashboard (see
# README.md's "Run the dashboard" section, which this reproduces without
# needing two terminals). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d frontend/node_modules ]; then
  echo "[dev.sh] frontend/node_modules missing -- running npm install (first run only)..."
  (cd frontend && npm install)
fi

export PYTHONPATH="$(pwd)/src"
.venv/bin/uvicorn pl_predictor.api.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000 &
backend_pid=$!

cleanup() {
  echo "[dev.sh] stopping backend (pid $backend_pid)..."
  kill "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT

(cd frontend && npm run dev)
