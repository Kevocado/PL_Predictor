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

# A previous run's backend can outlive this script (e.g. the terminal was
# closed instead of Ctrl+C'd) and still hold port 8000, which uvicorn then
# refuses with "Address already in use" -- silently leaving Vite proxying
# to a dead/stale backend rather than the one this run just started.
# Freeing the port first makes every run start from a clean slate.
stale_pid="$(lsof -ti :8000 2>/dev/null || true)"
if [ -n "$stale_pid" ]; then
  echo "[dev.sh] port 8000 already in use (pid $stale_pid) -- stopping it..."
  kill "$stale_pid" 2>/dev/null || true
  sleep 1
fi

export PYTHONPATH="$(pwd)/src"
# `python -m uvicorn`, not the `.venv/bin/uvicorn` script directly -- that
# script's shebang hardcodes the venv's path at creation time, which breaks
# (bad interpreter) if the repo folder is ever renamed/moved after the venv
# was created. `python -m` re-resolves everything from the interpreter
# actually invoked, so it can't go stale this way.
.venv/bin/python -m uvicorn pl_predictor.api.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000 &
backend_pid=$!

cleanup() {
  echo "[dev.sh] stopping backend (pid $backend_pid)..."
  kill "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT

(cd frontend && npm run dev)
