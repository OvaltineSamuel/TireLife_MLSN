#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"

cd "$ROOT_DIR"

if [ -f ".env.local" ]; then
  echo "Loading .env.local"
  set -a
  # shellcheck disable=SC1091
  source ".env.local"
  set +a
fi

if [ "$RELOAD" = "1" ]; then
  echo "Starting FastAPI at http://${HOST}:${PORT}"
  exec "$PYTHON_BIN" -m uvicorn backend.main:app --reload --host "$HOST" --port "$PORT"
fi

echo "Starting FastAPI at http://${HOST}:${PORT}"
exec "$PYTHON_BIN" -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
