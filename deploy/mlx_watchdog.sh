#!/bin/bash
# MLX watchdog — monitors model router and restarts if crashed.
# Designed for macOS (launchd). Adjust paths below.
set -euo pipefail

LOG_DIR="${LOG_DIR:-/tmp/isra-logs}"
INSTALL_DIR="${INSTALL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG="$LOG_DIR/watchdog.log"

mkdir -p "$LOG_DIR"

while true; do
  # Check if router is responding
  if ! curl -sf --max-time 5 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
    echo "$(date): Router not responding — restarting" >> "$LOG"
    pkill -f "model_router.py" 2>/dev/null || true
    sleep 2
    nohup "$PYTHON_BIN" "$INSTALL_DIR/model_router.py" --host 0.0.0.0 --port 8080 \
      >> "$LOG_DIR/router.log" 2>&1 &
    sleep 10
  fi
  sleep 30
done
