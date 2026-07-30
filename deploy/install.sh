#!/bin/bash
# Install ISRA orchestrator + model router as launchd services on macOS.
# Usage: ./install.sh /path/to/mlx-venv /path/to/models
set -euo pipefail

PYTHON_VENV="${1:?Usage: $0 <python-venv-bin> <model-path> <install-dir> <log-dir>}"
MODEL_PATH="${2:?}"
INSTALL_DIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="${4:-/tmp/isra-logs}"

PYTHON_BIN="$PYTHON_VENV/python3"
PYTHON_VENV_BIN="$(dirname "$PYTHON_BIN")"

mkdir -p "$LOG_DIR"
mkdir -p "$HOME/Library/LaunchAgents"

render_template() {
  local template="$1" output="$2"
  sed \
    -e "s|{{PYTHON_BIN}}|$PYTHON_BIN|g" \
    -e "s|{{PYTHON_VENV_BIN}}|$PYTHON_VENV_BIN|g" \
    -e "s|{{INSTALL_DIR}}|$INSTALL_DIR|g" \
    -e "s|{{MODEL_PATH}}|$MODEL_PATH|g" \
    -e "s|{{HOME_DIR}}|$HOME|g" \
    -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
    "$template" > "$output"
}

# Generate plist files from templates
render_template "$INSTALL_DIR/deploy/local.model-router.plist.template" \
  "$HOME/Library/LaunchAgents/local.model-router.plist"
render_template "$INSTALL_DIR/deploy/local.isra-orchestrator.plist.template" \
  "$HOME/Library/LaunchAgents/local.isra-orchestrator.plist"
render_template "$INSTALL_DIR/deploy/local.qwen3-a3b.plist.template" \
  "$HOME/Library/LaunchAgents/local.qwen3-a3b.plist"

# Load services
launchctl unload "$HOME/Library/LaunchAgents/local.qwen3-a3b.plist" 2>/dev/null || true
launchctl unload "$HOME/Library/LaunchAgents/local.model-router.plist" 2>/dev/null || true
launchctl unload "$HOME/Library/LaunchAgents/local.isra-orchestrator.plist" 2>/dev/null || true

launchctl load "$HOME/Library/LaunchAgents/local.qwen3-a3b.plist"
echo "Loaded qwen3-a3b MLX server (port 8081). Waiting for startup..."
sleep 30

launchctl load "$HOME/Library/LaunchAgents/local.model-router.plist"
echo "Loaded model router (port 8080)."
sleep 5

launchctl load "$HOME/Library/LaunchAgents/local.isra-orchestrator.plist"
echo "Loaded ISRA orchestrator (port 8083)."
sleep 3

echo ""
echo "=== Services started ==="
echo "  MLX backend:  http://localhost:8081"
echo "  Router:       http://localhost:8080"
echo "  ISRA:         http://localhost:8083"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo "Test: curl http://localhost:8083/health"
