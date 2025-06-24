#!/usr/bin/env bash

# Exit on error, undefined variables, or pipe failures
set -euo pipefail

# --- Configuration ---
readonly LOG_DIR="/app/emissions_logs"
readonly MONITOR_SCRIPT="/app/monitor.py"
readonly MONITOR_LOG="${LOG_DIR}/system_metrics.csv"

# --- Utility Functions ---
log() {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

# --- Shutdown Handler ---
# This function will be called when the script exits.
cleanup() {
    log "Shutdown signal received."
    if [[ -n "${MONITOR_PID:-}" ]]; then
        log "Stopping Python monitor (PID: ${MONITOR_PID})..."
        # Kill the process and ignore errors if it's already gone
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
    log "Cleanup complete. Exiting."
}

# Trap the EXIT signal to run the cleanup function, ensuring the monitor stops.
trap cleanup EXIT

# --- Main Script ---

# Unbuffered Python for real-time logs in the monitor script
export PYTHONUNBUFFERED=1


# Clean up any previous run artifacts
log "Cleaning up previous run artifacts..."
rm -f "${LOG_DIR}"/* 2>/dev/null || true
mkdir -p "${LOG_DIR}"
log "Cleanup finished."

# Ensure Hugging Face token is set
if [[ -z "${HF_TOKEN:-}" ]]; then
  log "❌ HF_TOKEN is not set; cannot download private models."
  exit 1
fi

# Launch the Python monitoring script in the background
if [[ -f "$MONITOR_SCRIPT" ]]; then
    log "Starting Python system monitor..."
    python3 "$MONITOR_SCRIPT" &
    MONITOR_PID=$!
    log "✅ Monitor started in background with PID: ${MONITOR_PID}"
else
    log "⚠️ Warning: Monitoring script not found at ${MONITOR_SCRIPT}. Skipping."
fi

# 5) Verify Phoenix key
if [[ -z "${PHOENIX_API_KEY:-}" ]]; then
  echo "❌ PHOENIX_API_KEY is not set in the container!"
  exit 1
else
  echo "✅ PHOENIX_API_KEY is set (length=${#PHOENIX_API_KEY})"
fi


# Launch the vLLM server in the foreground

log "🚀 Starting vLLM server..."
uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 600
