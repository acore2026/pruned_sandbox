#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Missing .venv. Run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

# CTranslate2 loads CUDA libraries directly. Add the NVIDIA wheel directories
# from this virtual environment so local GPU inference behaves like the image.
SITE_PACKAGES=$($PYTHON -c 'import site; print(site.getsitepackages()[0])')
CUDA_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH:$LD_LIBRARY_PATH"
else
    LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH

# The default is a lightweight smoke-test mode. Set SANDBOX_REAL_MODELS=true
# to load the locally downloaded Whisper and Qwen weights.
: "${SANDBOX_REAL_MODELS:=false}"
if [ "$SANDBOX_REAL_MODELS" = "true" ]; then
    : "${ASR_ENABLED:=true}"
    : "${ASR_MODEL:=$PROJECT_DIR/models/whisper-models/whisper-large-v3}"
    : "${ASR_DEVICE:=cuda}"
    : "${ASR_COMPUTE_TYPE:=float16}"
    : "${INTENT_BACKEND:=hybrid}"
    : "${INTENT_MODEL:=$PROJECT_DIR/models/semantic-models/Qwen/Qwen2.5-0.5B-Instruct}"
else
    : "${ASR_ENABLED:=false}"
    : "${INTENT_BACKEND:=rules}"
fi
: "${YOLO_ENABLED:=false}"
: "${LOG_DIR:=$PROJECT_DIR/logs}"
: "${LOG_MAX_BYTES:=10485760}"
: "${LOG_BACKUP_COUNT:=5}"
export SANDBOX_REAL_MODELS ASR_ENABLED ASR_MODEL ASR_DEVICE ASR_COMPUTE_TYPE
export INTENT_BACKEND INTENT_MODEL YOLO_ENABLED
export LOG_DIR LOG_MAX_BYTES LOG_BACKUP_COUNT

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
child_pids=""

stop_children() {
    if [ -n "$child_pids" ]; then
        kill $child_pids 2>/dev/null || true
        wait $child_pids 2>/dev/null || true
    fi
}
trap stop_children EXIT INT TERM

"$PYTHON" -m services.asr.main &
child_pids="$child_pids $!"
"$PYTHON" -m services.intent.main &
child_pids="$child_pids $!"
"$PYTHON" -m services.sandbox.main &
child_pids="$child_pids $!"

echo "Sandbox management is starting at http://127.0.0.1:28501"
echo "Sandbox user plane is starting at http://127.0.0.1:28502 (Ctrl-C to stop)."
echo "Discovery ASR is starting at http://127.0.0.1:9004; runtime ASR is internal at 127.0.0.1:9005; Intent is internal at 127.0.0.1:8011."
echo "Rolling logs are stored in $LOG_DIR (max $LOG_MAX_BYTES bytes, $LOG_BACKUP_COUNT backups per service)."
wait
