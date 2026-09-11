#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
asr_model_source="${ASR_MODEL_SOURCE:-${project_dir}/models/whisper-models/whisper-large-v3}"
intent_model_source="${INTENT_MODEL_SOURCE:-${project_dir}/models/semantic-models/Qwen/Qwen2.5-0.5B-Instruct}"
yolo_model_source="${YOLO_MODEL_SOURCE:-${project_dir}/../compute/yolo/assets/models}"

required_files=(
  "${asr_model_source}/model.bin"
  "${asr_model_source}/config.json"
  "${intent_model_source}/model.safetensors"
  "${intent_model_source}/config.json"
  "${yolo_model_source}/yolov8s-worldv2.pt"
)

for full_path in "${required_files[@]}"; do
  if [[ ! -s "${full_path}" ]]; then
    printf '缺少模型文件: %s\n' "${full_path}" >&2
    exit 1
  fi
done

printf '模型文件检查通过：Whisper=%s，Qwen=%s，YOLO=%s\n' \
  "${asr_model_source}" "${intent_model_source}" "${yolo_model_source}"
