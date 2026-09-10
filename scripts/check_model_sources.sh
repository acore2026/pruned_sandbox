#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
model_root="${ORIGINAL_MODEL_ROOT:-${project_dir}/../sandbox-demo}"

required_files=(
  "models/whisper-models/whisper-large-v3/model.bin"
  "models/whisper-models/whisper-large-v3/config.json"
  "models/semantic-models/Qwen/Qwen2.5-0.5B-Instruct/model.safetensors"
  "models/semantic-models/Qwen/Qwen2.5-0.5B-Instruct/config.json"
  "yolo/assets/models/yolov8s-worldv2.pt"
  "yolo/assets/models/box0612.pt"
  "yolo/assets/models/toy.pt"
  "yolo/assets/models/bottles.pt"
)

for relative_path in "${required_files[@]}"; do
  full_path="${model_root}/${relative_path}"
  if [[ ! -s "${full_path}" ]]; then
    printf '缺少原模型文件: %s\n' "${full_path}" >&2
    exit 1
  fi
done

printf '原模型文件检查通过: %s\n' "${model_root}"
