#!/usr/bin/env bash
set -euo pipefail

# IBVOID release bootstrap:
# - prepares local model cache layout for Sub-Zero
# - optionally copies pre-downloaded models from ./models

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUB_ZERO_HOME="${SUB_ZERO_HOME:-$HOME/.sub-zero}"
MODEL_ROOT="${SUB_ZERO_HOME}/models"

mkdir -p "${MODEL_ROOT}"
mkdir -p "${MODEL_ROOT}/whisper"
mkdir -p "${MODEL_ROOT}/nllb"

echo "sub-zero bootstrap:"
echo "  model_root=${MODEL_ROOT}"

if [[ -d "${ROOT_DIR}/models" ]]; then
  echo "  source_models=${ROOT_DIR}/models (detected)"
  shopt -s nullglob
  for item in "${ROOT_DIR}"/models/*; do
    base="$(basename "${item}")"
    case "${base}" in
      whisper*|ggml*|*.pt|*.bin)
        cp -R "${item}" "${MODEL_ROOT}/whisper/" 2>/dev/null || true
        ;;
      nllb*|*distilled*|*ct2*)
        cp -R "${item}" "${MODEL_ROOT}/nllb/" 2>/dev/null || true
        ;;
      *)
        ;;
    esac
  done
  shopt -u nullglob
fi

echo "  done."
echo "  whisper_dir=${MODEL_ROOT}/whisper"
echo "  nllb_dir=${MODEL_ROOT}/nllb"
