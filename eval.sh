#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
: "${GT_FOLDER:?Set GT_FOLDER to the reference-image directory}"
: "${GEN_FOLDER:?Set GEN_FOLDER to the generated-image directory}"

CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PYTHON" "$ROOT/eval.py" \
  --gt_folder "$GT_FOLDER" \
  --gen_folder "$GEN_FOLDER" \
  --size 384 \
  --batch_size 32 \
  --workers 8
