#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
: "${CALIB_DIR:?Set CALIB_DIR to the ImageNet training directory}"
: "${GENEVAL_META:?Set GENEVAL_META to GenEval evaluation_metadata.jsonl}"
OUT_TXT="${OUT_TXT:-$ROOT/outputs/calibration/janus_calibration.txt}"
mkdir -p "$(dirname "$OUT_TXT")"

CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PYTHON" "$ROOT/janus_calibration_analysis.py" \
  --model_path "${JANUS_MODEL:-deepseek-ai/Janus-Pro-7B}" \
  --prompts_file "$GENEVAL_META" \
  --num_prompts 100 \
  --image_token_num_per_image 576 \
  --janus_temperature 1.0 \
  --cfg_weight 5.0 \
  --calib_imgs_dir "$CALIB_DIR" \
  --calib_max_images 100 \
  --calib_recursive \
  --calib_exts jpg,jpeg,png,JPEG \
  --vq_similarity cosine \
  --heartbeat_interval 100 \
  --seed "${SEED:-42}" \
  --out_txt "$OUT_TXT"
