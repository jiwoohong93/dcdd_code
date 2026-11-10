#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-0,1}"
NPROC="${NPROC:-2}"
SEED="${SEED:-1}"
: "${CKPT_DIR:?Set CKPT_DIR to the trained DCDD checkpoint}"
BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-Fill-dev}"
: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet directory}"
OUT_PATH="${OUT_PATH:-$ROOT/outputs/janus_reconstruction/seed_${SEED}}"
STEPS="${STEPS:-30}"
SIZE="${SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
JANUS_MODEL="${JANUS_MODEL:-deepseek-ai/Janus-Pro-7B}"
MAX_ITEMS="${MAX_ITEMS:-0}"
CALIBRATION_FILE="${CALIBRATION_FILE:-}"
JANUS_SIMILARITY="${JANUS_SIMILARITY:-cosine}"
JANUS_TEMPERATURE="${JANUS_TEMPERATURE:-1.0000}"
JANUS_COSINE_SCALE="${JANUS_COSINE_SCALE:-29.0381}"
JANUS_LABEL_SMOOTHING_EPS="${JANUS_LABEL_SMOOTHING_EPS:-0.0100}"
if [[ -n "$CALIBRATION_FILE" ]]; then
  if [[ ! -f "$CALIBRATION_FILE" ]]; then
    echo "Calibration file not found: $CALIBRATION_FILE" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$CALIBRATION_FILE"
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" janus_reconstruction.py \
  --ckpt_dir "$CKPT_DIR" \
  --base_model "$BASE_MODEL" \
  --dataroot "$DATA_ROOT" \
  --out_dir "$OUT_PATH" \
  --height "$SIZE" --width "$SIZE" \
  --steps "$STEPS" --seed "$SEED" --guidance_scale 1.0 --dtype bf16 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --skip_existing \
  --max_items "$MAX_ITEMS" \
  --global_prompt "" \
  --janus_model "$JANUS_MODEL" \
  --janus_similarity "$JANUS_SIMILARITY" \
  --janus_temperature "$JANUS_TEMPERATURE" \
  --janus_cosine_scale "$JANUS_COSINE_SCALE" \
  --janus_label_smoothing_eps "$JANUS_LABEL_SMOOTHING_EPS"
