#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-0}"
SEED="${SEED:-1}"
: "${META_JSON:?Set META_JSON to MJHQ meta_data.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/janus_t2i_mjhq/seed_${SEED}}"
CATEGORIES="${CATEGORIES:-}"
: "${CKPT_DIR:?Set CKPT_DIR to the trained DCDD checkpoint}"
BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-Fill-dev}"
JANUS_MODEL="${JANUS_MODEL:-deepseek-ai/Janus-Pro-7B}"
SIZE="${SIZE:-384}"
STEPS="${STEPS:-30}"
PARALLEL_SIZE="${PARALLEL_SIZE:-16}"
BATCH_SIZE_PROMPTS="${BATCH_SIZE_PROMPTS:-16}"
TEMP="${TEMP:-1.0}"
CFG="${CFG:-5.0}"
DTYPE="${DTYPE:-bf16}"
MAX_ITEMS="${MAX_ITEMS:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" janus_t2i_mjhq.py \
  --gpu 0 \
  --meta_json "$META_JSON" \
  --out_root "$OUT_ROOT" \
  --categories "$CATEGORIES" \
  --model "$JANUS_MODEL" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --skip_existing \
  --ckpt_dir "$CKPT_DIR" \
  --base_model "$BASE_MODEL" \
  --height "$SIZE" --width "$SIZE" \
  --steps "$STEPS" \
  --parallel_size "$PARALLEL_SIZE" \
  --batch_size_prompts "$BATCH_SIZE_PROMPTS" \
  --temperature "$TEMP" \
  --cfg_weight "$CFG" \
  --max_items "$MAX_ITEMS" \
  --num_shards "$NUM_SHARDS" \
  --shard_id "$SHARD_ID"
