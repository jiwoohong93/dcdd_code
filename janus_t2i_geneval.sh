#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-0,1}"
SEED="${SEED:-1}"
: "${META_JSONL:?Set META_JSONL to GenEval evaluation_metadata.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/janus_t2i_geneval/seed_${SEED}}"
TAGS="${TAGS:-}"
: "${CKPT_DIR:?Set CKPT_DIR to the trained DCDD checkpoint}"
BASE_MODEL="${BASE_MODEL:-black-forest-labs/FLUX.1-Fill-dev}"
JANUS_MODEL="${JANUS_MODEL:-deepseek-ai/Janus-Pro-7B}"
SIZE="${SIZE:-384}"
STEPS="${STEPS:-30}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-4}"
PARALLEL_SIZE="${PARALLEL_SIZE:-$((4 * PROMPT_BATCH_SIZE))}"
TEMP="${TEMP:-1.0}"
CFG="${CFG:-5.0}"
DTYPE="${DTYPE:-bf16}"
MAX_ITEMS="${MAX_ITEMS:-0}"

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" janus_t2i_geneval.py \
  --gpu 0 \
  --meta_jsonl "$META_JSONL" \
  --out_root "$OUT_ROOT" \
  --tags "$TAGS" \
  --model "$JANUS_MODEL" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --skip_existing \
  --ckpt_dir "$CKPT_DIR" \
  --base_model "$BASE_MODEL" \
  --height "$SIZE" --width "$SIZE" \
  --steps "$STEPS" \
  --parallel_size "$PARALLEL_SIZE" \
  --prompt_batch_size "$PROMPT_BATCH_SIZE" \
  --temperature "$TEMP" \
  --cfg_weight "$CFG" \
  --max_items "$MAX_ITEMS"
