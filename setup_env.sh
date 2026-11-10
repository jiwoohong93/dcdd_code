#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT/.conda}"
CONDA="${CONDA:-$(command -v conda || true)}"

find_python() {
  local candidate
  for candidate in "$ENV_DIR/bin/python" "$ENV_DIR"/bin/python3.*; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(find_python || true)"
if [[ -z "$PYTHON" ]]; then
  if [[ -z "$CONDA" ]]; then
    echo "Conda was not found. Set CONDA=/path/to/conda or create $ENV_DIR manually." >&2
    exit 1
  fi
  "$CONDA" create -y -p "$ENV_DIR" python=3.10 pip
  PYTHON="$(find_python)"
fi

"$PYTHON" -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
"$PYTHON" -m pip install -r "$ROOT/requirements-dcdd.txt"
if [[ "${INSTALL_WANDB:-0}" == "1" ]]; then
  "$PYTHON" -m pip install -r "$ROOT/requirements-wandb.txt"
fi

PYTHONPATH="$ROOT:${PYTHONPATH:-}" "$PYTHON" -c \
  "import torch, diffusers, transformers, janus; print(torch.__version__, diffusers.__version__, transformers.__version__)"
