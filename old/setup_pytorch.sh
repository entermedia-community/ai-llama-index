#!/usr/bin/env bash
set -euo pipefail

# setup_pytorch.sh
# Create a Python venv (if missing) and install PyTorch + transformers + helpers.
# Usage:
#   bash scripts/setup_pytorch.sh [--venv .venv] [--cuda auto|cpu|cu118|cu117] [--nightly]
# Examples:
#   bash scripts/setup_pytorch.sh --venv .venv --cuda auto
#   bash scripts/setup_pytorch.sh --venv .venv --cuda cpu
#   bash scripts/setup_pytorch.sh --venv .venv --cuda cu118 --nightly

VENV_DIR=.venv
CUDA="auto"
NIGHTLY="no"

usage() {
  cat <<EOF
Usage: $0 [--venv <dir>] [--cuda auto|cpu|cu118|cu117] [--nightly]

Options:
  --venv   Path to virtualenv directory to create/use (default: .venv)
  --cuda   GPU variant to install. "auto" detects nvidia-smi and picks cu118 by default.
           Use "cpu" to force CPU-only wheels, or specify a CUDA tag like cu118.
  --nightly  Install transformers from the GitHub repo (latest)

This script will:
  - create a python venv if missing
  - activate it
  - upgrade pip/setuptools/wheel
  - install PyTorch (CPU or CUDA wheels)
  - install transformers, safetensors, accelerate
  - run a small import check
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"; shift 2;;
    --cuda)
      CUDA="$2"; shift 2;;
    --nightly)
      NIGHTLY="yes"; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "[setup] venv: $VENV_DIR, cuda: $CUDA, nightly: $NIGHTLY"

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] Creating venv in $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[setup] Using Python: $(which python3) ($(python3 --version))"

# Upgrade packaging
pip install -U pip setuptools wheel

# Decide CUDA variant
if [[ "$CUDA" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] nvidia-smi found: defaulting to cu118. If this is wrong, re-run with --cuda cpu or --cuda cu117/cu116"
    CUDA=cu118
  else
    echo "[setup] No GPU detected via nvidia-smi. Installing CPU-only PyTorch."
    CUDA=cpu
  fi
fi

# Install PyTorch
if [[ "$CUDA" == "cpu" ]]; then
  echo "[setup] Installing CPU-only PyTorch wheels"
  pip install --no-cache-dir --upgrade "torch" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
else
  echo "[setup] Installing PyTorch wheels for $CUDA"
  # Map common shorthand to PyTorch index url (cu118 etc)
  # Note: User should pick the correct CUDA version for their GPU. This uses the official PyTorch index URL.
  INDEX_URL="https://download.pytorch.org/whl/${CUDA}"
  pip install --no-cache-dir --upgrade "torch" "torchvision" "torchaudio" --index-url ${INDEX_URL}
fi

# Install transformers and helpers
if [[ "$NIGHTLY" == "yes" ]]; then
  echo "[setup] Installing latest transformers from GitHub (may be unstable)"
  pip install -U git+https://github.com/huggingface/transformers.git
else
  echo "[setup] Installing transformers and helper packages"
  pip install -U "transformers[torch]" safetensors accelerate
fi

# Quick import checks
echo "[setup] Running quick import checks..."
python3 - <<'PY'
import sys
errors = []
try:
    import torch
    print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
except Exception as e:
    errors.append(('torch', e))
try:
    from transformers import AutoModel, AutoTokenizer
    print('transformers:', getattr(__import__('transformers'), '__version__', 'unknown'))
except Exception as e:
    errors.append(('transformers', e))
try:
    import safetensors
    print('safetensors OK')
except Exception as e:
    errors.append(('safetensors', e))
if errors:
    print('\nErrors during imports:')
    for name, exc in errors:
        print(name, type(exc).__name__, exc)
    sys.exit(2)
print('\nAll imports OK')
PY

if [[ $? -ne 0 ]]; then
  echo "[setup] Import checks failed. See messages above."
  exit 2
fi

cat <<EOF

Setup completed successfully.
To use the environment:
  source ${VENV_DIR}/bin/activate
Example:
  python3 scripts/save_embeddings.py "Describe this" tests/landscape.jpg embeddings.pt --model /path/to/your/model.gguf --verbose

If your GPU requires a different CUDA tag (cu117, cu116), re-run:
  bash scripts/setup_pytorch.sh --venv ${VENV_DIR} --cuda cu117

EOF
