#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/home/hding22/binary
cd "$ROOT_DIR"

echo "[CI] Python: $(python -V)"
echo "[CI] CUDA available: $(python - << 'PY'
import torch
print(torch.cuda.is_available())
PY
)"

echo "[CI] Run default tests"
PYTHONPATH="$ROOT_DIR" pytest -q tests

if python - << 'PY'
import torch
exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "[CI] Run GPU-marked tests"
  PYTHONPATH="$ROOT_DIR" pytest -q -m gpu tests
else
  echo "[CI] Skip GPU-marked tests (no CUDA)"
fi

echo "[CI] Done"






