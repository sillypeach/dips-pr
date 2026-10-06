#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:?usage: run_synthetic_seed.sh SEED}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
exec "$PYTHON" -u synthetic_multiseed.py \
  --seed "$SEED" \
  --output-dir results/synthetic \
  --baseline-jobs 2 \
  --resume
