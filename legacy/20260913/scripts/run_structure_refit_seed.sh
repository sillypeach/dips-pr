#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:?usage: run_structure_refit_seed.sh SEED}"
PYTHON="${PYTHON:-$HOME/LARS_Poisson/.venv/bin/python}"
ORIGINAL_DIR="${ORIGINAL_DIR:-$ROOT/results/synthetic}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/structure_refits}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
exec "$PYTHON" -u synthetic_structure_refit.py \
  --seed "$SEED" \
  --original-dir "$ORIGINAL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --resume
