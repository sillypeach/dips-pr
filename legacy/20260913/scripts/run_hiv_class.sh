#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRUG_CLASS="${1:?usage: run_hiv_class.sh DRUG_CLASS}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
exec "$PYTHON" -u hiv_nonlinear_baselines.py \
  --drug-class "$DRUG_CLASS" \
  --data-dir data \
  --output-dir results/hiv \
  --seed 42 \
  --jobs 2 \
  --resume
