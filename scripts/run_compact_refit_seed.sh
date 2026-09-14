#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${1:?usage: run_compact_refit_seed.sh SEED}"
shift
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$ROOT"
exec "$PYTHON" -u synthetic_compact_refit.py \
  --seed "$SEED" \
  --source-results results/synthetic \
  --output-dir results/synthetic_compact_refit_ridge \
  --resume \
  "$@"
