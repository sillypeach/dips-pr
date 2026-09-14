#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$ROOT/logs" "$ROOT/results/synthetic_rmse_lambda"
for seed in 42 43 44 45 46; do
  log="$ROOT/logs/rmse_lambda_large_seed${seed}_${STAMP}.log"
  pid_file="$ROOT/logs/rmse_lambda_large_seed${seed}_${STAMP}.pid"
  nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -u "$ROOT/synthetic_rmse_lambda_refit.py" \
    --seed "$seed" --weights large \
    --source-results "$ROOT/results/synthetic" \
    --output-dir "$ROOT/results/synthetic_rmse_lambda" --resume \
    >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'seed=%s pid=%s log=%s\n' "$seed" "$pid" "$log"
done
