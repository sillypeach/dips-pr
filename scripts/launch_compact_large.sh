#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$ROOT/logs" "$ROOT/results/synthetic_compact_refit_ridge"
for seed in 42 43 44 45 46; do
  log="$ROOT/logs/compact_large_seed${seed}_${STAMP}.log"
  pid_file="$ROOT/logs/compact_large_seed${seed}_${STAMP}.pid"
  nohup env PYTHON="$PYTHON" \
    bash "$ROOT/scripts/run_compact_refit_seed.sh" "$seed" --weights large \
    >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'seed=%s pid=%s log=%s\n' "$seed" "$pid" "$log"
done
