#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$ROOT/logs" "$ROOT/results/agreement_multiseed"
for start in 1000 1002 1004 1006 1008 1010 1012 1014 1016 1018; do
  second=$((start + 1))
  log="$ROOT/logs/agreement_${start}_${second}_${STAMP}.log"
  pid_file="$ROOT/logs/agreement_${start}_${second}_${STAMP}.pid"
  nohup env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" -u "$ROOT/agreement_multiseed.py" \
    --seeds "$start" "$second" \
    --output-dir "$ROOT/results/agreement_multiseed" --resume \
    >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'seeds=%s,%s pid=%s log=%s\n' "$start" "$second" "$pid" "$log"
done
