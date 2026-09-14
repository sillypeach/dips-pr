#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/LARS_Poisson/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$ROOT/logs/structure_refits_$STAMP}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/structure_refits}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

for seed in 42 43 44 45 46; do
  log="$LOG_DIR/structure_seed${seed}_${STAMP}.log"
  pid_file="$LOG_DIR/structure_seed${seed}_${STAMP}.pid"
  {
    date -Is
    hostname
    sha256sum \
      "$ROOT/synthetic_multiseed.py" \
      "$ROOT/synthetic_structure_refit.py" \
      "$ROOT/model.py" \
      "$ROOT/common.py" \
      "$ROOT/utils.py"
    echo "COMMAND: seed=$seed output=$OUTPUT_DIR"
    PYTHON="$PYTHON" OUTPUT_DIR="$OUTPUT_DIR" \
      bash "$ROOT/scripts/run_structure_refit_seed.sh" "$seed"
  } >"$log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf 'seed=%s pid=%s log=%s\n' "$seed" "$pid" "$log"
done
