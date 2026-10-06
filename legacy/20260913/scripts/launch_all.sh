#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$ROOT/logs" "$ROOT/results/synthetic" "$ROOT/results/hiv"

launch() {
  local name="$1"
  shift
  local log="$ROOT/logs/${name}_${STAMP}.log"
  local pid_file="$ROOT/logs/${name}_${STAMP}.pid"
  nohup env PYTHON="$PYTHON" "$@" >"$log" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf '%-24s pid=%-8s log=%s\n' "$name" "$pid" "$log"
}

for seed in 42 43 44 45 46; do
  launch "synthetic_seed${seed}" \
    bash "$ROOT/scripts/run_synthetic_seed.sh" "$seed"
done

for drug_class in NRTI NNRTI PI INI CAI; do
  launch "hiv_${drug_class}" \
    bash "$ROOT/scripts/run_hiv_class.sh" "$drug_class"
done
