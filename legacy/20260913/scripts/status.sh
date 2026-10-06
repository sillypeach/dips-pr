#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Running processes"
pgrep -af "synthetic_multiseed.py|hiv_nonlinear_baselines.py" || true
echo
echo "Synthetic checkpoints"
find results/synthetic -type f -name '*.json' ! -name 'manifest.json' \
  ! -name '*.error.json' | wc -l
echo "HIV endpoint checkpoints"
find results/hiv -type f -name '*.json' | wc -l
echo
echo "Recent log tails"
for log in logs/*.log; do
  echo "--- $log ---"
  tail -n 4 "$log"
done
