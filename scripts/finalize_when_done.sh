#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

cd "$ROOT"
printf 'Waiting for experiment workers at %s\n' "$(date --iso-8601=seconds)"
while pgrep -u "$(id -u)" -f \
  'synthetic_multiseed.py|hiv_nonlinear_baselines.py' >/dev/null; do
  sleep 60
done

"$PYTHON" aggregate_results.py --results-dir results
printf 'Aggregation completed at %s\n' "$(date --iso-8601=seconds)"
printf 'Synthetic setting files: '
find results/synthetic -type f -name '*.json' ! -name 'manifest.json' \
  ! -name '*.error.json' | wc -l
printf 'HIV endpoint files: '
find results/hiv -type f -name '*.json' | wc -l
