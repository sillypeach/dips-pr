# DIPS-PR: Dual-Guided Interaction Pattern Screening for Poisson Regression

This repository contains a research implementation of DIPS-PR, a sparse
Poisson regression method for high-order itemset interaction discovery. The
algorithm combines a working-set sparse Poisson solver with LCM-style itemset
enumeration and dual-guided pattern screening/pruning.

## Repository Contents

- `model.py`: core optimization, dual-gap scoring, itemset enumeration, and
  pruning logic.
- `utils.py`: feature-matrix construction, recovery metrics, pseudo-R2, and
  reporting utilities.
- `exp_shared.py`: synthetic-data and baseline helpers used by the experiment
  scripts.
- `run_ablation.py`: generated-data pruning ablation used for the paper.
- `run_heuristic_vs_certified.py`: comparison between the local-curvature
  heuristic radius and the certified-radius variant.
- `run_comparison.py`, `run_diagnostics.py`: additional generated-data
  diagnostics.
- `plot_pruning_en.py`: workflow/pruning diagram generator.
- `examples/quick_synthetic.py`: small smoke test for a fresh installation.

Large datasets, intermediate outputs, LaTeX build products, notebooks, local
IDE metadata, and logs are intentionally excluded from this release package.

## Installation

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Check

Run the smoke test:

```bash
python examples/quick_synthetic.py
```

A successful run prints a small set of selected itemsets, pruning statistics,
and feature-recovery metrics.

## Reproducing Generated-Data Experiments

Run the main generated-data pruning ablation:

```bash
python run_ablation.py
```

Run the heuristic-vs-certified radius comparison:

```bash
python run_heuristic_vs_certified.py
```

These scripts generate synthetic transaction data internally, so no external
dataset is required for the generated-data experiments.

## Data Policy

The original working directory contained several public and derived datasets
that are not included here because of size, licensing, and reproducibility
concerns. Real-data scripts should be run only after downloading the relevant
datasets and adapting paths locally. See `docs/DATA.md` for details.

## Notes

This is research code. The default experiments match the generated-data
settings used during manuscript development, but hyperparameters may need
adjustment for new datasets.

