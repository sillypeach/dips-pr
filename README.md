# DIPS-PR reproducibility package

This package contains the source code, launch scripts, dependency versions,
and machine-readable summaries used for the revised IEEE Access manuscript
"Dual-Guided Screening of High-Order Interaction Patterns for Poisson
Regression."

Repository: https://github.com/sillypeach/dips-pr

```bash
git clone https://github.com/sillypeach/dips-pr.git
cd dips-pr
```

The package covers:

- five synthetic seeds (`42`-`46`) for nine sample-size/signal settings;
- Ridge, Lasso, Elastic Net, Poisson GLM, RBF-SVR, Random Forest, and MLP
  baselines;
- exact embedded-pattern precision/recall and the structural audit of every
  selected itemset;
- all 25 original Stanford HIVDB fold-change endpoints, grouped by drug class;
- 2,000-replicate paired test-set bootstrap comparisons; and
- local-curvature versus certified-radius agreement over 25 seeds.

All commands below are run from the package root. Every long experiment has a
shell launcher that writes standard output and errors to a timestamped file in
`logs/`. JSON checkpoints are written atomically, and `--resume` skips completed
settings.

## 1. Environment

The archived runs used Ubuntu Linux, Python 3.10.12, Intel Xeon Gold 6346 CPUs,
and the exact package versions in `requirements.txt`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code uses CPU implementations only. Independent settings can be launched
in parallel, but the wrappers set BLAS thread counts to one to avoid nested
parallelism.

## 2. External HIVDB data

The Stanford HIVDB files are not redistributed in this archive. Download the
five high-quality filtered datasets from:

https://hivdb.stanford.edu/pages/genopheno.dataset.html

Place the files below in `data/` without renaming them:

```text
data/PI_DataSet.txt
data/NRTI_DataSet.txt
data/NNRTI_DataSet.txt
data/INI_DataSet.txt
data/CAI_DataSet.txt
```

The files used for the reported experiment had these SHA-256 checksums:

```text
6fd96655bf8d5314dde46ceb014d86e5e023a505309a6ed77a3e4f9be46be1b5  PI_DataSet.txt
459a4c49f49b7ef960fc8da1ed2fe1c2fec68b04cc78be675a1fa83de7295fa0  NRTI_DataSet.txt
753bf2da2b323735cc5b310d903590e01ef481cbbbee6c30a3177c9c6db57a66  NNRTI_DataSet.txt
3c87a4e588723da461cf2dc2008ca95a7b5067aafa1a74b103a81febd5277c80  INI_DataSet.txt
8082e16158dac4804e9ff9d76a8f60dee58d60c8f97189d35e6134e6cb479524  CAI_DataSet.txt
```

The response is each drug's original positive fold-change value. The code does
not shift, discretize, or log-transform the response. Missing fold-change
entries are removed endpoint by endpoint.

## 3. Main experiments

Launch the five synthetic jobs and five HIV drug-class jobs:

```bash
PYTHON="$PWD/.venv/bin/python" bash scripts/launch_all.sh
```

Monitor checkpoints and log tails:

```bash
bash scripts/status.sh
```

Aggregate the completed experiments:

```bash
.venv/bin/python aggregate_results.py --results-dir results
.venv/bin/python summarize_revision_results.py \
  --results results --output revision_tables
```

The generated-data results use five independent data seeds for each of the
nine settings. Hyperparameters are selected using training-set cross-validation
only; the held-out test partition is evaluated once after refitting. The HIVDB
experiment uses fixed seed 42 and an 80/10/10 train/validation/test split, with
three-fold cross-validation confined to the training partition.

## 4. Structural audit for Table 3

After `results/synthetic/` contains the 45 main synthetic JSON checkpoints,
launch final-model refits that save every active itemset and classify it as an
exact match, strict subset, strict superset, partial overlap, or disjoint from
the eight embedded patterns:

```bash
PYTHON="$PWD/.venv/bin/python" \
ORIGINAL_DIR="$PWD/results/synthetic" \
OUTPUT_DIR="$PWD/results/structure_refits" \
bash scripts/launch_structure_refits.sh
```

Then aggregate and verify all 45 refits:

```bash
.venv/bin/python aggregate_structural_audit.py \
  --input-dir results/structure_refits \
  --output-dir results/structure_summary \
  --expected-runs 45
```

`reproduction_verification.json` records whether each refit exactly reproduces the
stored `R2`, pseudo-`R2`, and nonzero-pattern count. In the archived audit,
all 45 refits matched, with maximum absolute differences of zero.

## 5. Radius-agreement experiment for Figure 2

```bash
.venv/bin/python agreement_multiseed.py \
  --seeds 42 99 137 256 512 1000 1001 1002 1003 1004 1005 1006 1007 \
  1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 \
  --output-dir results/agreement_multiseed --resume
```

Summarize and plot the comparison:

```bash
.venv/bin/python summarize_revision_results.py \
  --results results --output revision_tables
.venv/bin/python plot_agreement_figure.py \
  --summary revision_tables/agreement_summary.csv \
  --output revision_tables/heuristic_certified_output_agreement_25seeds.pdf
```

## 6. Output map

- `results/synthetic_mean_std.csv`: Table 1 predictive `R2` summaries.
- `revision_tables/synthetic_pr2_rows.tex`: Table 2 pseudo-`R2` summaries.
- `results/structure_summary/structural_audit_mean_sd.csv`: Table 3 exact and
  structural-composition summaries.
- `results/hiv_nonlinear_all_endpoints.csv`: complete Table 4 and Appendix
  Table 9 comparisons.
- `results/hiv_paired_bootstrap_vs_dips.csv`: paired bootstrap intervals used
  in the main-text HIV discussion.
- `revision_tables/agreement_summary.csv`: Figure 2 radius-mode agreement.
- `revision_tables/agreement_overall.json`: aggregate agreement counts.

The archived main results and structural audit are included under
`archived_results/`. They are provided for direct inspection, not as a
substitute for the source code and commands above.

## 7. Important interpretation details

For Table 3, a true positive requires exact equality with an embedded itemset.
Subsets, supersets, and partially overlapping itemsets are false positives
under that exact-recovery definition, although their structural categories are
reported separately. This is intentionally stricter than prediction-oriented
feature relevance.

For nonnegative continuous HIVDB fold-change responses, the objective is used
as a Poisson pseudo-likelihood for a log-linear conditional mean. The analysis
does not claim that fold changes are Poisson-distributed.
