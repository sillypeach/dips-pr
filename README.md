# DIPS-PR: reproducible regularization-path experiments

Code for **Dual-Guided Screening of High-Order Interaction Patterns for Poisson
Regression**, by Ran Tao, Minrui Chen, and Hiroto Saigo.

This release accompanies the manuscript revised on **1 October 2026**. It
provides the exact frozen numerical kernels used in the reported experiments,
portable local experiment runners, pinned dependencies, synthetic inputs and
folds, and archived seed-level numbers. The previous published code is
preserved in [`legacy/20260913`](legacy/20260913); it does **not** reproduce the
current manuscript's regularization paths or repeated HIV evaluation.

## Start here

```sh
git clone https://github.com/sillypeach/dips-pr.git
cd dips-pr
git checkout reproducibility-2026-10-06
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python reproduce.py verify
python reproduce.py test -q
python reproduce.py tables --output results/paper_tables
```

Use Linux and Python 3.10 or later; the historical fitting environment was
Linux/Python 3.10.12. Small portability tests also run on macOS/Python 3.12.
Windows users need WSL because historical workers use POSIX facilities.
There is no GPU dependency. Numerical packages are pinned to NumPy 2.2.6,
SciPy 1.15.3, and scikit-learn 1.7.2. BLAS thread counts are set to one by the
entry points. Full experiments can require substantial CPU time and memory.

`verify` checks file hashes, original synthetic inputs/folds, and **561**
summary cells. `tables` recomputes means and sample standard deviations from
archived seed-level evidence; it **does not rerun training**. The separate
commands below perform fresh fits. The release validation uses small numerical
tests and input reconstruction checks; it is not a new full production run.

## What reproduces each experiment?

| Manuscript result | Scope | Code and instructions |
| --- | --- | --- |
| Table 1: prediction | C2, C4, C8, L2–5; 5 seeds; 9 methods | [Synthetic experiments](experiments/synthetic/README.md) |
| Table 2: exact recovery | Same 20 datasets; DIPS, LCM+Lasso, RF paths | [Synthetic experiments](experiments/synthetic/README.md) |
| Table 3: dictionary exclusion | C2, C4, C8, C12, L2–5, L3–6, S30; 35 full training paths | [Pruning](experiments/pruning/README.md) |
| Table 4: component ablation | C2, C4, L2–5; v, u, final v+u; 45 displayed paths | [Ablation](experiments/ablation/README.md) |
| NRTI and other HIV prediction tables | 25 endpoints; 5 grouped splits; 8 methods | [HIV experiments](experiments/hiv/README.md) |
| HIV learned terms | Representative seed-42 coefficients and mutation annotations | [Numerical snapshot](paper_snapshot/README.md) |

The original 45 synthetic datasets are included. Extra datasets outside the
listed complete conditions do not enter the current manuscript tables.
All table averages require five complete seeds; partial conditions are not
reported as complete means.

## Fresh synthetic prediction and recovery

```sh
python reproduce.py synthetic verify-generator
python reproduce.py synthetic prepare --output runs/synthetic
# Required only for LCM+Lasso: explicitly download and compile official LCM 5.3.
python reproduce.py synthetic fetch-lcm --output runs/synthetic
python reproduce.py synthetic run --output runs/synthetic --workers 1
python reproduce.py synthetic aggregate --output runs/synthetic --require-complete
```

The default scope is the 20 fully matched datasets. For a single dataset, add
`--datasets count_02_s2609271101` to `run`. For a run without external LCM, add
`--methods raw,dips,rf`; that run cannot produce the complete LCM comparison.
The DIPS kernel itself requires no external LCM executable. The optional
LCM+Lasso comparator downloads the official archive, verifies its pinned SHA,
and compiles it with a C compiler. It is governed by its own upstream terms;
it is not bundled or relicensed here.

Each DIPS fit uses 101 decreasing relative penalties from 1 to 0.01, exact
training-subset lambda-max calibration, its own warm starts, and an independent
full-dictionary check. Three saved training folds select the predictive
penalty by RMSE. The complete outer-training path also supplies the diagnostic
best-F1 recovery point.

## Fresh pruning and ablation

```sh
# Small invented-data numerical checks (not paper results):
python reproduce.py pruning smoke --output runs/pruning_smoke
python reproduce.py ablation smoke --output runs/ablation_smoke

# One original dataset; explicitly request --all for the registered full scope:
python reproduce.py pruning run --dataset count_02_s2609271101 --output runs/pruning_one
python reproduce.py ablation run --dataset count_02_s2609271101 --output runs/ablation_one

# Reconstruct the original tables from archived traversal evidence, without fitting:
python reproduce.py pruning paper --output results/table3
python reproduce.py ablation paper --output results/table4
```

`pruning run --all` fits 35 paths. `ablation run --all` fits the full five-arm
75-path study; the main table displays v, u, and final v+u. Each point is
verified independently, and failures are retained. `pruning count` can count
the complete support/length/closed dictionary without retraining.

## Fresh HIV evaluation

Obtain the five original filtered genotype–phenotype files from
[Stanford HIVDB](https://hivdb.stanford.edu/pages/genopheno.dataset.html).
The exact filenames and SHA-256 hashes are listed in the
[HIV instructions](experiments/hiv/README.md). Individual records are not
redistributed. The preparer rejects a changed upstream snapshot rather than
silently claiming it is the original experiment.

```sh
python reproduce.py hiv validate-inputs --data-dir /path/to/hivdb
python reproduce.py hiv prepare --data-dir /path/to/hivdb --output runs/hiv
# Example of an explicitly selected task:
python reproduce.py hiv run --root runs/hiv --endpoint NRTI/AZT --seed 42 --method Ridge
# Full retraining, from a fresh prepared study rather than the partly run example:
python reproduce.py hiv prepare --data-dir /path/to/hivdb --output runs/hiv_full
python reproduce.py hiv run --root runs/hiv_full --all
python reproduce.py hiv aggregate --root runs/hiv_full
```

All 125 endpoint/split input and fold fingerprints are checked against the
historical study. Split seeds are 42–46; estimator RNG is fixed at 42. Every
method uses the same SeqID-grouped outer holdout and three training folds.
The response remains the released fold change. The Poisson objective
is used as a pseudo-likelihood, without asserting that HIV responses are counts.

To reconstruct the descriptive metadata annotations of all 1,451 archived
seed-42 rules from the original raw records, without refitting:

```sh
python reproduce.py annotations --data-dir /path/to/hivdb --output results/rule_annotations
```

This checks every rule's support and observed annotation against the archived
catalogue. Coefficients remain those of the original position rules. The
output contains aggregate annotations, not individual records.

## Interpretation and provenance

* Prediction is selected using training-only CV. DIPS best-F1 recovery uses
  generating truth on synthetic data; it is not an automatic selection rule
  for unknown supports. The baselines' recovery rows use prediction-oriented
  selection, so this is not an equally tuned oracle comparison.
* RF is trained on raw binary positions/items. Its synthetic recovery score
  uses the documented pure-positive tree-prefix extraction rule, not an
  intrinsic RF support set. HIV has no known generating support and no
  biological Precision/Recall/F1 ranking is claimed.
* Pruning means `1 - sum(D-U)/(M*sum(R))` along a full training path. It counts
  excluded candidate opportunities before top-K and later filters. It is not
  a runtime or memory speedup. Zero-traversal lambda-max is NA.
* Component v/u rows use one reference, while final v+u uses two; their main
  table does not isolate the second reference. Full ablation evidence remains
  available. Timings exclude calibration, counting, serialization, and audit.
* Five-split HIV SDs describe variability across overlapping splits, not
  confidence intervals, independent-replicate SEs, or significance tests.
* Frozen kernels evaluate gaps and screening bounds in floating point. The
  `certified` mode selects the proved radius formula; it does not mean every
  floating-point exclusion has an outward-rounded numerical certificate.

Frozen-file manifests preserve kernel identity. New portable runners register
new output roots, while retaining the original objective, data, folds, grids,
selection rules, and tolerances. Failed or interrupted fits are not silently
retried with different parameters. Do not treat packaging tests as a claim of
bitwise-identical floating-point training on every platform.

See [validation](docs/VALIDATION.md), [the numerical snapshot](paper_snapshot/README.md),
and [release changes](docs/RELEASE_NOTES.md) for the verified scope.
