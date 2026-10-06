# Synthetic Tables 1 and 2

This package reproduces the four matched conditions C2, C4, C8, and L2–5, with
five seeds each. Table 1 contains DIPS, LCM+Lasso, and seven original-item
baselines (Ridge, Lasso, ElasticNet, PoissonGLM, RBF-SVR, RandomForest, MLP).
Table 2 contains DIPS best-F1 oracle recovery, LCM+Lasso CV recovery, and RF
positive-path CV recovery. Oracle recovery is truth-informed and is not the
model used for prediction. DecisionTree and target-scaled MLP are excluded.

The original 45 synthetic input files and training folds are included in
`inputs/`; the extra conditions support the separate pruning experiments.
The default run uses only the 20 matched datasets. These are generated data,
not patient records. `frozen/SHA256.json` checks the unmodified historical
Python sources. New runs create new registration hashes and preserve original
numerical settings. Historical cluster launchers and completed jobs are not
required.

## Environment

Use Python 3.10+ on Linux or macOS with NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2, and threadpoolctl 3.6.0. Linux/Python 3.10.12 was the
historical fitting environment. Windows requires WSL because frozen drivers
use `fcntl`. The wrapper sets BLAS/OMP thread counts to one. No GPU is needed.
LCM+Lasso additionally needs an external LCM 5.3 source archive and C compiler.

Run commands from the repository root:

```sh
python experiments/synthetic/run.py verify-inputs
python experiments/synthetic/run.py verify-generator
python -m unittest discover -s experiments/synthetic/tests -v
python experiments/synthetic/run.py prepare --output runs/synthetic
```

`verify-generator` regenerates all 40 count/length inputs, checks observations,
truths, item permutations, metadata and split indices exactly, and permits
only floating-point roundoff in the theoretical `true_mean` array. Five S30
inputs are checked against their original byte hashes, without regeneration.
The original JSON files are used in fitting so platform roundoff does not
change historical input bytes. Preparation creates registrations, not fits.

## Optional external LCM dependency

Official source: [Takeaki Uno's LCM](https://research.nii.ac.jp/~uno/code/lcm.html).
The archived `lcm53.zip` SHA256 is
`e294c1a015a45897cf0f173ee3d1b27c984ad656a7992a4c962c3c0942e7c0f7`.
LCM is not distributed or relicensed by this repository. Its own readme
specifies academic use, asks commercial users to contact the author, and
requires an author/homepage/newest-code reference when redistributing.
Consult those terms before use. The wrapper checks the archive hash and builds
the unchanged C source with `cc -O3`.

Either obtain the official archive yourself and run:

```sh
python experiments/synthetic/run.py build-lcm --output runs/synthetic --lcm-archive /path/to/lcm53.zip
```

or explicitly allow this command to download the pinned official archive:

```sh
python experiments/synthetic/run.py fetch-lcm --output runs/synthetic
```

All other commands are local. A changed upstream archive is rejected rather
than silently accepted.

## Run and aggregate

```sh
python experiments/synthetic/run.py run --output runs/synthetic --workers 1
python experiments/synthetic/run.py aggregate --output runs/synthetic --require-complete
```

This is a substantial CPU workload: each DIPS dataset has three training-fold
paths plus an outer-training path, each with 101 penalty values and independent
full-dictionary checks. LCM enumerates complete dictionaries and performs its
own 101-point three-fold Lasso search. There is no automatic time limit.
Default memory is 8 GiB per process; `--memory-gib` may increase LCM/baseline
memory while the frozen DIPS driver retains its 8 GiB limit. `--workers`
controls DIPS parallelism; baseline/structure jobs run sequentially. CPU IDs
come from the current process's available affinity, not historical servers.

Use `--datasets count_02_s2609271101` for an explicit single-dataset run, or
`--methods raw,dips,rf` when LCM is unavailable. RF requires the raw RandomForest
output first. `--scope pruning` selects 35 completed pruning conditions;
`--scope all` explicitly requests all 45, including harder conditions outside
the main paper comparison. Partial runs never produce five-seed means.
Completed results are verified before reuse; failures and interrupted jobs are
preserved and not retried automatically. Use a fresh output directory for a
new attempt.

`aggregate.json` uses schema `synthetic-seed-metrics-v1`. Its `rows` contain
`dataset_id`, `family`, `level`, `seed`, `method`, `selection`, `metrics`,
`recovery`, and relative source/hash fields. `groups` contains only complete
five-seed conditions with arithmetic `mean` and `sample_sd` (ddof=1).
`missing` and `complete` expose unfinished work. `table1_methods` and
`table2_methods` select the current manuscript scope; DIPS CV recovery remains
available without mixing it into the oracle rows. Native frozen-worker files
are retained under `raw/jobs`, `dips/jobs/full`, and `structure/jobs`.

The release was checked with input/generator verification and small numerical
truth tests; its 20-dataset training workload was not rerun during packaging.
