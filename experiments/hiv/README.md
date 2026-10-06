# HIV five-repeat grouped holdout reproduction

This entry point reproduces the **25 endpoints × five SeqID grouped splits (42–46)** used by the current manuscript. It preserves the numerical workers from `hiv_repeated_split_20260930_v1` byte for byte. The public adapter prepares inputs from user-supplied source files, checks the saved 125 data/fold fingerprints, creates a new local registration, and runs the frozen workers sequentially.

No individual source records, SeqIDs, patient identifiers, responses, predictions, or prepared HIV datasets are bundled. The `fixtures/` files contain settings, source checksums, counts, and irreversible SHA256 fingerprints only. Preparation and training create individual-level artifacts in your chosen output directory; keep that directory outside the public checkout.

## Environment

Linux is the canonical platform for the original workers (Python 3.10.12, NumPy 2.2.6, scikit-learn 1.7.2 in saved server receipts). The release adapter and invented-data integration tests were also checked with Python 3.12.14, NumPy 2.2.6, SciPy 1.15.3, and scikit-learn 1.7.2 on macOS. Use the repository's pinned requirements. Workers run with one numerical-library thread and an 8 GiB address-space limit on Linux. Optional `--cpu N` sets CPU affinity. The sequential wrapper does not use the original server controller or connect to a server.

Commands below assume the repository root is the current directory. `python ... --help` only prints help; it does not import numerical libraries or start fitting.

## Obtain the source snapshot

Use the [Stanford HIVDB genotype–phenotype dataset page](https://hivdb.stanford.edu/pages/genopheno.dataset.html) to obtain these **filtered**, tab-separated files into a private directory:

- `NRTI_DataSet.txt`
- `NNRTI_DataSet.txt`
- `PI_DataSet.txt`
- `INI_DataSet.txt`
- `CAI_DataSet.txt`

Read the upstream terms and citation guidance on the [Stanford HIVDB website](https://hivdb.stanford.edu/). This release does not redistribute or relicense Stanford data. The `.Full.txt` files are not training inputs; they were used for the historical metadata audit only and are unnecessary for reconstructing the predictive experiment.

The exact source hashes and dimensions are in [`fixtures/raw_files.json`](fixtures/raw_files.json). The tool **refuses a changed upstream snapshot**: accepting newer source data would no longer reproduce the frozen manuscript experiment. It does not silently relax this check or download files. If the upstream files have changed, obtain the matching historical snapshot from the data provider.

```bash
python experiments/hiv/run.py validate-inputs --data-dir /path/to/private/hiv-data
python experiments/hiv/run.py prepare --data-dir /path/to/private/hiv-data --output /path/to/private/hiv-reproduction
```

`validate-inputs` performs no fitting and writes no observations. `prepare` requires a new output directory and verifies all five raw file hashes plus all 125 source-record, split-core, and fold-core fingerprints **before** writing the new datasets. The fingerprints match both the original local and server reconstruction receipts. Fresh registration hashes differ from the original server registration because timestamps and portable metadata differ; the observations, memberships, numerical workers, and experiment settings are checked independently.

## Run and verify

A small explicit target can be selected to check the installation. This is a real HIV fit and is distinct from the invented-data tests below:

```bash
python experiments/hiv/run.py run --root /path/to/private/hiv-reproduction --endpoint CAI/LEN --seed 42 --method Ridge
```

For one endpoint and seed, `--method DIPS-PR` runs three 101-point CV paths and the 101-point final path. `--method all` runs DIPS-PR and all seven baselines and independently audits the completed endpoint. Alternatively, run each method exactly once and call `verify` after all eight methods finish.

```bash
python experiments/hiv/run.py run --root /path/to/private/hiv-reproduction --endpoint NRTI/AZT --seed 43 --method all
python experiments/hiv/run.py verify --root /path/to/private/hiv-reproduction --endpoint NRTI/AZT --seed 43
```

For the full manuscript experiment, use a **fresh prepared output directory**, then explicitly request all 125 endpoint/split runs:

```bash
python experiments/hiv/run.py run --root /path/to/private/hiv-reproduction-full --all
python experiments/hiv/run.py aggregate --root /path/to/private/hiv-reproduction-full
```

Prepare `/path/to/private/hiv-reproduction-full` using the `prepare` command first. `--all` entails 50,500 DIPS path points, 22,125 baseline CV fits, and 875 baseline final fits; runtime depends strongly on hardware. The frozen parallel experiment used 1,375 worker processes. The portable wrapper executes these jobs sequentially. It refuses to overwrite or retry any selected prior attempt. To reproduce afresh, prepare a new output directory. A failed job stops the requested run; no split or failed method is silently replaced.

The full run calls the frozen endpoint analyzer and aggregator. The latter accepts only all 125 verified runs and writes:

- `analysis/aggregate_repeats.json` and `analysis/seed_metrics.json`
- `analysis/nrti_main.tex`, `analysis/other_endpoints.tex`, and `analysis/one_se_supplement.tex`

The aggregator preserves failures and returns a nonzero exit code for an incomplete study. It never computes a formal five-seed result from a partial set. `verify` and `aggregate` read existing fits and do not retrain. Mean and sample SD use equal weighting of exactly five test scores (`ddof=1`). These overlapping grouped splits describe split variability; their SD is not an independent-sample standard error or a confidence interval.

## Frozen protocol

All finite, nonnegative published filtered drug responses are retained without further transformation, rounding, shifting, clipping, or capping. Missing/nonfinite responses are excluded for that endpoint. Repeated SeqIDs are retained and grouped together. A feature is one position column in original header order, with presence iff its stripped source token is not empty, `-`, or `.`. This position encoding does not identify amino-acid substitutions.

Each endpoint uses `GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)` on SeqIDs. Three-fold CV applies `KFold(3, shuffle=True, random_state=seed)` to sorted unique training SeqIDs, then maps groups back to rows. All outer-training observations are used for both CV and final fitting; there is no unused validation subset. The outer split seed and CV seed vary together over 42–46; estimator randomness remains fixed at 42. This guarantees SeqID separation, not patient independence or separation of identical encoded bit-vectors.

DIPS uses `tau[k] = 10**(-2*k/100)`, `k=0,...,100`, with an independently calibrated exact training-subset `lambda_max` for each CV and final path. The chosen tau minimizes arithmetic mean fold RMSE; exact ties choose the largest tau. The primary prediction is the corresponding point in the final path. The one-SE selection is secondary: sample SD of the three RMSEs divided by `sqrt(3)`, then the largest eligible tau. The dictionary uses closed patterns of length at most 4 and minimum support `max(2, floor(0.02*n_fit))` separately within each fitting subset. `kappa=0.05`; certified radius, global gap and KKT tolerances are `1e-7`; all other solver settings are preserved in `fixtures/protocol_settings.json`.

The seven displayed and retrained baselines have these exact candidate counts and grids:

| Method | Grid | Candidates |
|---|---|---:|
| Ridge | alpha 0.01, 0.1, 1, 10 | 4 |
| Lasso | alpha 0.001, 0.01, 0.1 | 3 |
| Elastic Net | alpha 0.001, 0.01, 0.1 × l1_ratio 0.2, 0.5, 0.8 | 9 |
| Poisson GLM | alpha 0.001, 0.01, 0.1, 1 | 4 |
| RBF-SVR | C 0.1, 1, 10, 100 × epsilon 0.01, 0.1, 1 | 12 |
| Random Forest | max_depth None, 8, 16 × min_samples_leaf 1, 2, 5 × max_features 1.0, sqrt; 300 trees | 18 |
| MLP | hidden sizes (64), (128), (64,32) × alpha 0.0001, 0.001, 0.01 | 9 |

The frozen factories define all remaining estimator arguments. StandardScaler is fitted inside each fitting fold for all methods except Random Forest. Baselines select minimum mean CV RMSE, with first `ParameterGrid` index resolving exact ties. Finite fits remain eligible when they emit warnings; warnings are saved. Invalid candidates are retained and excluded. The preserved worker contains historical DecisionTree compatibility code, but DecisionTree is absent from the seven-method public protocol and CLI.

R² uses raw predictions. Primary Poisson pseudo-R² is `PseudoR2_train_null`, whose null mean is computed from the final fitting responses. Only deviance calculations floor predictions/null means at `1e-6`; raw predictions are not clipped. The test-mean null is supplemental. Undefined zero-denominator scores are null. HIV has no known true interaction support, so this experiment does not produce recovery precision, recall, or F1. The data were previously used in development; repeated splits are not external validation.

## Tests and provenance

```bash
python -m unittest discover -s experiments/hiv/tests -v
```

Tests fit only invented observations, including the full 404-point DIPS path through the public CLI and a Ridge CV/final fit. Other tests check grouped repetitions, encoding, corrupted source rejection, hash integrity, explicit training targets, and no retry of previous attempts. `frozen/MANIFEST.json` records the original byte-level checksums. The original frozen preparer is retained for its exact split and fingerprint functions; the public CLI does not invoke its private-study-dependent freeze routine. The manuscript's representative seed-42 term/mutation catalogue derives from the earlier full grouped study; that descriptive catalogue is separate from the five-repeat prediction averages.
