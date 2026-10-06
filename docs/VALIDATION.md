# Release validation

The release was checked locally on 6 October 2026 in an isolated Python 3.12.14
environment with the exact numerical dependencies in `requirements.txt`.
The publication runs used Linux/Python 3.10.12. The CI configuration checks
the public package on Linux/Python 3.10; a configured workflow is not itself
evidence that a remote run has completed.

## Checks performed

* SHA-256 checks for frozen numerical sources and all 45 original synthetic
  dataset/fold pairs.
* Regeneration of all 40 count/length datasets with exact fitted observations,
  truths, metadata, item permutations, and split indices. The theoretical
  mean array permits explicitly bounded floating-point roundoff. The five
  shared-item datasets retain verified original bytes.
* Reconstruction of all 125 real-data endpoint/split inputs and folds from
  the five hash-matched raw snapshots, without fitting on those observations.
  No individual records were added to the release.
* Recalculation of all 561 manuscript summary cells from per-seed evidence,
  using sample SD and requiring five complete values.
* Independent reconstruction of all seven pruning-table rows and all nine
  displayed ablation rows from saved per-point/per-path counter evidence.
* Reconstruction and verification of the observed annotations for all 1,451
  archived seed-42 position rules, using source metadata. This is descriptive
  metadata reconstruction, not biological validation or model fitting.
* Small numerical tests compare exact lambda-max and screened solutions with
  brute-force dictionaries/full-dictionary optimization. Other tests cover
  grouped folds, source/hash corruption, incomplete summaries, RF prefix
  semantics, component switches, and refusal to overwrite prior attempts.
* An invented-data public-runner integration completes all four 101-point
  paths and a Ridge CV/final fit. Separate tiny pruning/ablation smoke runs
  pass independent full-dictionary checks.
* One original C2 synthetic Ridge job selects the same alpha (100) as the
  archived result; the largest absolute difference among stored metrics is
  approximately 8.73e-11. This is one comparator check, not a rerun of all
  publication results.
* The pinned external LCM archive was built locally in a temporary directory;
  its small closed dictionary matched brute-force enumeration. The archive
  and local binary are not distributed in this repository.

Run `python reproduce.py test -q` for the current test count and
`python reproduce.py verify` for the committed release's hash/arithmetic checks.
Fresh release validation results are recorded in `VALIDATION.json`.

## What is not claimed

No full production study was rerun during packaging. Hardware-dependent
timings are not guaranteed. Exact source/input identity does not guarantee
bitwise-identical floating-point fits on every platform. Archived summary
reconstruction is distinct from fresh fitting. The statistical interpretation
of the manuscript, including oracle recovery and overlapping-split SDs, is
unchanged by this software release.
