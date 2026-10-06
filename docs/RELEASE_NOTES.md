# Reproducibility release — 6 October 2026

This release replaces the repository's primary instructions with the
regularization-path and repeated-split experiments in the 1 October manuscript.
The previously published revision remains intact under `legacy/20260913` and
in Git history.

## Included

* Frozen solver kernels for Poisson fitting, dual reference transfer, fused
  pattern enumeration/screening, and independent full-dictionary checks.
* Portable synthetic prediction/recovery, pruning, component-ablation, and
  repeated grouped-holdout runners. Their new registrations contain local
  output paths; their numerical kernels and experiment settings are preserved.
* All 45 original synthetic datasets and saved folds with checksums, plus
  count/length generation and validation against the original observations.
* Original-item baselines, LCM+Lasso, and documented RF positive-prefix
  extraction. The optional external LCM source is fetched only on request,
  checked by SHA-256, and governed by its own terms.
* Data preparation from user-provided original real-data files, with exact
  checksums and all 125 input/fold fingerprints. Individual records are not
  included in this release.
* Archived per-seed metrics, traversal counters, coefficient catalogue, and
  deterministic table/annotation reconstruction tools.
* Pinned numerical dependencies, release and kernel hashes, small numerical
  tests, and GitHub Actions configuration for those tests.

## Scope retained from the manuscript

The prediction and recovery tables use four fully matched conditions. Pruning
uses seven complete conditions. The displayed ablation has three arms, while
the archived five-arm study is retained. Real-data prediction uses all 25
endpoints and five grouped splits; the pattern case studies describe seed 42.
Decision Tree, target-scaled MLP, and earlier agreement-figure experiments do
not enter the current reported comparisons.

Recovery selected with generating truth is explicitly distinguished from
prediction selected with training-only CV. Sample SD is not relabeled as a
confidence interval. Counter-based exclusion is not relabeled as time or
memory acceleration. The single-component versus final ablation display is
not described as an isolated two-reference comparison.

## Validation boundary

This release preparation did not rerun the entire publication-scale study.
It verified the original inputs, frozen sources, archived arithmetic, and
portable execution using small numerical fixtures and one original synthetic
Ridge run. Full experiment commands are included for independent reruns.
Original server process logs and every raw full-path artifact are not bundled;
the documented archived evidence is not presented as those absent files.
See [VALIDATION.md](VALIDATION.md).
