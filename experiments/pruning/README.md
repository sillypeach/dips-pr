# Complete-path pruning (Table 3)

This directory preserves the frozen September 28 PATH solver and the numeric evidence for C2, C4, C8, C12, L2–5, L3–6, and S30. Each displayed condition has five complete outer-training paths. The other registered length conditions are not included in this table.

Run from the repository root. Every command requires a **new output directory** and preserves existing results.

```sh
# Reconstruct all seven rows and the per-lambda curve data; standard library only.
python experiments/pruning/run.py paper --output runs/paper_pruning

# Tiny generated fixture: three lambda points, independent audit after every fit.
python experiments/pruning/run.py smoke --output runs/pruning_smoke

# Fresh full 101-point outer-training path for one frozen dataset.
python experiments/pruning/run.py run --dataset count_02_s2609271101 --output runs/pruning_one

# Optional: all 35 complete-condition datasets. This can take substantial time.
python experiments/pruning/run.py run --all --output runs/pruning_all

# Verify saved point hashes/counters and summarize only complete five-seed groups.
python experiments/pruning/run.py summarize --input runs/pruning_all --output runs/pruning_summary

# Count the full eligible dictionary once, without fitting.
python experiments/pruning/run.py count --dataset count_02_s2609271101 --output runs/count_c2
```

Fresh fitting/counting uses the immutable shared files in `experiments/synthetic/inputs`. `--inputs PATH` can point to another copy; data and fold hashes must match. Fresh fitting needs NumPy and SciPy. The adapters are portable Python and use one BLAS thread. They compute the exact lambda maximum and count the dictionary once per dataset, then retain each path's own preceding-lambda state. They run the frozen independent full-dictionary gap/KKT checks after every fitted point. A failed fit or audit stops the invocation and preserves diagnostic files. There is no automatic retry, tolerance adjustment, or CV in this outer-path table runner. The original study performed CV for prediction selection; Table 3 reports all outer-path lambda values and does not depend on a selected lambda.

`paper` writes `table3.json`, `table3.csv`, LaTeX table-body rows in `table3.tex`, `seed_paths.json`, and `pruning_curves.csv`. The curve file preserves the analytic endpoint as NA. `summarize` writes the same format, with NA for every incomplete five-seed condition. Smoke output cannot be used as formal evidence. A single fresh seed is useful for inspecting execution; it cannot reproduce a five-seed mean/SD.

The full eligible dictionary size is M: all closed patterns meeting the training subset's support and maximum-length rules, with u/v screening disabled. The support threshold is `max(3, ceil(0.015*n_train))`, hence 21 for the 1,400 outer-training rows. At each actual fused traversal, D is `correlation_evaluations = nodes - support_exclusions - canonical_exclusions - v_exclusions`. U is `screened_u`. The surviving count is D−U before top-K, candidate post-filtering, or working-set removal. It includes existing working-set patterns.

The complete-path exclusion rate is `1 - sum(D-U)/(M*sum(traversals))`. Sum counts first, then calculate a ratio for each seed, then report the arithmetic seed mean and sample SD. Averaging lambda percentages gives a different quantity. Repeated opportunities count repeatedly. At lambda/lambda_max=1 the analytic intercept-only solution has no traversal and the pruning rate is NA.

`config.json` records the original protocol digest, full scientific settings, the 101-point grid, data/fold/training-row hashes, and each frozen source digest. Kernels in `frozen/` are byte-identical to their archived originals. `common.py` is a new release adapter, shared with the ablation runner; it does not change those kernels. Separate CLI processes keep unqualified `model` imports isolated.

`evidence/points.csv` contains 3,535 sanitized scalar point records, with hashes of the original result files. Its manifest retains the accepted upstream hashes and the release evidence hash. These records reproduce table arithmetic and preserve the original independent-audit status. The complete original raw job trees are **not bundled**; `paper` checks the scalar evidence and does not rerun the original audits. Fresh runs produce new, auditable point files, dictionary/calibration files, and manifests.

The Python enumerator and dictionary counter do not require an external LCM executable. Pruning percentages alone make no claim about runtime or memory savings. Fresh runtime measurements depend on hardware and numerical libraries.
