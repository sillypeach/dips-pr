# Component ablation (Table 4)

The archived October 1 experiment has five arms, 15 datasets, and 75 complete paths (7,575 lambda points). The manuscript displays `v_only`, `u_only`, and `vu_two` for C2, C4, and L2–5, with five seeds each. The final displayed v+u arm is **two-reference `vu_two`**. The archive also preserves `none` and `vu_single`; it is not the separate adaptive-reference experiment.

Run from the repository root and choose a **new output directory** for every invocation.

```sh
# Reconstruct the nine manuscript rows and all 15 source-study rows.
python experiments/ablation/run.py paper --output runs/paper_ablation

# Tiny generated fixture, five arms × three points, independent checks.
python experiments/ablation/run.py smoke --output runs/ablation_smoke

# Fresh complete 101-point path for one dataset and one arm.
python experiments/ablation/run.py run --dataset count_02_s2609271101 --arm v_only --output runs/ablation_one

# Selected datasets/arms are also accepted. Default arms are the three displayed arms.
python experiments/ablation/run.py run --dataset count_02_s2609271101 --arm v_only u_only vu_two --output runs/ablation_three

# Optional full study: all 15 datasets × all five arms, 101 points per path.
python experiments/ablation/run.py run --all --output runs/ablation_all

# Hash/provenance checks, counter reconstruction, and complete-five-seed reporting.
python experiments/ablation/run.py summarize --input runs/ablation_all --output runs/ablation_summary
```

`paper` uses only Python's standard library. Fresh fitting needs NumPy and SciPy and reads the immutable shared `experiments/synthetic/inputs`; `--inputs PATH` selects another copy with matching hashes. `--all` cannot be combined with `--arm`. The full run is sequential with a cyclic arm order by dataset index and one BLAS thread. It may take substantial time. No external server, external LCM executable, new CV, or model selection is involved.

Each dataset's complete eligible dictionary is counted once and its exact lambda maximum is constructed once. Each arm starts its own state and follows all 101 original normalized penalties from 1 to 0.01. The adapter invokes the original independent full-dictionary gap/KKT checks after every fit. A failed fit/audit stops the invocation with its evidence preserved; no automatic retries or convergence changes occur. The tiny smoke is explicitly a prefix diagnostic and cannot contribute to a formal table.

| Arm | Subtree v | Individual u | Second reference |
|---|---|---|---|
| none | off | off | off |
| v_only | on | off | off |
| u_only | off | on | off |
| vu_single | on | on | off |
| vu_two | on | on | on |

The u component includes the fused individual screen, candidate post-filter, and working-set removal. All arms retain their own primal warm start and recertified primary preceding-lambda reference. They share the same dictionary rules, candidate budget, solver settings, and verification requirements. No permanently screened tree crosses lambda values.

The exclusion formula and full-dictionary denominator follow [the pruning experiment](../pruning/README.md). Node visits are repeated raw visits, including support/canonical/ineligible nodes, displayed in millions. Fit time is the sum of timed `fit_path_point` calls; it excludes dictionary counting, lambda calibration, output serialization, and independent verification. Reference work inside the fitting kernel remains included. A high pruning rate does not imply a speedup. Historical seconds are measured results on the original execution environment, not a promise for a new machine.

`paper` writes `table4.json`, `table4.csv`, LaTeX table-body rows in `table4.tex`, `all_arms.json`, and `seed_paths.json`. `summarize` reconstructs these from newly saved point files and their hash chain. It emits condition means/SDs only when all three displayed arms have all five complete seeds. A single-path run therefore produces incomplete-condition entries, not a substitute table.

`frozen/` contains the byte-identical model, path utilities, explicit ablation kernel, and independent audit model from the original snapshot. `config.json` identifies each source and the exact registered data/settings. The portable adapter lives in `run.py` and the sibling `pruning/common.py`; neither original server controllers nor operational logs are included.

`evidence/points.csv` has all 7,575 sanitized point records, including the original result hashes and independent-audit status. Its manifest records the accepted validated summary/point/path hashes. These records reproduce every source-study table cell. The original full raw fit/state/result trees are not bundled, so archive regeneration verifies numeric exports rather than rerunning the original full audits. Fresh runs generate new complete point files for direct inspection.
