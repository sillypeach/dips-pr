# Data Notes

This GitHub-ready package does not include the original datasets from the
working directory.

Reasons:

- some datasets are large;
- some files are derived or locally preprocessed;
- some real-data benchmarks require separate citation or download steps;
- hard-coded local paths from exploratory scripts should not be published.

The generated-data experiments in `run_ablation.py`,
`run_heuristic_vs_certified.py`, and `examples/quick_synthetic.py` do not require
external data.

For real-data reproduction, create a local `data/` directory and adapt scripts
to load files from relative paths. Keep raw datasets out of Git unless their
license explicitly allows redistribution.

