# Reference numbers for the 1 October 2026 manuscript

These are archived numerical results, not new fits performed when this release
was prepared. `python reproduce.py tables --output results/paper_tables`
recomputes the means and sample standard deviations from their individual
seed/split values and exports CSV and LaTeX tables. Full retraining uses the
separate experiment commands in the repository README.

* Tables 1 and 2: C2, C4, C8, and L2–5, each with all five seeds. Prediction
  uses training-only selection. The DIPS-PR recovery row selects the maximum
  F1 along the complete path using generating truth; LCM+Lasso and RF retain
  prediction-oriented selection. This is not a matched oracle comparison.
  Historical DIPS prediction-selected recovery values remain in the source
  JSON for transparency, but are not exported as a manuscript table row.
* Table 3: seven complete conditions, 35 outer-training paths. The rate is
  `1 - sum(D-U)/(M*sum(R))` per path, followed by an equal-weight five-seed
  mean and sample SD. The null endpoint with no traversals is NA.
* Table 4: v, u, and the final v+u configuration on C2, C4, and L2–5. The
  full five-arm study remains in the JSON; the displayed single components
  use one reference and the final configuration uses two. This display does
  not isolate the incremental benefit of the second reference. Archived
  runtimes describe the original machine, not runtime guarantees.
* HIV: all 25 endpoints and eight methods, with five grouped splits
  (seeds 42–46). Standard deviations describe split variation, not confidence
  intervals or independent-repeat standard errors. NRTI is the main table;
  the other drug classes belong in the appendix.
* The two term CSVs contain aggregate coefficients and mutation annotations
  for the representative seed-42 fits. They contain no patient identifiers
  or individual HIV response records. Position-presence rules are annotated
  by observed mutation profiles; these are not mutation-specific fitted
  coefficients or experimentally validated biological interactions.

`source_hashes.json` identifies the approved numerical supplement. Original
machine paths were omitted from the ablation metadata. No numerical values
were changed. Additional retained archival values do not expand the scope of
the current paper. Failed or incomplete conditions are not silently included
in a five-seed table.
