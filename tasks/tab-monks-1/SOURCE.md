# UCI MONK's Problems, task 1 source

- Dataset page: https://archive.ics.uci.edu/dataset/70/monk+s+problems
- Archive: https://archive.ics.uci.edu/static/public/70/monk+s+problems.zip
- Archive SHA-256: 7501acef87251b1d427726d7c381a2488c166fbbf70ac4096e09f05951c30a29
- Source data SHA-256: e501ada3b21d876ab86f3e1614747bd2edc443a685f229a25e480aa1dea629f6
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Wnek, J. (1993). MONK's Problems [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R30R.
- Changes: deterministic grouped 60/20/20 train/dev/holdout split; original column names normalized, labels mapped to 0/1, missing markers normalized where present.

Preparation: scripts/prepare_bank_v2_tabular.py, seed 20261115; split counts and duplicate check in split_manifest.json. Calibration uses only train and dev; generated holdout must stay unopened.

The original `monks-1.test` is treated only as the pinned source table; this project makes new train/dev/holdout splits. Its final source row ID is visible in training only as an explicit ID-like leakage trap. The scorer omits it from dev/holdout inference features, and task.json forbids using it; the baseline and reference drop it.

## Bank-v2 dev-only calibration

The fixed Logistic baseline in `workspace/original.py` scores 0.7544618713 AUC;
the fixed LightGBM reference in `scripts/tabular_v2_lightgbm.py` scores 1.0.
The dev gap is 0.2455381287, above the preregistered 0.01 minimum. Exact
model hashes, sandbox seconds and split hash are recorded in
`hidden/bank_v2_reference_measurements.json`. A perfect dev reference can be
optimistic; the hidden holdout is the independent check and was not opened or scored.
