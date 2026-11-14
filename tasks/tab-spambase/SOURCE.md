# UCI Spambase source

- Dataset page: https://archive.ics.uci.edu/dataset/94/spambase
- Archive: https://archive.ics.uci.edu/static/public/94/spambase.zip
- SHA-256: 813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636
- Raw file SHA-256 hashes: see split_manifest.json.
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X.
- Changes: predictors renamed in source order and deterministic grouped 60/20/20 train/dev/holdout split.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

The source is unheaded; its 57 predictors are named feature_01 through
feature_57 in source order.

The pinned first dev calibration found AutoGluon AUC 0.9832661907 below
LightGBM 0.9839129319. The normalized gap is invalid, so this package is
excluded from bank-v1; its metric was unset in that snapshot. No holdout scoring occurred.

## Bank-v2 dev-only calibration

The fixed Logistic baseline in `workspace/original.py` scores 0.9683812712 AUC;
the fixed LightGBM reference in `scripts/tabular_v2_lightgbm.py` scores
0.9839129319. The dev gap is 0.0155316607, above the preregistered 0.01
minimum. Exact model hashes, sandbox seconds and split hash are recorded in
`hidden/bank_v2_reference_measurements.json`. The holdout was not opened or scored.
