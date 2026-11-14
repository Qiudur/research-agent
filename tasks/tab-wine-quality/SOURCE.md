# UCI Wine Quality source

- Dataset page: https://archive.ics.uci.edu/dataset/186/wine+quality
- Archive: https://archive.ics.uci.edu/static/public/186/wine+quality.zip
- SHA-256: 3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4
- Raw file SHA-256 hashes: see split_manifest.json.
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.
- Changes: red and white records combined, quality mapped to >=7, and deterministic grouped 60/20/20 train/dev/holdout split.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

Red and white wines are combined; y=1 means source quality >= 7.

The pinned first dev calibration found AutoGluon AUC 0.8435233758 below
LightGBM 0.8459483868. The normalized gap is invalid, so this package is
excluded from bank-v1; its metric was unset in that snapshot. No holdout scoring occurred.

## Bank-v2 dev-only calibration

The fixed Logistic baseline in `workspace/original.py` scores 0.8016064257 AUC;
the fixed LightGBM reference in `scripts/tabular_v2_lightgbm.py` scores
0.8459483868. The dev gap is 0.0443419611, above the preregistered 0.01
minimum. Exact model hashes, sandbox seconds and split hash are recorded in
`hidden/bank_v2_reference_measurements.json`. The holdout was not opened or scored.
