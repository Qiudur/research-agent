# UCI Spambase source

- Dataset page: https://archive.ics.uci.edu/dataset/94/spambase
- Archive: https://archive.ics.uci.edu/static/public/94/spambase.zip
- SHA-256: 813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636
- Raw file SHA-256 hashes: see split_manifest.json.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

The source is unheaded; its 57 predictors are named feature_01 through
feature_57 in source order.

The pinned first dev calibration found AutoGluon AUC 0.9832661907 below
LightGBM 0.9839129319. The normalized gap is invalid, so this package is
excluded from bank-v1; task.metric remains unset. No holdout scoring occurred.
