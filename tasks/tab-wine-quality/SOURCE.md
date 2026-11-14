# UCI Wine Quality source

- Dataset page: https://archive.ics.uci.edu/dataset/186/wine+quality
- Archive: https://archive.ics.uci.edu/static/public/186/wine+quality.zip
- SHA-256: 3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4
- Raw file SHA-256 hashes: see split_manifest.json.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

Red and white wines are combined; y=1 means source quality >= 7.

The pinned first dev calibration found AutoGluon AUC 0.8435233758 below
LightGBM 0.8459483868. The normalized gap is invalid, so this package is
excluded from bank-v1; task.metric remains unset. No holdout scoring occurred.
