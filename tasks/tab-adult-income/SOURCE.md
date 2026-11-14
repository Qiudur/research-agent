# UCI Adult source

- Dataset page: https://archive.ics.uci.edu/dataset/2/adult
- Archive: https://archive.ics.uci.edu/static/public/2/adult.zip
- SHA-256: 7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb
- Raw file SHA-256 hashes: see split_manifest.json.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

The public adult.data and adult.test files are combined before the new grouped
split. Question-mark fields become missing values.
