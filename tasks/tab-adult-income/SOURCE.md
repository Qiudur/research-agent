# UCI Adult source

- Dataset page: https://archive.ics.uci.edu/dataset/2/adult
- Archive: https://archive.ics.uci.edu/static/public/2/adult.zip
- SHA-256: 7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb
- Raw file SHA-256 hashes: see split_manifest.json.
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Barry Becker and Ronny Kohavi, Adult [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20.
- Changes: selected/normalized features and deterministic grouped train/dev/holdout splits; details below.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

The public adult.data and adult.test files are combined before the new grouped
split. Question-mark fields become missing values.
