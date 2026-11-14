# UCI Breast Cancer Wisconsin Diagnostic source

- Dataset page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Archive: https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip
- SHA-256: bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af
- Raw file SHA-256 hashes: see split_manifest.json.
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street, Breast Cancer Wisconsin Diagnostic [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B.
- Changes: selected/normalized features and deterministic grouped train/dev/holdout splits; details below.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

The sample ID is excluded; all 30 tissue measurements are retained. M is the positive class.
