# UCI Heart Disease source

- Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease
- Archive: https://archive.ics.uci.edu/static/public/45/heart+disease.zip
- SHA-256: b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc
- Raw file SHA-256 hashes: see split_manifest.json.
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano, Heart Disease [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X.
- Changes: selected/normalized features and deterministic grouped train/dev/holdout splits; details below.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

Only the 303-row processed Cleveland subset is used. The source disease severity (0-4) is binarized as any disease versus none; ? becomes missing.
