# UCI Mushroom source

- Dataset page: https://archive.ics.uci.edu/dataset/73/mushroom
- Archive: https://archive.ics.uci.edu/static/public/73/mushroom.zip
- SHA-256: face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a
- Raw file SHA-256 hashes: see split_manifest.json.
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Mushroom [Dataset], UCI Machine Learning Repository, https://doi.org/10.24432/C5959T.
- Changes: selected/normalized features and deterministic grouped train/dev/holdout splits; details below.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

This task deliberately uses only four visible traits, omitting odor and other near-direct identification clues. An initial eight-feature pilot achieved a perfect LightGBM dev AUC, so the final four-feature definition was narrowed before AutoGluon reference calibration. Because dev feedback shaped the task, use this package only for practice, not validation or exam.
