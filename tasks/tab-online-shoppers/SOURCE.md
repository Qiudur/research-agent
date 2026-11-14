# UCI Online Shoppers Purchasing Intention source

- Dataset page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Archive: https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip
- SHA-256: 2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14
- Raw file SHA-256 hashes: see split_manifest.json.

Preparation: scripts/prepare_public_tabular.py, seed 20261114. Exact feature-vector duplicates stay in one split; train/dev/holdout are class-stratified 60/20/20. Hidden holdout is never read during calibration.

Baseline: sandboxed default LightGBM. Reference: AutoGluon Tabular medium_quality, 180-second cap, measured on dev only. See hidden/reference_measurements.json after calibration.

Boolean source fields are normalized to 1/0 strings before splitting.
