# UCI Ionosphere source

- Dataset page: https://archive.ics.uci.edu/dataset/52/ionosphere
- Archive: https://archive.ics.uci.edu/static/public/52/ionosphere.zip
- Archive SHA-256: 4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860
- Source data SHA-256: 46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd
- License on UCI page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
- Attribution: Sigillito, V., Wing, S., Hutton, L., & Baker, K. (1989). Ionosphere [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W01B.
- Changes: deterministic grouped 60/20/20 train/dev/holdout split; original column names normalized, labels mapped to 0/1, missing markers normalized where present.

Preparation: scripts/prepare_bank_v2_tabular.py, seed 20261115; split counts and duplicate check in split_manifest.json. Calibration uses only train and dev; generated holdout must stay unopened.

## Bank-v2 dev-only calibration

The fixed Logistic baseline in `workspace/original.py` scores 0.9351111111 AUC;
the fixed LightGBM reference in `scripts/tabular_v2_lightgbm.py` scores
0.9688888889. The dev gap is 0.0337777778, above the preregistered 0.01
minimum. Exact model hashes, sandbox seconds and split hash are recorded in
`hidden/bank_v2_reference_measurements.json`. The holdout was not opened or scored.
