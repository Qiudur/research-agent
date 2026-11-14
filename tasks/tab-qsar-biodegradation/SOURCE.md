# Predict ready biodegradability from molecular descriptors — data provenance

UCI Machine Learning Repository dataset 254:
https://archive.ics.uci.edu/dataset/254/qsar+biodegradation

Download: https://archive.ics.uci.edu/static/public/254/qsar+biodegradation.zip
Pinned ZIP SHA-256: b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac
Pinned source data SHA-256: c66ac1a23aba11fb166fd5054fea14ac97b8e4f7e4940ae679dc863d1cc4b3c9

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). Attribution: Mansouri, K.; Ringsted, T.; Ballabio, D.;
Todeschini, R.; Consonni, V. (2013), QSAR biodegradation [Dataset], UCI
Machine Learning Repository, https://doi.org/10.24432/C5H60M. This task
redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: RB (readily biodegradable) is positive; NRB is negative.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.

## Bank-v2 dev-only calibration

The fixed Logistic baseline in `workspace/original.py` scores 0.9165995976 AUC;
the fixed LightGBM reference in `scripts/tabular_v2_lightgbm.py` scores
0.9380281690. The dev gap is 0.0214285714, above the preregistered 0.01
minimum. Exact model hashes, sandbox seconds and split hash are recorded in
`hidden/bank_v2_reference_measurements.json`. The holdout was not opened or scored.
