# Predict ready biodegradability from molecular descriptors — data provenance

UCI Machine Learning Repository dataset 254:
https://archive.ics.uci.edu/dataset/254/qsar+biodegradation

Download: https://archive.ics.uci.edu/static/public/254/qsar+biodegradation.zip
Pinned ZIP SHA-256: b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac
Pinned source data SHA-256: c66ac1a23aba11fb166fd5054fea14ac97b8e4f7e4940ae679dc863d1cc4b3c9

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: RB (readily biodegradable) is positive; NRB is negative.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
