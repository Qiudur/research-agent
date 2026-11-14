# Predict bad German credit risk — data provenance

UCI Machine Learning Repository dataset 144:
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Download: https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
Pinned ZIP SHA-256: e12d9d5def6845c0622634a1cd2ab87fa470668c4298f1ec52a4e403376a435b
Pinned source data SHA-256: b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: Source class 2 (bad credit risk) is positive; class 1 (good) is negative.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
