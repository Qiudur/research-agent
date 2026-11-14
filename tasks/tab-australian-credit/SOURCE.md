# Predict Australian credit approval — data provenance

UCI Machine Learning Repository dataset 143:
https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval

Download: https://archive.ics.uci.edu/static/public/143/statlog+australian+credit+approval.zip
Pinned ZIP SHA-256: f0bd9cd9b1d133cb20456b8953e1f8139fd30ed2b3f66da540db75136f1e9e0b
Pinned source data SHA-256: 3abe5af151afa50b999a9ba21cbc884e80d80a4050b716cf88b34a2e6ecb731c

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: Source class 1 is positive; class 0 is negative.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
