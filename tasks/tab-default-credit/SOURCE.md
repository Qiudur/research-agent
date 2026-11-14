# Predict next-month credit-card default — data provenance

UCI Machine Learning Repository dataset 350:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Download: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
Pinned ZIP SHA-256: 56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602
Pinned source data SHA-256: 30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: 1 means default next month; source ID column is removed.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
