# Identify pulsar candidates — data provenance

UCI Machine Learning Repository dataset 372:
https://archive.ics.uci.edu/dataset/372/htru2

Download: https://archive.ics.uci.edu/static/public/372/htru2.zip
Pinned ZIP SHA-256: ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379
Pinned source data SHA-256: b13b4d8929e96ecd196e464c1c8a454c3ac2ffa631015f6388957531a9923f59

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: 1 means pulsar candidate; 0 means non-pulsar.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
