# Classify gamma-ray telescope events — data provenance

UCI Machine Learning Repository dataset 159:
https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

Download: https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip
Pinned ZIP SHA-256: 252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d
Pinned source data SHA-256: e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: g (gamma event) is positive; h (hadron event) is negative.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
