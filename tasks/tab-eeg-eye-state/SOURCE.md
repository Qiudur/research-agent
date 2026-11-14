# Predict eye state from EEG signals — data provenance

UCI Machine Learning Repository dataset 264:
https://archive.ics.uci.edu/dataset/264/eeg+eye+state

Download: https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip
Pinned ZIP SHA-256: 49e7cbd14d303d63e772c7fcffa6b063bc52b97a7fa01b1135a9907b7715f415
Pinned source data SHA-256: e6eccba033d8ce56d38d2f0cca2087356705425a5a2ddbb1012ef3b892bd5a4b

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: Source eyeDetection=1 is positive; rows retain source temporal order.

Split strategy: chronological, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
