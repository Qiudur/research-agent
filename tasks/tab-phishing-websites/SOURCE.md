# Predict the UCI phishing-website source label — data provenance

UCI Machine Learning Repository dataset 327:
https://archive.ics.uci.edu/dataset/327/phishing+websites

Download: https://archive.ics.uci.edu/static/public/327/phishing+websites.zip
Pinned ZIP SHA-256: 514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940
Pinned source data SHA-256: 3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6

The UCI dataset page states Creative Commons Attribution 4.0 International
(CC BY 4.0). The original creators and citation are credited on the linked
UCI page. This task redistributes a deterministic derived train/dev/holdout
split under that license. Target mapping: Source Result=1 is positive and Result=-1 is negative.

Split strategy: group_stratified, seed 20261114; identical feature vectors do not
cross splits. The target y is available only in workspace/train.csv. Candidate
code receives hidden test features in memory and cannot read hidden labels.
The reference measurement uses dev only; holdout remains unopened.
