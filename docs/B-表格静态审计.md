# B 表格题静态与数据审计

日期：2026-11-14。审计人：队友甲（B 表格），分支 `macavity17-b-tabular`，基于提交 `c13ba01`。范围是 bank-v1 收录的 12 道表格题。本审计只读 `task.json`、`SOURCE.md`、`split_manifest.json`、`workspace/`、`hidden/dev.csv`、`hidden/reference_measurements.json` 和 `tasks/bank.json`，**没有打开任何 `hidden/holdout.csv`**，也没有修改题包。数据来源许可另见 `B-表格数据来源许可.md`。

本页是静态检查：核对文件之间是否自洽。用原始归档重跑准备脚本复现切分、在沙箱中复测 LightGBM 基线，属于后续动态检查，结果另行补充。

## 结论

| 检查项 | 结果 |
|---|---|
| 训练/dev 切分 | 12 题通过：行数、正例数与 `split_manifest.json` 一致；train 与 dev 之间没有特征完全相同的行；列集合一致（Bank 的 train 多 duration 属设计） |
| 基线与上限 | 12 题通过：`bank.json` 的 `baseline_raw`、`ceiling_raw` 与 dev 实测的 LightGBM、AutoGluon AUC 逐位相同；参考测量均标明 dev-only、holdout 未打开 |
| Bank Marketing duration 防护 | 通过：四项检查全部成立（见下文） |
| 弱 gap 题的分层 | **需总指挥决定**：Bank Marketing（考试）与 WDBC（验证）的参考 gap 小于 0.005，没有像 Online Shoppers、Phishing 那样限制在练习层 |
| 切分清单格式 | 轻微不一致：两个准备脚本写的字段名不同，Bank 的清单不含切分方法字段，方法只写在 `SOURCE.md` |

## 方法

逐题读取 `train.csv` 与 `dev.csv`，最后一列视为目标列，以 `"1"` 计正例。“重复行”指在 dev 与 train 共有的特征列上取值逐字相同的行，用 SHA-256 比较。参考测量与题库对比的容差为 1e-12。gap = AutoGluon AUC − LightGBM AUC；“弱 gap”的 0.005 阈值是审计人自定，用来提示归一化分数对噪声敏感，不是项目既有规则。审计脚本没有提交进仓库；可按上述规则重写复核。

## 逐题结果

AutoGluon 统一为 `medium_quality`、180 秒上限；切分种子统一为 20261114。

| 层 | 任务 | train / dev | 正例率 train / dev | LightGBM AUC | AutoGluon AUC | gap | 切分方法 |
|---|---|---:|---:|---:|---:|---:|---|
| 练习 | tab-adult-income | 29,306 / 9,768 | 0.2393 / 0.2393 | 0.929975 | 0.930998 | 0.001023 | 特征向量分组 + 分层 |
| 练习 | tab-german-credit | 600 / 200 | 0.3000 / 0.3000 | 0.726667 | 0.806905 | 0.080238 | group_stratified |
| 练习 | tab-mushroom-edibility | 5,254 / 1,384 | 0.4785 / 0.5058 | 0.850150 | 0.967920 | 0.117769 | 特征向量分组 + 分层 |
| 练习 | tab-online-shoppers | 7,399 / 2,466 | 0.1548 / 0.1549 | 0.923225 | 0.923358 | 0.000133 | 特征向量分组 + 分层 |
| 练习 | tab-phishing-websites | 6,635 / 2,211 | 0.5569 / 0.5617 | 0.993560 | 0.993571 | 0.000011 | group_stratified |
| 验证 | tab-breast-cancer-diagnostic | 343 / 113 | 0.3732 / 0.3717 | 0.990946 | 0.992622 | 0.001677 | 特征向量分组 + 分层 |
| 验证 | tab-magic-gamma | 11,413 / 3,804 | 0.6484 / 0.6483 | 0.935812 | 0.940891 | 0.005079 | group_stratified |
| 考试 | tab-australian-credit | 415 / 138 | 0.4458 / 0.4420 | 0.918459 | 0.942516 | 0.024058 | group_stratified |
| 考试 | tab-bank-marketing | 24,714 / 8,238 | 0.1130 / 0.1128 | 0.798154 | 0.802219 | 0.004065 | 分层（见 SOURCE.md） |
| 考试 | tab-default-credit | 18,001 / 6,000 | 0.2212 / 0.2208 | 0.784409 | 0.789986 | 0.005577 | group_stratified |
| 考试 | tab-heart-disease | 183 / 61 | 0.4590 / 0.4590 | 0.883117 | 0.892857 | 0.009740 | 特征向量分组 + 分层 |
| 考试 | tab-htru2 | 10,740 / 3,580 | 0.0916 / 0.0916 | 0.977710 | 0.983052 | 0.005342 | group_stratified |

“特征向量分组 + 分层”是 `split_method` 字段原文 “class-stratified 60/20/20 grouped by exact feature vector” 的简写，`group_stratified` 是另一准备脚本的 `strategy` 字段；两者描述的是同一做法。

## 需要说明的个案

- **Mushroom**：清单的 `excluded` 去掉了 18 个特征，只保留 4 个。特征少导致完全相同的特征向量很多、分组很大，实际切分比例为 64.7 / 17.0 / 18.3，正例率 train 与 dev 相差 2.7 个百分点。这是分组切分的正常结果。B-决策记录已说明它的特征选择受早期 dev 反馈影响，只放练习层，本审计同意。
- **WDBC**：`excluded` 只去掉了 `sample_id`，属于正常去除编号列。
- **Bank Marketing**：清单没有 `strategy` 或 `split_method` 字段，`SOURCE.md` 第 20 行写明是“固定的类别分层 60/20/20 随机切分”，没有提到按特征向量分组；但清单 `feature_hash_overlap=0`，本审计也没有发现 train/dev 重复行。

## Bank Marketing duration 防护

| 检查 | 结果 |
|---|---|
| `train.csv` 保留 duration（诱饵） | 是 |
| `dev.csv` 不含 duration | 是 |
| 基线 `original.py` 先丢弃 duration | 是，`drop(columns=["duration"], errors="ignore")` |
| `task.json` 明确禁止使用 | 是，写明 duration is forbidden 且推断行中不存在 |

清单记录只用 duration 在训练集上的 AUC 为 0.819829，说明诱饵确实有预测力；丢弃后的合法基线和上限分别为 0.798154 和 0.802219。

## 待总指挥决定

1. **弱 gap 题的分层**。B-决策记录以“单次 dev 差异过小”为由，把 Online Shoppers（0.000133）和 Phishing（0.000011）只放在练习层；按同样的理由，考试层的 Bank Marketing（0.004065）和验证层的 WDBC（0.001677）也偏弱。Adult Income（0.001023）在迁移复评中出现过 −2.53 的 holdout 归一化分数，说明这一量级的 gap 会让分数大幅摆动。是否调整分层、或在报告中对这两题加注，需要在冻结 bank-v1 之前定。
2. **清单格式统一**（可选）：两个准备脚本的切分字段名不同，Bank 的清单缺少方法字段。不影响评分，但统一后更便于机器核对。
