# B 表格题静态与数据审计

日期：2026-11-14。审计人：队友甲（B 表格），分支 `macavity17-b-tabular`，基于提交 `c13ba01`。范围是 bank-v1 收录的 12 道表格题。本审计只读 `task.json`、`SOURCE.md`、`split_manifest.json`、`workspace/`、`hidden/dev.csv`、`hidden/reference_measurements.json` 和 `tasks/bank.json`，**没有打开任何 `hidden/holdout.csv`**，也没有修改题包。数据来源许可另见 `B-表格数据来源许可.md`。

本页先做静态检查（核对文件之间是否自洽），再做动态检查：从 UCI 重新下载原始归档、重跑准备脚本复现切分，并在真实沙箱中复测 LightGBM 基线（见文末“动态复核”）。AutoGluon 上限没有复测，原因见该节。

## 结论

| 检查项 | 结果 |
|---|---|
| 训练/dev 切分 | 12 题通过：行数、正例数与 `split_manifest.json` 一致；train 与 dev 之间没有特征完全相同的行；列集合一致（Bank 的 train 多 duration 属设计）。**从 UCI 原始归档重跑准备脚本，train.csv、dev.csv、切分清单逐字节复现** |
| 基线与上限 | 12 题通过：`bank.json` 的 `baseline_raw`、`ceiling_raw` 与 dev 实测的 LightGBM、AutoGluon AUC 逐位相同；参考测量均标明 dev-only、holdout 未打开。**LightGBM 基线在本机沙箱复测，12 题与题库差值均为 0** |
| 基线/上限对 agent 可见 | **需总指挥知悉**：表格题 `task.json` 的 `metric` 含 `baseline_raw`、`ceiling_raw`，主循环把 `metric` 传给模型，与接口约定 1.1 节不符（见“动态复核”） |
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
3. **基线和上限原始分对 agent 可见**（见下节），是否按接口约定移出 agent 可见的 `task.json`。
4. **`xlrd` 未声明**（可选）：Default Credit 的准备脚本需要 `xlrd` 读 Excel，`pyproject.toml` 没有列出。只影响重新生成题包，不影响评分与校准。

## 动态复核

在暂存目录导出提交 `11cf5bb` 的干净副本，脚本只写副本里的 `tasks/`，没有触碰工作区题包；holdout 文件虽由脚本生成，但没有读取或比对。

**原始归档**：按各题 `split_manifest.json` 记录的 URL 从 UCI 重新下载 12 个归档，SHA-256 全部与清单记录一致。

**切分复现**：用 `prepare_public_tabular.py`（Adult、Mushroom、Online Shoppers、WDBC、Heart）、`prepare_uci_tabular.py`（German、Phishing、Magic、Australian、Default、HTRU2）和 `prepare_bank_task.py`（Bank）以种子 20261114 重新生成。12 题的 `workspace/train.csv`、`hidden/dev.csv`、`split_manifest.json` 与题包逐字节相同（SHA-256 比对）。其余差异：
- 11 题的 `task.json` 不同，唯一区别是题包里 `metric.baseline_raw`、`metric.ceiling_raw` 已回填实测值，脚本生成的是 `null`；回填值与 `bank.json` 一致。Bank 的 `task.json` 完全相同。
- Adult 的 `SOURCE.md` 只有一处换行不同，文字相同。
- `prepare_public_tabular.py` 拒绝覆盖已存在的题包（防护设计），需先删除副本中的旧目录；Default Credit 需要额外安装 `xlrd`（只装在暂存目录，未进 `.venv`）。

**LightGBM 基线复测**：把各题 `workspace/original.py` 作为候选，调用 `tabular_harness.raw_score(..., "dev", seed=20261114)` 在真实 macOS 沙箱中评分，每题约 1.5 秒。12 题全部 `ok`，AUC 与 `bank.json` 的 `baseline_raw` 完全相同（差值 0.0）。本机 lightgbm 4.7.0、pandas 2.3.3、scikit-learn 1.9.1 与参考测量环境一致；Python 为 3.11.18（参考 3.11.12），操作系统版本不同。现有测试 `test_all_original_solutions_score_dev_in_sandbox` 只断言 AUC ≥ 0.5，不核对与题库是否相等，本次复测补上了这一点。

**AutoGluon 上限未复测**：每题 180 秒、12 题约 36 分钟，且 AutoGluon 的模型集成本身有非确定性，复测值不会逐位相同，只能看偏差范围。如总指挥需要，可在校准结束后单独跑。

**基线和上限对 agent 可见**：表格题 `task.json` 的 `metric` 写有 `baseline_raw`、`ceiling_raw`；评分器 `tabular_harness.py:201` 从这里读取归一化参数，主循环 `research_agent/runner.py:126` 又把整个 `metric` 放进模型的任务上下文。因此规划者、执行者能看到 LightGBM 基线 dev AUC 和 AutoGluon 上限。接口约定 1.1 节要求这些原始分只放 `bank.json`、A 加载任务时拿不到。泄露的不是标签，各对照线看到的信息相同，对比公平性不受影响；但 agent 可据此判断离上限还有多远，可能影响何时停止。修改需要同时改题包和评分器，不在本审计范围，未改动。
