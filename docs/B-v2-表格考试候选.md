# bank-v2 表格考试候选（仅 train/dev）

日期：2026-11-15。五题都不在 bank-v1 的 24 题中。选择阈值由总指挥在本轮 dev 校准前固定为参考 AUC 差至少 0.01；所有测量用 `tabular_harness.raw_score(..., split="dev", seed=20261114)` 在 macOS sandbox-exec 下执行。没有读取或评分 `hidden/holdout.csv`。这不是新正式考试结果。

| 任务 | 来源 / 许可证 | train / dev / holdout 行数 | Logistic 基线 dev AUC | 参考方案 dev AUC | 差值 |
|---|---|---:|---:|---:|---:|
| tab-spambase | UCI 94 / CC BY 4.0 | 2,761 / 921 / 919 | 0.9683812712 | LightGBM 0.9839129319 | 0.0155316607 |
| tab-qsar-biodegradation | UCI 254 / CC BY 4.0 | 634 / 211 / 210 | 0.9165995976 | LightGBM 0.9380281690 | 0.0214285714 |
| tab-wine-quality | UCI 186 / CC BY 4.0 | 3,905 / 1,294 / 1,298 | 0.8016064257 | LightGBM 0.8459483868 | 0.0443419611 |
| tab-ionosphere | UCI 52 / CC BY 4.0 | 211 / 70 / 70 | 0.9351111111 | LightGBM 0.9688888889 | 0.0337777778 |
| tab-monks-1 | UCI 70 / CC BY 4.0 | 260 / 86 / 86 | 0.7544618713 | LightGBM 1.0000000000 | 0.2455381287 |

Spambase、QSAR 和 Wine 先前用 LightGBM 对 AutoGluon 的 dev 参考差为负，故未入 bank-v1；本次以固定 Logistic 作为可见 `workspace/original.py` 基线，LightGBM 作为参考。新引入的 Ionosphere 与 MONK task-1 使用两份 SHA-256 钉住的 UCI 原始压缩包生成类分层、相同特征不跨 split 的 60/20/20 题包；原始下载、许可、作者、DOI、加工方式见各自 `SOURCE.md` 和 `split_manifest.json`。五份 `hidden/bank_v2_reference_measurements.json` 记录模型源码 SHA、dev AUC、沙箱秒数与 split manifest SHA。

MONK task-1 的源文件附带唯一行 ID：`source_row_id` 只出现在训练 CSV，dev/holdout 推断无此列；task.json 明确禁用，基线与参考代码均丢弃它。这是实际可验证的 ID-like 泄漏陷阱，冻结 bank-v2 时对应 exam 条目应标为 `trap=leakage`。其参考模型在 dev 恰好得到 1.0 AUC，可能乐观，须由未打开的 holdout 做独立检查。EEG Eye State 的 chronological dev 上 LightGBM AUC 0.4758、Logistic 0.5906、RandomForest 0.3388，分布漂移明显且以 LightGBM 作基线会低于随机，所以不选入五题。Mammographic Mass 和 Car Evaluation 的固定替换方案在 dev 未过门槛，退出过程见 [作废切分记录](B-v2-作废切分记录.md)。

`tabular_harness.score(..., split="dev")` 的三次重排复核中，五题可见基线均得到归一化 0，参考方案均得到 1，未出现超时或异常。测试：`tests.test_public_tabular_tasks`、`tests.test_bank_v2_tabular_tasks`、`tests.test_tabular_harness`，共 16 项通过。

**尚未完成：**五题的正式 naive 模型成本三遍校准、全新 bank-v2 冻结与一次性验证/考试。不得复用 bank-v1 的考试评分或把此 dev 表当正式成绩。
