# 两道新表格题的预冻结切分作废记录

日期：2026-11-15。Ionosphere 与 Mammographic Mass 最初以 seed 20261114 生成 train/dev/holdout，仅做过 dev 校准，未运行 holdout 评分，也未进入任何冻结 bank 或正式批次。两份旧 `split_manifest.json` 的 SHA-256 分别为：

- Ionosphere：`c58a57f4c88d481d713b102d68953b93ae2c5dc4e3c2c9f87a6c7afc79622764`
- Mammographic Mass：`a3f6c6f0f8d0534ca7d94254093c8f7cd78e0f228a3a1736d26f7937db068940`

暂存检查时，Git 将新 CSV 的 CRLF 行尾识别为尾随空白，在内部工具输出中意外打印了 CSV 行。为严格保持新考试 holdout 不暴露，该初始切分及其 dev 结果全部作废，不纳入 bank-v2。总指挥在看替换 dev 结果前固定新 seed **20261115**；本分支只提交替换后的题包和 dev 测量，不提交作废 CSV。后续 Git 空白检查只覆盖 Python、Markdown、JSON；CSV 仅报告 SHA、行数和重复特征交集计数，不打印行内容。

替换切分的 Ionosphere dev 参考差为 0.0337777778，达到了预定 0.01 门槛。Mammographic Mass 在同一预定 seed 下的固定 Logistic→RandomForest dev 差为 -0.00378747，故整道题被排除，未改 seed 或模型。接着预定的 Car Evaluation（UCI 19，unacc vs 其余三类，seed 20261115，Logistic→LightGBM）首次 dev 差为 0.00715194，亦低于门槛，整道题被排除。两道未达标题均未评分 holdout；最终第五题改从独立来源 MONK's Problems task 1 按预先固定的单一配置准备。
