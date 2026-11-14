# B 表格题数据来源许可审计

日期：2026-11-14。审计人：队友甲（B 表格），分支 `macavity17-b-tabular`，基于提交 `c13ba01`。范围是 bank-v1 收录的 12 道表格题；未入库的 4 道只在文末简记。本次没有打开任何 holdout 数据，也没有修改题包文件。

## 结论

12 道入库表格题都来自 UCI Machine Learning Repository，许可证均为 **Creative Commons Attribution 4.0 International（CC BY 4.0）**，允许改编（含重新切分和特征处理）及再分发，前提是署名并注明做了改动。其中 7 道已在各自的 `SOURCE.md` 写明许可证；另外 5 道的 `SOURCE.md` 漏写，本次逐页核对了 UCI 数据集页面，许可证同为 CC BY 4.0。许可证本身没有障碍，缺的只是 5 份题包的书面记录。

## 逐题

“依据”列说明许可证的出处：`SOURCE.md` 表示题包内已有记录（本次未逐页复核 UCI 页面）；“UCI 页面”表示 `SOURCE.md` 漏写，本次直接核对了数据集页面，页面原文为 “This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.”

| 层 | 任务 | UCI 编号 | 许可证 | 依据 | DOI / 引用 |
|---|---|---:|---|---|---|
| 练习 | tab-adult-income | 2 | CC BY 4.0 | UCI 页面 | 10.24432/C5XW20；Becker & Kohavi (1996) |
| 练习 | tab-german-credit | 144 | CC BY 4.0 | SOURCE.md | SOURCE.md 未记 |
| 练习 | tab-mushroom-edibility | 73 | CC BY 4.0 | UCI 页面 | 10.24432/C5959T；页面引用无作者，“Mushroom [Dataset]. (1981)” |
| 练习 | tab-online-shoppers | 468 | CC BY 4.0 | UCI 页面 | 10.24432/C5F88Q；Sakar & Kastro (2018) |
| 练习 | tab-phishing-websites | 327 | CC BY 4.0 | SOURCE.md | SOURCE.md 未记 |
| 验证 | tab-breast-cancer-diagnostic | 17 | CC BY 4.0 | UCI 页面 | 10.24432/C5DW2B；Wolberg, Mangasarian, Street & Street (1993) |
| 验证 | tab-magic-gamma | 159 | CC BY 4.0 | SOURCE.md | SOURCE.md 未记 |
| 考试 | tab-australian-credit | 143 | CC BY 4.0 | SOURCE.md | SOURCE.md 未记 |
| 考试 | tab-bank-marketing | 222 | CC BY 4.0 | SOURCE.md | Moro, Cortez & Rita (2014)，10.1016/j.dss.2014.03.001 |
| 考试 | tab-default-credit | 350 | CC BY 4.0 | SOURCE.md | SOURCE.md 未记 |
| 考试 | tab-heart-disease | 45 | CC BY 4.0 | UCI 页面 | 10.24432/C52P4X；Janosi, Steinbrunn, Pfisterer & Detrano (1989) |
| 考试 | tab-htru2 | 372 | CC BY 4.0 | SOURCE.md | SOURCE.md 未记 |

各题的下载地址、归档 SHA-256 和切分方法见 `tasks/<task_id>/SOURCE.md` 与 `split_manifest.json`，两处记录的来源 URL 与上表 UCI 编号一致。

## 待总指挥决定

1. **补写 5 份 `SOURCE.md`**（adult-income、mushroom-edibility、online-shoppers、breast-cancer-diagnostic、heart-disease）：加上许可证、DOI 和引用。这会改变题包文件哈希，所以本次没有改；建议在冻结 bank-v1 之前由总指挥统一补写，避免校准中途两组题包哈希不一致。
2. **CC BY 4.0 的署名义务**：对外展示或分发题包（答辩材料、公开仓库）时，要署名原作者并说明“已重新切分为 60/20/20、部分题做了特征选择或目标映射”。现有 `SOURCE.md` 已记录改动内容，但答辩材料里还没有统一的致谢页。
3. **引用信息不统一**：6 道已写许可证的题只写了“作者和引用见 UCI 页面”，没有记录 DOI。如果报告需要正式引用，建议一起补齐。

## 未入库题（不在本次范围）

`tab-eeg-eye-state`、`tab-qsar-biodegradation` 的 `SOURCE.md` 已写许可证；`tab-spambase`、`tab-wine-quality` 没写，本次未核对。这 4 道因参考 gap 非正没有进入 bank-v1，不影响正式评估。
