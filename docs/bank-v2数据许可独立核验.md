# bank-v2 新表格题来源与许可独立核验

核验时间：2026-11-15（Asia/Shanghai）。范围为五道拟用于新考试的 UCI 数据集；**仅核对 UCI 官方元数据与归档哈希，没有打开、运行或输出任何 holdout CSV 或 CSV 行**。bank-v2 题库和考试尚未冻结。

| 候选题 | UCI dataset ID | 官方 DOI | UCI 标示许可 | 官方 ZIP SHA-256（流式计算） | 题包归因状态 |
|---|---:|---|---|---|---|
| Spambase | 94 | 10.24432/C53G6X | CC BY 4.0 | `813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636` | SOURCE.md 已列作者、年份、DOI、变换与许可。 |
| QSAR Biodegradation | 254 | 10.24432/C5H60M | CC BY 4.0 | `b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac` | 题包初稿只笼统指向 UCI 原作者；已请求补明 Mansouri 等五位作者、2013 与 DOI，待最终 SHA 核查。 |
| Wine Quality | 186 | 10.24432/C56S3T | CC BY 4.0 | `3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4` | SOURCE.md 已列作者、年份、DOI、二分类变换与许可。 |
| Ionosphere | 52 | 10.24432/C5W01B | CC BY 4.0 | `4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860` | SOURCE.md 已列作者、年份、DOI、变换与许可；旧 split 已作废，须只审新 seed 的最终 SHA。 |
| MONK's Problems | 70 | 10.24432/C5R30R | CC BY 4.0 | `7501acef87251b1d427726d7c381a2488c166fbbf70ac4096e09f05951c30a29` | 题包和 dev gap 待交付；官方归档文件名为 `monk+s+problems.zip`，须记录所取子文件、作者 Wnek（1993）、二分类目标、许可与重切分。 |

UCI 官方页面分别为 `https://archive.ics.uci.edu/dataset/94/spambase`、`/dataset/254/qsar+biodegradation`、`/dataset/186/wine+quality`、`/dataset/52/ionosphere`、`/dataset/70/monks+problems`（后四个同域）。五页均明确标示 Creative Commons Attribution 4.0 International；该许可允许署名条件下的再分发，题包应保留出处、许可链接与加工说明。以上 ZIP 哈希是直接从官方 `/static/public/{id}/...zip` 流式 SHA-256 计算，与已交付的前四题 SOURCE.md 声明一致。

**冻结闸门：**MONK dev gap 必须按预注册达到 0.01；最终五份 SOURCE.md 逐一确认作者、DOI、许可证及加工说明。`scripts/propose_bank_v2.py` 的 `bank_v2_license_review/1` 须在真实 proposal 生成后，用其 SHA-256 和最终 SOURCE.md SHA-256 绑定五题；目前**不得签发 ready=true 的许可复核**。Car Evaluation 与 Mammographic Masses 因 dev gap 不达标或旧 split 作废而不在本审计范围。
