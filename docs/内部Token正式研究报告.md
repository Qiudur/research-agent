# 冻结题库内部 token 研究报告

更新：2026-11-14。本页的“正式”指项目对冻结题库、真实模型调用和一次性阶段标记的内部研究协议；比赛没有指定模型、单题预算或现金价格。所有成本均为网关返回的四桶 token 数，**不是现金节省或官方比赛成绩**。本页纳入已审计的冻结题库、首次 practice、完整 v2 自研究及正式 batch practice；validation 六份原始报告已重算准入拒绝。一次性 exam 在 holdout 中止，完整考试成绩及最终回本点不可报告。

## 冻结题库与成本依据

- bank-v1 已冻结：提速和表格各 12 题，每场景分别为 5 道 practice、2 道 validation、5 道 exam；24 题都填有实测 naive_cost、按各场景成本四等分的 easy/medium/hard（每档 4 题），单题预算统一为自选的 40,000 token。题库冻结提交 5cb5b290cb85e6b5496479f09c830f8237725e97；当前 tasks/bank.json 的 SHA-256 为 6f6e30b8c585629a901ecda9f9fc66807565e8eb79d805a2831b6534e3a31d6c。
- 模型为可本地部署的 qwen3.5-27b，本轮经授权网关调用；config/prices.json 的 currency=TOKENS、unit_tokens=1、四桶费率均为 1、test_only=false。该表 SHA-256 为 bd42eb7295ecf0c07d2d5291ab6701b3d054a069f51d0af48b12670725f6346e。正式任务调用沿用单次输出上限 2,048 token 和请求前 12,000 token 预留；实际单题账本仍逐项核对 40,000 上限。
- 提速朴素 dev 校准 36 遍中 34 次正常完成、2 次提前错误，合计 531,929 token；表格朴素 dev 校准 36 遍中 35 次正常完成、1 次提前错误，合计 1,041,097 token。失败已花 token 计入成本分子。主工作区私有审计目录 runs/naive-cost-audit-20261114-1041-agent/ 的 audit.json 核验两组共 72/72 次尝试、holdout 尝试 0、ready=true；bank_proposal.json 是冻结前的成本与难度提案。原始 campaign 位于独立校准 worktree，runs/ 未入 Git。逐题证据见 [提速校准](提速朴素成本校准.md)、[表格校准](表格朴素成本校准.md) 和 [预先固定的合并规则](成本校准合并规则.md)。

## 首次正式 practice

原始研究目录：/Users/qizetao/Desktop/research-agent-formal-20261114/runs/auto-internal-20261114-105159-8cbb/。关键文件为 auto_plan.json、auto_self_research_report.json 和 proposal_ledger.jsonl；三个 run 的 manifest、逐题 ledger、实验树、回放上下文及练习题评分在同一正式 worktree 的 runs/run-20261114-105159-9661/、runs/run-20261114-105159-2f0d/、runs/run-20261114-105159-8c51/。代码提交 7c801cf585ed6a6df2f71d8f3d2b871a6ca8cca0，默认配置 SHA-256 为 b50bd43b43c25e40eb2006bf0abb3ed1aacd200058c15d2f43ee0a599c272d68。

计划只使用 10 道 practice 题（每场景 5 道），请求 2 轮受限候选提案、3 个并发 worker；默认配置先独立跑 3 遍，每遍 10 题。两轮模型提案均收到已计量响应，但第一轮因 rationale 无效被拒，第二轮因 planner 策略附录不安全或过大被拒；分别消耗 4,068 和 3,878 token。**selection=null，未生成可送唯一正式验证的候选配置或 selection SHA。** 这次运行没有打开 validation/exam，也没有写对应的一次性 attempt 标记。

| 默认配置遍次 | 10 题任务 token | 10 题平均归一化分数 |
|---:|---:|---:|
| 1 | 221,911 | -20.6816 |
| 2 | 240,280 | -19.9484 |
| 3 | 187,913 | -20.0141 |
| 三遍合计 / 平均 | 650,104 / 每遍 216,701.3 | -20.2147 |

30 个任务的四桶分别为 input 347,437、cached_input 0、output 66,638、reasoning 236,029，合计 **650,104 token**；提案两次四桶合计 7,946 token。因此首次 practice 的完整已记录研究成本为 **658,050 token**，不是 650,104。proposal_ledger.jsonl 的 SHA 与正式报告一致，两次提案均有返回用量；171 次任务模型调用也没有零用量记录。

独立只读复核对 30/30 个 task_report 重新执行账本、源文件与节点回放封存核验：逐题四桶之和等于成本，最高单题 34,762 token，均不超过 40,000；30 个回放上下文的输出上限均为 2,048。三遍 manifest.phase 均为 self_research；每个练习任务均有一次练习期 holdout_attempt 文件。30 个任务的停止原因分别为达到评分上限 13、预算闸门 14、错误 3；共 4 次模型调用报错且已有用量，其中 Edit Distance 两遍各有一次 JSON 语法错误，Mushroom 一遍返回多余 JSON，German Credit 一遍规划 JSON 缺字段。错误停止的任务费用未删除，不能把 30 个任务都表述为正常完成。

## 修复后 v2 practice：已完成并审计

独立工作区：/Users/qizetao/Desktop/research-agent-formal-20261114-v2/；自研究目录为 runs/auto-internal-20261114-v2/。代码提交 bf9de62c52a16893e99d5dd5d93228bdd19a8532，默认配置 SHA-256 仍为 b50bd43b43c25e40eb2006bf0abb3ed1aacd200058c15d2f43ee0a599c272d68。v2 修复了练习提案的长度与合规误判；该批次只使用 10 道 practice 题，进行了两轮提案并让同候选三遍并发。默认版、auto-01、auto-02 共 9 次运行、90 个任务均已只读审计。

| 默认配置遍次 | 10 题任务 token | 10 题平均归一化分数 |
|---:|---:|---:|
| 1 | 236,028 | -19.6775 |
| 2 | 218,265 | -31.8127 |
| 3 | 245,986 | -19.4979 |
| 三遍合计 / 平均 | 700,279 / 每遍 233,426.3 | -23.6627 |

默认版 30/30 个 task_report 的四桶合计为 input 374,214、cached_input 0、output 70,355、reasoning 255,710，逐题均未超过 40,000 token（最高 38,441）。30/30 个最佳节点的回放封存文件、源文件哈希与输出上限已只读核验；停止原因为达到评分上限 10、预算闸门 16、规划者主动停止 1、错误 3，共 4 次模型调用带错误。第 2 遍 Phishing Websites 得分 -341.6368，是极小参考 AUC 差放大的波动；剔除该题仅作解释时，其余九题三遍均分为 3.6575、2.6122、3.8571，不更改正式记录。

| auto-01 候选遍次 | 10 题任务 token | 10 题平均归一化分数 |
|---:|---:|---:|
| 1 | 218,127 | -19.9846 |
| 2 | 218,897 | -19.9657 |
| 3 | 227,576 | -20.1072 |
| 三遍合计 / 平均 | 664,600 / 每遍 221,533.3 | -20.0192 |

auto-01 配置 SHA-256 为 0930f90dad325dc63fc0583aeb233b5a945e0b3df7ab3b008548d201c696466a；四角色提示词 SHA-256 依次为 planner b6bae2137fd9909913bf2df1fac03a99f2dde236ff326468639d046556c25c90、executor 4b75443aaf378e47c2ff31b17075bce72d953dedfc3adfd84f1c1e43329433d0、diagnoser ef45cb45b6a072925221c423b89fb6e17a64da9d8d7e132cc818fccffa5c904b、summarizer 2e4c5a5039578809350da84f5078136770d89f2eade18b14fde61fa7e13bb300。30/30 个任务及最佳节点回放封存已只读核验；四桶为 input 388,914、cached_input 0、output 69,333、reasoning 206,353，最高单题 32,674 token。停止原因为评分上限 11、预算闸门 18、规划者主动停止 1，另有 3 次模型调用错误。相对默认版，三遍任务 token 均下降，但第 1、3 遍归一化分数也下降，不能据此声称稳定提升。

| auto-02 候选遍次 | 10 题任务 token | 10 题平均归一化分数 |
|---:|---:|---:|
| 1 | 220,705 | -20.0153 |
| 2 | 240,358 | -19.9495 |
| 3 | 253,460 | -19.8643 |
| 三遍合计 / 平均 | 714,523 / 每遍 238,174.3 | -19.9430 |

auto-02 配置 SHA-256 为 e9b92ef66ad1b4d6e084257c8e73987ccccce7afaca64da1c26eee95e83cead7；四角色提示词 SHA-256 依次为 planner 1b24b392c24e8898795c0e07cb4d20b18238662bfdd1b496752f8acacae2575c、executor 0a3bb86a02942847e6eaa6fb211a00afb352cb108292b5d5e4f5b436a8190aef、diagnoser a5851ec370827fdb46975028a848c8262131e7ad8479fb847bfe3c537421613b、summarizer 707cbe6170da14f0f645990b73a6aabd6214fa1047dd34eca8dbe7ad724c979a。30/30 个任务及最佳节点回放封存已只读核验，回放输出上限均为 2,048；四桶为 input 416,347、cached_input 0、output 80,527、reasoning 217,649，最高单题 34,208 token。停止原因为预算闸门 20、评分上限 8、错误 2；共有 9 次模型调用错误，已计入相应用量。

两轮提案分别消耗 6,228 与 9,098 token，合计 15,326；三组任务分别消耗 700,279、664,600 与 714,523 token，合计 2,079,402。**v2 完整已记录研究成本为 2,094,728 token。**独立核验覆盖提案 ledger 与原始响应、候选配置及提示词 SHA、9 次运行的逐题账本和四桶、报告排序与选择，以及 90/90 个最佳节点的回放封存。practice 排序的三组平均分分别为 -23.6627、-20.0192、-19.9430，自动选中 auto-02，选择记录的 beats_default_on_practice=true。但三组分数都为负，score_per_currency 均为 0，实际由平均分破同分；默认版第 2 遍 Phishing Websites 的 -341.6368 显著影响结果。该题默认版第 2 遍原始 AUC 为 0.9898697144，其余八次均为 0.9910789173；相对于下文仅约 0.00001080185 的参考差，微小的原始 AUC 变化会被大幅放大。auto-02 相比默认版仅第 2 遍分数较好、仅第 1 遍 token 较低；剔除 Phishing 只作诊断时九题三遍均分为 3.2822、3.3553、3.4500，默认版为 3.6575、2.6122、3.8571。**practice 选中不等于已证明稳定增效，更不等于 validation 准入或 exam 成绩。**

## 四线正式 batch：practice 已审计

独立工作区：/Users/qizetao/Desktop/research-agent-formal-20261114-v2/；批次目录为 runs/batch-internal-20261114-v2/，12 个原始运行在 runs/run-20261114-132520-*/。批次计划固定 naive、default、default_lib、tuned_lib 四线，各跑三遍相同的 10 道 practice 题，代码提交 bf9de62c52a16893e99d5dd5d93228bdd19a8532，模型 qwen3.5-27b。计划封存的 40 个输入文件 SHA-256 全部匹配，包括冻结 bank、四线所需配置和提示词；tuned_lib 指向 v2 练习选出的 auto-02 配置。

| 方案 | 遍次 | 10 题 token | 平均归一化分数 |
|---|---:|---:|---:|
| naive | 1 | 228,932 | -19.9809 |
| naive | 2 | 192,221 | -15.2315 |
| naive | 3 | 227,086 | -19.9019 |
| naive | 三遍合计 / 均值 | 648,239 / 216,079.7 | -18.3715 |
| default | 1 | 273,571 | -13.1086 |
| default | 2 | 238,234 | -20.2146 |
| default | 3 | 287,801 | -19.9192 |
| default | 三遍合计 / 均值 | 799,606 / 266,535.3 | -17.7474 |
| default_lib | 1 | 255,159 | -20.0315 |
| default_lib | 2 | 268,855 | -19.9850 |
| default_lib | 3 | 241,876 | -18.3649 |
| default_lib | 三遍合计 / 均值 | 765,890 / 255,296.7 | -19.4605 |
| tuned_lib | 1 | 247,385 | -20.0774 |
| tuned_lib | 2 | 246,916 | -20.1291 |
| tuned_lib | 3 | 287,626 | -19.9864 |
| tuned_lib | 三遍合计 / 均值 | 781,927 / 260,642.3 | -20.0643 |

12/12 个 run、120/120 个 task_report 已按账本、冻结配置、逐题四桶和源文件只读重算；120/120 个最佳节点回放封存再次核验通过。practice 任务四桶合计为 input 1,674,063、cached_input 0、output 338,770、reasoning 982,829，合计 **2,995,662 token**；最高单题 37,322，均低于 40,000 上限。停止原因是预算闸门 73、评分上限 37、时间 4、错误 5、规划者主动停止 1；16 次模型调用错误的已记录用量仍计入成本。tuned_lib 相比 default_lib 三遍平均分低约 0.6038、平均任务成本高约 5,345.7 token；practice 未显示研究版的稳定质量或成本优势，不据此推断唯一 validation 的准入结果。

下表每格为三遍逐题平均归一化分数 / 平均任务 token；它保留了四线在同题上的差异，而不是只展示受极小参考差影响的十题总均分。

| 题目 | Naive 分 / token | Default 分 / token | Default+库 分 / token | Tuned+库 分 / token |
|---|---:|---:|---:|---:|
| spd-distance-matrix | 1.119 / 21,060 | 0.899 / 25,678 | 1.034 / 14,793 | 0.996 / 18,005 |
| spd-rolling-mean | 1.358 / 8,656 | 1.385 / 18,897 | 1.165 / 19,494 | 1.318 / 16,995 |
| spd-max-drawdown | 2.256 / 8,661 | 2.012 / 11,505 | 2.038 / 18,531 | 1.699 / 13,321 |
| spd-edit-distance | 0.010 / 28,245 | 0.010 / 29,514 | 0.009 / 30,771 | 0.010 / 31,806 |
| spd-membership-filter | 0.993 / 10,928 | 0.987 / 29,941 | 0.978 / 28,299 | 1.073 / 23,780 |
| tab-adult-income | -2.422 / 29,652 | -2.530 / 29,462 | -2.530 / 20,773 | -2.530 / 29,846 |
| tab-mushroom-edibility | 0.303 / 21,206 | 0.303 / 29,896 | 0.549 / 32,000 | 0.180 / 31,472 |
| tab-online-shoppers | 41.694 / 28,498 | 27.846 / 26,482 | 31.021 / 28,471 | 25.586 / 31,635 |
| tab-phishing-websites | -229.693 / 28,147 | -209.258 / 33,206 | -229.693 / 31,596 | -229.693 / 33,279 |
| tab-german-credit | 0.668 / 31,026 | 0.872 / 31,954 | 0.824 / 30,569 | 0.718 / 30,503 |

表格题原始 holdout AUC 的三遍平均值及冻结参考如下；分数按参考差归一化。Default 的 Phishing Websites 第 1 遍原始 AUC 为 0.991741，其余遍次均为 0.991079；仅这一小幅原始差异就显著改变十题均分。以下参考数与原始 AUC 分开列出，避免把极小 gap 的归一化分数误读为同等幅度的 AUC 变化。

| 表格题目 | 参考 baseline / ceiling (gap) | Naive AUC | Default AUC | Default+库 AUC | Tuned+库 AUC |
|---|---:|---:|---:|---:|---:|
| tab-adult-income | 0.929975 / 0.930998 (0.00102256) | 0.927498 | 0.927388 | 0.927388 | 0.927388 |
| tab-mushroom-edibility | 0.850150 / 0.967920 (0.11776942) | 0.885786 | 0.885786 | 0.914755 | 0.871301 |
| tab-online-shoppers | 0.923225 / 0.923358 (0.00013315) | 0.928776 | 0.926932 | 0.927355 | 0.926631 |
| tab-phishing-websites | 0.993560 / 0.993571 (0.00001080) | 0.991079 | 0.991300 | 0.991079 | 0.991079 |
| tab-german-credit | 0.726667 / 0.806905 (0.08023810) | 0.780278 | 0.796667 | 0.792817 | 0.784286 |

## 唯一 validation：依据原始报告重算拒绝准入

冻结题库的唯一 validation 阶段已执行并写入一次性标记；runs/validation-bank-v1.attempt.json 记录 bank-v1、代码提交 bf9de62c52a16893e99d5dd5d93228bdd19a8532 与本批次路径。default_lib 和 tuned_lib 各三遍相同的四题，共 6/6 个完整 run、24/24 个任务。原始报告、逐题 ledger、四桶、封存的练习期库和最佳节点回放均已只读复核；最高单题 35,042 token，未超 40,000。四桶合计 input 262,805、cached_input 0、output 33,100、reasoning 134,808，共 **430,713 token**。但最终 batch_report.json 因随后考试中止而未生成；以下准入是按照冻结规则从六份已完成原始报告**独立重算**，不是声称存在一个已保存的 batch_report.admission 字段。

| 验证方案 | 遍次 | 四题 token | 四题平均归一化分数 |
|---|---:|---:|---:|
| default_lib | 1 | 74,877 | 0.7449 |
| default_lib | 2 | 67,648 | 0.5827 |
| default_lib | 3 | 70,068 | 0.7179 |
| default_lib | 三遍合计 / 均值 | 212,593 / 70,864.3 | 0.6819 |
| tuned_lib | 1 | 66,312 | 0.5448 |
| tuned_lib | 2 | 67,406 | 0.3875 |
| tuned_lib | 3 | 84,402 | -0.1091 |
| tuned_lib | 三遍合计 / 均值 | 218,120 / 72,706.7 | 0.2744 |

准入规则要求研究版三次配对分数均不低于默认带库版，且平均任务成本更低。tuned_lib 的三次分数差依次为 -0.2001、-0.1952、-0.8270，平均任务 token 高约 1,842.3，两项门槛均失败，重算结论为 **rejected**。下表每格为三遍逐题平均归一化分数 / 平均任务 token；表格题另列原始 AUC 与冻结参考差。

| 验证题 | Default+库 分 / token | Tuned+库 分 / token |
|---|---:|---:|
| spd-prefix-sum | 1.8121 / 9,004.0 | 1.0561 / 13,871.7 |
| spd-dedupe-order | 1.4365 / 9,727.0 | 1.2293 / 7,016.0 |
| tab-breast-cancer-diagnostic | 0.2000 / 20,125.7 | -0.4667 / 20,067.7 |
| tab-magic-gamma | -0.7212 / 32,007.7 | -0.7212 / 31,751.3 |

| 表格验证题 | 参考 baseline / ceiling (gap) | Default+库原始 AUC | Tuned+库原始 AUC |
|---|---:|---:|---:|
| tab-breast-cancer-diagnostic | 0.990946 / 0.992622 (0.00167673) | 0.991281 | 0.990163 |
| tab-magic-gamma | 0.935812 / 0.940891 (0.00507864) | 0.932150 | 0.932150 |

## 一次性 exam：holdout 中止、无完整成绩

runs/exam-bank-v1.attempt.json 和批次内 exam_attempt.json 均已封存一次性正式考试尝试。按验证准入拒绝后的执行路径，exam 启动 naive、default、default_lib 三线各三遍；tuned_lib 没有进入考试。九个 run 中八个形成完整 report，共 80 题；default 第 3 遍 run-20261114-145216-0e55 仅前六题形成完整 task_report。该运行在 tab-heart-disease 的最佳节点 n0003 停止后写入 holdout_attempt.json，但缺 holdout_score.json 与 task_end；其后的 tab-htru2、tab-default-credit、tab-australian-credit 均未启动。该 run 没有 report.json，批次也没有 batch_report.json，主全量审计因此明确返回 ready=false。失败题的最佳节点与源文件封存 SHA、2,048 输出上限仍可核对通过；持久化产物未给出可据以归因的 holdout 异常详情。

八个完整 run 的 80/80 题通过独立运行账本核验；未完 run 的前六题也通过逐题账本、预算及回放复核，共 86/86 个已完成任务，最高单题 35,635 token。已完成 86 题的四桶记录为 input 980,952、cached_input 0、output 159,562、reasoning 449,731，合计 1,590,245 token；失败 Heart Disease 题的五次模型调用另有 input 21,803、cached_input 0、output 2,797、reasoning 4,471，合计 29,071。**当前可核验的考试模型调用已记录用量为 1,619,316 token，仅是中止批次的部分账本，不是完整考试成本。**因缺少 default 第 3 遍的完整结果和最后三题，三线完整考试总分、均分、成本比较及回本点均不可计算；不得把八份完整 run 当成九份或补零。

## 分数解释与独立复核边界

Phishing Websites 的冻结参考 baseline AUC 为 0.9935600225、ceiling AUC 为 0.9935708244，差仅约 0.00001080185。B 的 gap 归一化按 (raw-baseline)/(ceiling-baseline) 计算；首次 practice 三遍该题均为 -229.6927，对应原始 AUC 中位数约 0.991079，比 baseline 低约 0.002481。这个很小的参考分母主导了十题均分的极端负值；剔除该题只作解释、不改正式成绩时，其余九题均分分别为 2.5419、3.3565、3.2835。Adult Income、Online Shoppers 的参考 AUC 差也只有约 0.001023、0.000133；验证题 Breast Cancer Diagnostic 的差约 0.001677。正式报告须并列给出原始 AUC、参考差及逐题归一化分数，不凭单一均分宣称整体能力或稳定提升。

- [成员乙十题基准](A-十题练习试验.md)来自 Git 提交 633f4ad 的 runs/token-practice10-default-20261114-091613-02ab/：同一组 10 题、同一模型和 40,000/2,048/12,000 token 限制，默认版三遍记录 758,055 token，平均分 -19.6670。但其代码提交为 482f3ef，默认配置 SHA 为 f0ba329，采用 TEST 价目及另一网关；这不是与本轮冻结配置、同一代码和同一环境配对的实验。不能把两组记录差额解释为 107,951 token 的节省或质量改善。成员乙的 Phishing 三遍也为 -229.6927、-206.4446、-229.6927，显示极小参考 gap 的波动并非本批次独有。
- [成员甲表格复核](队友甲表格成本独立复核.md)的完整批次记录 36 次、35 次正常完成，账面合计 1,025,623 token；原始压缩包备份于主工作区私有的 runs/external/member-a-tabular-20261114/runs.zip，逐题审计见 [JSON 摘要](队友甲表格成本独立复核.json)。其代码提交 8340996、网关与冻结校准不同，Default Credit 有一次 HTTP 400 未返回用量；1,025,623 仅为已记录用量。跨 campaign 审计明确返回 Calibration commits differ outside docs/，未生成可替换 bank 的提案。这是一组独立交叉复核，不能与本机校准合并为精确朴素成本，也不能据其差额归因于配置。

## 结果边界与待补项

| 阶段 | 当前可报告状态 | 依据或待补证据 |
|---|---|---|
| v2 practice 自研究 | **已完成并审计**；选中 auto-02，仅为练习集结果 | 9 次运行、90 题、2 轮提案、回放封存、完整 ranking/selection 与 2,094,728 token 总成本 |
| 四线正式 batch practice | **已完成并审计**；120/120 题 | 四线各三遍、逐题归一化分数与 token、表格原始 AUC、冻结参考及 2,995,662 token 总成本 |
| 唯一 validation 准入 | **依据六份原始报告重算为拒绝**；最终 batch_report 未生成 | 一次性 attempt 标记、六份原始 run、24/24 题、配对分数与成本；研究版三遍分数均低且平均成本更高 |
| 一次性 exam | **中止，无完整成绩** | 九个 run 中八个完整；第九个六题完成、一题 holdout 失败、三题未启动；batch_report 缺失 |
| 图表、回本点与路演结论 | **完整考试与回本点不可报告** | 仅能展示明确标为部分结果的账本；不能据缺失的考试三线成绩计算研究版回本点或声称稳定提升 |

完整考试缺口不能由 TEST 联调、成员甲乙独立试验、首次失败的提案或练习集自身分数代替。正式考试总成绩和现金价格没有完整实测，应继续保持空缺；已封存的一次性 attempt 不得重跑冒充同一次考试。
