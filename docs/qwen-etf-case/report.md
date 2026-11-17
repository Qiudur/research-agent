# Qwen ETF experiment: no candidate improved on the original signal

[English](#english) · [中文](#中文)

<a id="english"></a>

Across two planning rounds, Qwen proposed four directions and executed and scored three candidates. All three underperformed the original five-session momentum rule. The Agent then chose to stop and retained the original signal, `n0000`. The fourth direction, scaling five-session momentum by ten-session volatility, was not executed and is not counted as a tested result. [Planning and stopping decisions](research-decisions.json) preserve that distinction. **This experiment provides no evidence that Qwen improved the strategy.**

The original signal returned +6.07% net in the holdout, versus -12.47% for daily equal weighting, a difference of +18.54 percentage points. The Agent-selected signal and the original signal have identical daily evidence, so the improvement over the original is **0.00 percentage points**. This performance belongs to the original rule and cannot be attributed to a model-generated strategy improvement.

## Frozen protocol and scope

- Run: `qwen-etf-dev-20261117-05`; development status: `dev_complete`; holdout status: `scored`.
- Protocol: [v3 snapshot](protocols/qwen_etf_comparison_v3.json), SHA-256: `62219b3c3ac0aa7130930e2aa2d814187430e54a94032c167beeef5d5b2caf10`.
- Execution code commit: `e4dff91ae7226543e2e61c04df98b85b1ee9e675`; model: `qwen3.5-27b`.
- Fixed ETFs: EEM, EFA, GLD, IEF, IWM, QQQ, SPY and TLT. Top 2, equally weighted at 50% each. The comparator holds all eight at 12.5% each, rebalanced daily.
- Training data ends on 2014-12-29; development is 2015-01-05–2017-12-29 (754 sessions); holdout is 2020-01-06–2020-04-01 (61 sessions).
- The 2018–2019 interval had already been exposed in the fixed-grid study and was quarantined from this selection and holdout evaluation.
- Signals are generated after the t close, traded at the t+1 close and realize returns at t+2. One-way fees are 2 bps and slippage is 3 bps, including initial entry costs.
- Budget: 40,000 TOKENS; maximum output per call: 6,000 tokens; wall-clock limit: 3,600 seconds; request timeout: 600 seconds; no automatic retry of an individual request.
- The v1 wall-clock/request limits were 900/180 seconds; v2 raised them to 3,600/600 seconds. V3 retained those settings and started from the user's terminal with updated credentials. There is one completed development repetition; all four earlier failures are retained and reported separately.
- The dataset manifest and price CSV have SHA-256 values `85c27ca02f0df99fcd4759b13d7094189afa803b7997616cafe22e9c83b101a7` and `38f1025dfa0197c05fb8e631d1ac191fed831f5f6c3135ad2403e75983e069d5`, respectively. They identify different files.

## All executed development results

The development score is the net period return minus the daily equal-weight net period return; higher is better. The original rule also lagged equal weighting, but remained the highest-scoring accepted node under the frozen selection rule.

| Node | Hypothesis | Status | Net period return | Excess over equal weight (percentage points) |
| --- | --- | --- | ---: | ---: |
| n0000 | Baseline | accepted | +4.20% | -22.29 |
| n0001 | Extend momentum lookback window from 5 to 20 trading sessions. | rejected | +0.25% | -26.24 |
| n0002 | Invert baseline signal to implement short-term reversal strategy. | rejected | +2.16% | -24.33 |
| n0003 | Implement cross-sectional ranking of 5-day momentum across all symbols. | rejected | -10.82% | -37.32 |

[Development terminal report](dev_report.json) · [Complete experiment tree](tree.jsonl) · [Frozen selection](selection.json) · [All node code and scores](evidence-index.json)

## The single holdout evaluation

| Strategy | Net period return | Maximum drawdown |
| --- | ---: | ---: |
| Agent-selected: original five-session momentum | +6.07% | 6.72% |
| Original five-session momentum | +6.07% | 6.72% |
| Daily equal weight | -12.47% | 20.96% |

Each of the three holdout lines contains 61 observations of dates, positions, returns after costs and turnover, plus 62 equity values including the initial value. The first two lines have identical code and complete daily evidence. The [original holdout report](report.json) and [single-attempt marker](holdout-attempt.json) are preserved byte for byte. The candidate was not changed or scored again after evaluation.

## Model calls and resources

| Order | Role / purpose | Input | Cached input | Output | Reasoning | Total tokens | Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | planner / plan | 821 | 0 | 416 | 4,661 | 5,898 | 523.7 |
| 2 | executor / code | 757 | 0 | 139 | 565 | 1,461 | 98.8 |
| 3 | executor / code | 741 | 0 | 119 | 381 | 1,241 | 45.8 |
| 4 | planner / plan | 961 | 0 | 391 | 4,451 | 5,803 | 366.2 |
| 5 | executor / code | 734 | 0 | 182 | 826 | 1,742 | 126.3 |
| 6 | planner / review | 760 | 0 | 112 | 774 | 1,646 | 57.7 |
| Total | 6 calls, all successful with complete usage | 4,774 | 0 | 1,359 | 11,658 | 17,791 | — |

The four token buckets are disjoint: reasoning tokens are not counted again as output. TOKENS measure resources; cash cost is unknown. The 17,791 tokens belong only to this successful development run and are not the total cost of all attempts, including earlier failures. [Per-call ledger](task-ledger.jsonl)

## All earlier failures

| Run | Verifiable failure class | Terminal status | Total tokens | Holdout scored |
| --- | --- | --- | --- | --- |
| qwen-etf-dev-20261117-01 | http_401 | failed | Unknown | No |
| qwen-etf-dev-20261117-02 | http_401 | failed | Unknown | No |
| qwen-etf-dev-20261117-03 | request_timeout_180_seconds | failed | Unknown | No |
| qwen-etf-dev-20261117-04 | http_401 | failed | Unknown | No |

None of the four failures returned trustworthy usage, so their total tokens remain unknown and cannot be treated as zero. Reporting only the successful run would omit these attempts. The `-04` attempt received 401 because the agent process still had old environment credentials; `-05` started from the same terminal in which the user had updated the credentials. The [failure summary and local ledger hashes](prior-attempts.json) retain each failure class, elapsed time, usage status and failure report.

## Evidence and checks

[evidence-index.json](evidence-index.json) maps original run paths to published-copy paths and SHA-256 values. The development report, selection, ledgers, tree, node code and scores, configuration, prompts and protocol snapshots can all be checked by hash. Report and source-code artifacts are exact copies; original `runs/` paths refer to local execution directories.

Before the holdout was scored, admission checks verified successful development completion, complete usage, the highest accepted node, matching input and frozen-artifact hashes, and the absence of an earlier holdout attempt. After scoring, an independent check verified daily compounded returns, 5 bps costs, position turnover and summary metrics for all three lines; the arithmetic agreed. The data provenance manifest records sources and price hashes. Raw and normalized prices and raw model responses remain local and are excluded from the publication bundle. The bundle supports evidence review; rerunning the model may produce different candidates, and an already exposed interval cannot become unseen again.

## Interpretation limits

This is one exploratory development run followed by a single 61-session historical holdout observation. The interval includes the sharp market moves of early 2020 and cannot establish long-term stability, live execution or capacity; short-period annualized figures are not emphasized. The fixed surviving-ETF universe, retrospective price adjustments, closing-price fills and constant trading costs also limit extrapolation. See [data sources and rights](../etf-data-sources.md).

The supported finding is that the audit trail completed real model calls and a single holdout evaluation while preserving the negative result: the model research did not improve the original signal.

---

<a id="中文"></a>

# Qwen ETF 实验：未找到优于原始信号的候选

Qwen 在两轮规划中提出四个方向，执行并评分三个候选，均未超过原始 5 日动量，随后主动停止，最终保留 `n0000` 原始信号。第四个“以 10 日波动率缩放 5 日动量”方向没有执行，不计入已测试结果。[规划与停止决策](research-decisions.json) 保留了这些区别。**本次没有证据表明 Qwen 改进了策略。**

封存期原始信号净收益为 +6.07%，每日等权为 -12.47%，相差 +18.54 个百分点；Agent 入选方案与原始信号逐日完全相同，相对原始方案的改进为 **0.00 个百分点**。该表现属于原始规则，不能归为模型生成策略的提升。

## 冻结协议与范围

- 运行：`qwen-etf-dev-20261117-05`；开发状态 `dev_complete`；封存状态 `scored`。
- 协议：[v3 快照](protocols/qwen_etf_comparison_v3.json)，SHA-256：`62219b3c3ac0aa7130930e2aa2d814187430e54a94032c167beeef5d5b2caf10`。
- 执行代码 commit：`e4dff91ae7226543e2e61c04df98b85b1ee9e675`；模型：`qwen3.5-27b`。
- 固定 ETF：EEM、EFA、GLD、IEF、IWM、QQQ、SPY、TLT；Top-2，等权各 50%。比较线每日持有全部八只、各 12.5%。
- 训练数据截至 2014-12-29；开发期 2015-01-05—2017-12-29（754 日）；封存期 2020-01-06—2020-04-01（61 日）。
- 2018—2019 已在固定网格研究中暴露，予以隔离，不用于这次选优或封存评分。
- 信号在 t 日收盘后生成，t+1 收盘成交，t+2 实现收益；单边手续费 2 bps、滑点 3 bps，包含初次建仓成本。
- 预算 40,000 TOKENS；每次最大输出 6,000 token；墙钟上限 3,600 秒；请求超时 600 秒；单次请求不自动重试。
- v1 的墙钟与请求上限为 900 / 180 秒；v2 延长至 3,600 / 600 秒；v3 保留这些设置，改从更新过凭据的用户终端启动。只有一个完整开发 repetition；前四次失败均保留并单独报告。
- 数据 manifest 与价格 CSV 的 SHA-256 分别为 `85c27ca02f0df99fcd4759b13d7094189afa803b7997616cafe22e9c83b101a7`、`38f1025dfa0197c05fb8e631d1ac191fed831f5f6c3135ad2403e75983e069d5`。二者对应不同文件。

## 开发期全部已执行结果

开发分数是区间净收益减每日等权的区间净收益，越高越好。原始规则也落后等权；按冻结规则，它仍是开发期最高分的 accepted 节点。

| 节点 | 假设 | 状态 | 区间净收益 | 相对等权（百分点） |
| --- | --- | --- | ---: | ---: |
| n0000 | Baseline | accepted | +4.20% | -22.29 |
| n0001 | Extend momentum lookback window from 5 to 20 trading sessions. | rejected | +0.25% | -26.24 |
| n0002 | Invert baseline signal to implement short-term reversal strategy. | rejected | +2.16% | -24.33 |
| n0003 | Implement cross-sectional ranking of 5-day momentum across all symbols. | rejected | -10.82% | -37.32 |

[开发终态](dev_report.json) · [完整实验树](tree.jsonl) · [冻结选择](selection.json) · [全部节点源码和评分](evidence-index.json)

## 唯一一次封存评分

| 策略 | 区间净收益 | 最大回撤 |
| --- | ---: | ---: |
| Agent 入选：原始 5 日动量 | +6.07% | 6.72% |
| 原始 5 日动量 | +6.07% | 6.72% |
| 每日等权 | -12.47% | 20.96% |

封存三线各有 61 组日期、持仓、费用后的收益和换手，净值含初值共 62 点。前两线的代码和完整逐日证据相同。[原始封存报告](report.json) 和 [唯一尝试标记](holdout-attempt.json) 均按原字节保留；评分后没有修改候选或再次评分。

## 模型调用与资源

| 次序 | 角色 / 用途 | 输入 | 缓存输入 | 输出 | 推理 | 合计 token | 秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | planner / plan | 821 | 0 | 416 | 4,661 | 5,898 | 523.7 |
| 2 | executor / code | 757 | 0 | 139 | 565 | 1,461 | 98.8 |
| 3 | executor / code | 741 | 0 | 119 | 381 | 1,241 | 45.8 |
| 4 | planner / plan | 961 | 0 | 391 | 4,451 | 5,803 | 366.2 |
| 5 | executor / code | 734 | 0 | 182 | 826 | 1,742 | 126.3 |
| 6 | planner / review | 760 | 0 | 112 | 774 | 1,646 | 57.7 |
| 合计 | 6 次调用，均成功且用量完整 | 4,774 | 0 | 1,359 | 11,658 | 17,791 | — |

四桶互斥，推理用量不会再次计入输出；TOKENS 是资源计量，现金费用未知。17,791 tokens 仅属于这次成功开发运行，不能代表包括前序失败在内的总成本。[逐次调用账本](task-ledger.jsonl)

## 全部前序失败

| 运行 | 可核对的故障类别 | 终态 | token 总量 | 封存已评分 |
| --- | --- | --- | --- | --- |
| qwen-etf-dev-20261117-01 | http_401 | failed | 未知 | 否 |
| qwen-etf-dev-20261117-02 | http_401 | failed | 未知 | 否 |
| qwen-etf-dev-20261117-03 | request_timeout_180_seconds | failed | 未知 | 否 |
| qwen-etf-dev-20261117-04 | http_401 | failed | 未知 | 否 |

四次失败均未获得可信用量，因此总 token 未知，不能按零计算，也不能只报告这次成功运行而省略这些尝试。`-04` 是代理进程仍使用旧环境凭据造成的 401；`-05` 从用户已更新凭据的同一终端启动。[失败摘要与本地账本哈希](prior-attempts.json) 保留每次故障类别、耗时、用量状态及失败报告。

## 证据与复核

[evidence-index.json](evidence-index.json) 映射原始运行路径、发布副本路径和 SHA-256；开发报告、选择、账本、树、节点代码与分数、配置、提示词和协议快照均可按哈希核对。报告与源码副本是原字节复制，原始路径中的 `runs/` 指执行时的本地目录。

封存评分前完成准入检查：开发成功结束、用量完整、最高 accepted 节点匹配、输入及冻结哈希一致、没有既有封存尝试。评分后独立核对三线逐日复合收益、5 bps 成本、持仓换手和指标汇总，算术一致。数据 provenance manifest 用于核对来源与价格哈希；原始及标准化行情、原始模型响应只保留在本地，不纳入发布包。此发布包可审阅证据；重跑模型可能产生不同候选，不能重开已经暴露的区间并仍称它未见。

## 解释边界

这是一遍探索性开发运行，加一次 61 日历史封存观察。该区间包含 2020 年初市场剧烈波动，不能推断长期稳定、实盘成交或容量；不突出短区间年化数字。固定存续 ETF 池、事后复权、收盘价成交与恒定交易成本也限制外推。数据来源与权利边界见 [来源说明](../etf-data-sources.md)。

本次观察支持的结论是：审计链完成了真实模型调用与单次封存评分，同时保留了模型研究没有改进原始信号的负结果。
