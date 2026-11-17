# Qwen ETF comparison｜Qwen ETF 对照实验

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

This Qwen-only comparison is separate from the fixed-grid case. It fixes `qwen3.5-27b`, the repository gateway, eight ETFs, Top-2 allocation, 2 bps fees and 3 bps slippage. Qwen proposes and revises signals only on development data; an independent finalizer scores the sealed interval once after development ends.

The fixed-grid case already exposed 2018–2019, so that interval is quarantined here. This experiment's sealed interval was 2020-01-06 through 2020-04-01, covering 61 common trading days. It supports only a short exploratory out-of-sample observation, not a claim of stable or tradable performance.

## Results

`runs/qwen-etf-dev-20261117-05` completed development, and its sole sealed evaluation finished with status `scored`. All three candidates actually executed and scored by Qwen were rejected; the highest accepted node remained `n0000`, the original five-session momentum signal. The selected and original signals both returned **+6.07%** net with **6.72%** maximum drawdown. Daily equal weight returned **−12.47%** net with **20.96%** maximum drawdown. Excess return over equal weight was **18.54 percentage points**, but improvement over the original signal was **0**: this experiment provides no evidence that Qwen found a better signal.

The successful development run recorded six model calls, with four disjoint token buckets: input **4,774**, cached_input **0**, output **1,359** and reasoning **11,658**, totaling **17,791**. This excludes unknown usage from earlier failed calls; it is neither the cost of all attempts nor a cash expense.

[Results and negative results](qwen-etf-case/report.md) · [Evidence index](qwen-etf-case/evidence-index.json)

## Frozen inputs

- Current protocol: [config/qwen_etf_comparison_v3.json](../config/qwen_etf_comparison_v3.json), ID `qwen-etf-comparison-v3-shell-credential`; original [v1](../config/qwen_etf_comparison.json) and [v2](../config/qwen_etf_comparison_v2.json) remain available.
- Dataset manifest: `runs/etf-qwen-data-v1/manifest.json`, extending the fixed Kaggle v2 source snapshot through 2020-04-01. Price CSV files remain in the ignored local `runs/` directory.
- Snapshot provenance, adjusted-price semantics and rights limitations follow the [data-source notes](etf-data-sources.md).
- Run budget: 40,000 TOKENS, maximum output 6,000 tokens, wall-clock limit 3,600 seconds and per-request timeout 600 seconds. TOKENS measures resources, not cash prices.
- Fixed comparisons are daily equal weight and the original five-session momentum in the task package. The sealed report records the Qwen-selected signal, original momentum and equal weight.

## Earlier failures and protocol revisions

| Run | Protocol | Terminal failure | Token usage |
| --- | --- | --- | --- |
| `qwen-etf-dev-20261117-01` | v1 | HTTP 401 authentication failure | Unknown |
| `qwen-etf-dev-20261117-02` | v1 | HTTP 401 authentication failure | Unknown |
| `qwen-etf-dev-20261117-03` | v1 | Planning request reached the 180-second timeout | Unknown |
| `qwen-etf-dev-20261117-04` | v2 | Tool process used stale environment credentials; HTTP 401 | Unknown |

These failed calls returned no trustworthy usage and were recorded as `unknown_usage`, not zero consumption. None produced a selection record or opened the sealed period. Before seeing sealed results, v2 increased the request timeout from 180 to 600 seconds and the wall-clock limit from 900 to 3,600 seconds. V3 kept those parameters and started from the user's terminal with updated credentials. The successful `-05` and four failures are reported separately; they are not five valid repetitions from which a winner was selected.

## Preflight and execution

The following preserves this experiment's v3 parameters. The 2020 interval has now been scored; replaying these parameters cannot create an unseen out-of-sample experiment. A new study needs another preregistered interval that has not yet been opened.

Preflight does not access the network, read the API key, create a run directory or call a model:

```bash
PYTHON=.venv/bin/python
$PYTHON -m quant_research.run --project-root . --mode qwen \
  --protocol config/qwen_etf_comparison_v3.json \
  --dataset-manifest runs/etf-qwen-data-v1/manifest.json \
  --train-end 2014-12-29 --dev-start 2015-01-05 \
  --dev-end 2017-12-29 --holdout-start 2020-01-06 \
  --top-k 2 --fee-bps 2 --slippage-bps 3 \
  --budget-tokens 40000 --max-output-tokens 6000 \
  --wall-seconds 3600 --request-timeout-seconds 600 \
  --preflight-only
```

After preflight, set the API key in the current terminal using hidden input. Then run the same command without `--preflight-only` and add a fresh `--run-dir runs/qwen-etf-dev-<timestamp>`. A live run requires a clean committed working tree. Do not store the key in configuration or Git.

Successful development generates `dev_report.json` and `selection.json` before opening the sealed interval. Failure retains a failure report and produces no selection. Run the following only after development completes, usage is complete and the selection record passes verification:

```bash
$PYTHON -m quant_research.finalize_cli \
  --run-dir runs/qwen-etf-dev-<timestamp> --evaluate-sealed-once
```

`final/report.json` retains the selection, Qwen candidate, original five-session momentum, equal weight, costs, daily equity, model ledger and input hashes. Success or failure consumes the run's sole sealed attempt; a failed attempt must not be rerun in another directory under the same protocol to select a favorable result.

## Interpretation limits

This study contains one completed Qwen repetition and supports only an exploratory comparison. Development scores are not out-of-sample results, and a 61-day historical backtest is not live performance. A claim of stability would require another preregistered, unopened interval and an aggregation rule fixed before inspecting results.

---

<a id="中文"></a>

## 中文

这是一项独立于固定网格案例的 Qwen-only 对照。它固定使用 `qwen3.5-27b`、仓库网关、8 只 ETF、Top-2、2 bps 手续费和 3 bps 滑点。Qwen 只在开发区间提出和改进信号，开发结束后由独立 finalizer 对封存区间评分一次。

2018—2019 已被固定网格案例打开，因此在本实验中隔离为历史暴露区间。本实验使用的封存区间是 2020-01-06 至 2020-04-01，共 61 个共同交易日。它很短，只能支持一次探索性样本外观察，不能据此声称策略稳定或可交易。

## 本次结果

`runs/qwen-etf-dev-20261117-05` 已完成开发，唯一一次封存评分状态为 `scored`。Qwen 实际执行并评分的 3 个候选均被拒绝，最高的 accepted 节点仍是 `n0000` 原始 5 日动量。封存期入选方案与原始信号净收益均为 **+6.07%**、最大回撤 **6.72%**；每日等权净收益 **−12.47%**、最大回撤 **20.96%**。相对等权超额 **18.54 个百分点**，相对原始信号改进为 **0**：本次没有证据表明 Qwen 找到了更好的信号。

成功开发运行记录 6 次模型调用，四桶 token 为 input **4,774**、cached_input **0**、output **1,359**、reasoning **11,658**，合计 **17,791**。这不包括此前失败调用的未知用量，因此不是全部尝试的总成本，也不是现金支出。

[结果与负结果](qwen-etf-case/report.md) · [证据索引](qwen-etf-case/evidence-index.json)

## 已冻结的输入

- 本次协议：[config/qwen_etf_comparison_v3.json](../config/qwen_etf_comparison_v3.json)，ID 为 `qwen-etf-comparison-v3-shell-credential`；原始 [v1](../config/qwen_etf_comparison.json) 和 [v2](../config/qwen_etf_comparison_v2.json) 均保留。
- 数据清单：`runs/etf-qwen-data-v1/manifest.json`，由固定 Kaggle v2 原始快照扩展到 2020-04-01；价格 CSV 只留在被忽略的 `runs/`。
- 原始快照来源、调整价口径和权利边界沿用[数据来源说明](etf-data-sources.md)。
- 本次运行预算 40,000 TOKENS，最大输出 6,000 token，墙钟上限 3,600 秒、单次请求超时 600 秒；TOKENS 是资源计量，不是现金价格。
- 固定对照包括每日等权和任务包中的原始 5-session 动量；封存报告同时记录 Qwen 入选信号、原始动量和等权。

## 前序失败与协议修订

| 运行 | 协议 | 终态原因 | token 用量 |
| --- | --- | --- | --- |
| `qwen-etf-dev-20261117-01` | v1 | HTTP 401 鉴权失败 | 未知 |
| `qwen-etf-dev-20261117-02` | v1 | HTTP 401 鉴权失败 | 未知 |
| `qwen-etf-dev-20261117-03` | v1 | 规划请求达到 180 秒超时 | 未知 |
| `qwen-etf-dev-20261117-04` | v2 | 工具进程使用旧环境凭据，HTTP 401 | 未知 |

这些失败调用没有可信用量，账本标为 `unknown_usage`，不按零消耗处理，均未生成选择记录或打开封存期。v2 在看到封存结果前将请求超时从 180 秒提高到 600 秒、墙钟上限从 900 秒提高到 3,600 秒；v3 保留这些参数，从已经更新凭据的用户终端启动。成功的 `-05` 与四次失败分别报告，不把它们当作五次有效实验后挑选赢家。

## 预检与运行

以下保留本次 v3 的运行参数。当前 2020 区间已经评分，重放这些参数不能再称为未打开的样本外实验；新的研究须先注册另一个尚未打开的区间。

预检不会联网、读取 API key、创建运行目录或调用模型：

```bash
PYTHON=.venv/bin/python
$PYTHON -m quant_research.run --project-root . --mode qwen \
  --protocol config/qwen_etf_comparison_v3.json \
  --dataset-manifest runs/etf-qwen-data-v1/manifest.json \
  --train-end 2014-12-29 --dev-start 2015-01-05 \
  --dev-end 2017-12-29 --holdout-start 2020-01-06 \
  --top-k 2 --fee-bps 2 --slippage-bps 3 \
  --budget-tokens 40000 --max-output-tokens 6000 \
  --wall-seconds 3600 --request-timeout-seconds 600 \
  --preflight-only
```

通过预检后，在当前终端设置 API key（输入不会回显），然后执行同一命令，去掉 `--preflight-only`，并加一个新的 `--run-dir runs/qwen-etf-dev-<timestamp>`。真实运行要求工作树干净且已提交；不要把 key 写入配置或 Git。

开发成功后生成 `dev_report.json` 和 `selection.json`，此时尚未打开封存区间；失败则保留失败报告，不生成选择记录。只有开发完成、用量完整且选择记录复核通过后，才执行一次：

```bash
$PYTHON -m quant_research.finalize_cli \
  --run-dir runs/qwen-etf-dev-<timestamp> --evaluate-sealed-once
```

`final/report.json` 会保留 selection、Qwen 候选、原始 5-session 动量、等权、费用、逐日净值、模型账本和输入哈希。成功或失败都会消耗该运行唯一的封存尝试；失败不得换目录补跑同一协议后挑结果。

## 解释边界

这次只有一个 Qwen repetition，报告只能称为探索性对照。不能把开发分数写成样本外效果，也不能把 61 日历史回测写成实盘收益。若要声称“稳定”，下一步应在另一个预先注册、同样未打开的时间区间上重复，并固定聚合规则后再看结果。
