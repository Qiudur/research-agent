# Qwen ETF 对照实验

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
