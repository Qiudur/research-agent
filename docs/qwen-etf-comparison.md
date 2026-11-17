# Qwen ETF 对照实验

这是一项独立于固定网格案例的 Qwen-only 对照。它固定使用 `qwen3.5-27b`、仓库网关、8 只 ETF、Top-2、2 bps 手续费和 3 bps 滑点。Qwen 只在开发区间提出和改进信号，开发结束后由独立 finalizer 对封存区间评分一次。

2018—2019 已被固定网格案例打开，因此在本实验中隔离为历史暴露区间。新的封存区间是 2020-01-06 至 2020-04-01，共 61 个共同交易日。它很短，只能支持一次探索性样本外观察，不能据此声称策略稳定或可交易。

## 已冻结的输入

- 协议：[config/qwen_etf_comparison.json](../config/qwen_etf_comparison.json)
- 数据清单：`runs/etf-qwen-data-v1/manifest.json`，由固定 Kaggle v2 原始快照扩展到 2020-04-01；价格 CSV 只留在被忽略的 `runs/`。
- 原始快照来源、调整价口径和权利边界沿用[数据来源说明](etf-data-sources.md)。
- 每次运行预算 40,000 TOKENS，最大输出 6,000 token，墙钟上限 900 秒；TOKENS 是资源计量，不是现金价格。
- 固定对照包括每日等权和任务包中的原始 5-session 动量；封存报告同时记录 Qwen 入选信号、原始动量和等权。

## 预检与运行

预检不会联网、读取 API key、创建运行目录或调用模型：

```bash
PYTHON=.venv/bin/python
$PYTHON -m quant_research.run --project-root . --mode qwen \
  --dataset-manifest runs/etf-qwen-data-v1/manifest.json \
  --train-end 2014-12-29 --dev-start 2015-01-05 \
  --dev-end 2017-12-29 --holdout-start 2020-01-06 \
  --top-k 2 --fee-bps 2 --slippage-bps 3 \
  --budget-tokens 40000 --max-output-tokens 6000 \
  --preflight-only
```

通过预检后，在当前终端设置 API key（输入不会回显），然后执行同一命令，去掉 `--preflight-only`，并加一个新的 `--run-dir runs/qwen-etf-dev-<timestamp>`。真实运行要求工作树干净且已提交；不要把 key 写入配置或 Git。

开发运行只生成 `dev_report.json` 和 `selection.json`，还没有打开封存区间。确认开发完成后，只允许执行一次：

```bash
$PYTHON -m quant_research.finalize_cli \
  --run-dir runs/qwen-etf-dev-<timestamp> --evaluate-sealed-once
```

`final/report.json` 会保留 selection、Qwen 候选、原始 5-session 动量、等权、费用、逐日净值、模型账本和输入哈希。成功或失败都会消耗该运行唯一的封存尝试；失败不得换目录补跑同一协议后挑结果。

## 解释边界

这次只有一个 Qwen repetition，报告只能称为探索性对照。不能把开发分数写成样本外效果，也不能把 61 日历史回测写成实盘收益。若要声称“稳定”，下一步应在另一个预先注册、同样未打开的时间区间上重复，并固定聚合规则后再看结果。
