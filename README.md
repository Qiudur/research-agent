# Quant Research Agent｜量化因子研究与样本外验证

一个可审计的研究工作流：给定有来源声明的日收盘价，Agent 提出有限的截面排序信号，在开发期逐一证伪，冻结入选代码后仅对封存期评分一次。目标是检验研究流程是否可信，**不是自动选股、实盘交易或收益承诺**。本项目从 [QZTnb/research-agent](https://github.com/QZTnb/research-agent) 分叉，保留原有实验树、预算、账本与能力库；原系统说明归档在 [legacy-project-readme.md](docs/legacy-project-readme.md)。

## 已实现的闭环

1. 数据清单声明 `source`、`license`、`price_basis` 和 CSV SHA-256。首版只接收完整的固定股票池，拒绝缺失、重复、非正和非有限价格。
2. 训练／开发／封存时间切分留出标签跨度。信号在 `t` 收盘后只见历史，`t+1` 收盘交易，`t+2` 实现收益；回测计手续费、滑点、换手与最大回撤，并与等权比较。
3. 候选 `compute_signal(history)` 在 macOS `sandbox-exec` 子进程执行。子进程逐日只收到截至信号日的价格前缀，没有 API key、数据文件路径或网络权限；无沙箱时拒绝评分。
4. Agent 的每次尝试写入实验树、原始代码、开发期评分和账本；失败与负结果保留。数据、全部角色提示词、预算规则、触发词和任务可见输入均记录哈希，运行结束时复核，再冻结最优开发期节点和账本。
5. 用量缺失、运行错误或输入变更会记录失败终态并禁止生成选择记录。独立 finalizer 在评分前复核开发运行准入条件和冻结输入，再创建排他尝试标记；成功或失败都不能重试同一封存期。

## 已跑出的真实案例

在 8 只 ETF 的公开历史快照上，7 个固定候选经过三个开发窗口选出了 126 日动量。冻结后的 2018—2019 年结果如下，已扣除手续费与滑点：

| 策略 | 区间净收益 | 最大回撤 |
| --- | ---: | ---: |
| 每日等权 | +14.06% | 12.90% |
| 预先固定的 63 日动量 | +16.25% | 10.73% |
| 开发期选出的 126 日动量 | −0.49% | 14.92% |

**开发期选优在封存期落后等权 14.56 个百分点。** 我们保留这次失败与全部候选证据，不根据封存结果重新选赢家。这是固定规则的历史研究，模型调用为零，Qwen 对照尚未运行。

![封存期三条净值曲线](docs/etf-case-study/holdout-net-equity.svg)

[案例结论与复现](docs/etf-case-study/README.md) · [完整报告](docs/etf-case-study/report.md) · [逐日 JSON 证据](docs/etf-case-study/study.json)

Qwen 对照实验已单独注册，但尚未调用模型；新的 2020-01-06—2020-04-01 封存区间与运行步骤见 [Qwen ETF 对照协议和手册](docs/qwen-etf-comparison.md)。

## 无密钥复现

### 真实 ETF 基线案例

研究问题是：**简单的 ETF 截面排序，在不同年份扣除交易成本后，能否稳定超过每日等权？** 固定 8 只 ETF，下载 2008—2019 年的公开历史快照；先在 2012—2017 年的三个开发窗口比较固定网格，再冻结选择并评估 2018—2019 年封存窗口。所有候选和负结果均保留，规则见运行绩效计算前提交的 [研究协议](config/etf_case_study.json)。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m quant_research.etf_case
```

该命令不需要 API key，只为获取固定版本数据访问公开网络，不调用模型。输出 `runs/etf-case-*/report/index.html` 和 `report.md`；JSON 证据保留逐日持仓、费用后的回报、净值、换手和输入哈希。已有下载可用 `--dataset-manifest runs/etf-case-data-v1/manifest.json` 复用，输出目录必须是新目录。

比较线是每日等权、预先固定的 63 日动量，以及 7 个固定候选中按“最差开发窗口净超额、平均净超额、候选 ID”依次排序的入选方案。每个窗口从现金起步，计入初次建仓成本，单边手续费 2 bps、滑点 3 bps。**这是零模型调用的固定规则研究，不是 Qwen 或 Agent 效果。** 后续模型对照需要另行冻结协议与尚未打开的最终区间。

数据是 Kaggle 上传者提供的 Yahoo/yfinance 调整价快照，存在固定存续 ETF 池、事后复权和成交假设限制。原行情仅保存在本地被忽略的 `runs/`；上游权利未独立核验，仓库公开下载代码、来源哈希与派生研究结果。见 [数据来源与边界](docs/etf-data-sources.md)。

### 合成数据的 Agent 闭环演示

在 macOS、Python 3.11 环境从仓库根目录运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m quant_research.run --mode scripted
```

命令会打印 `runs/<run_id>/dev_report.json`。查看同目录的 `selection.json`、`tasks/quant-demo/tree.jsonl` 和 `ledger.jsonl`，确认包括未入选方案及其分数。此演示使用**确定性合成价格和脚本角色，模型调用为零**；开发期报告不会打开封存期。若要演示单次评分，检查选择记录后明确执行：

```bash
.venv/bin/python -m quant_research.finalize_cli \
  --run-dir runs/<run_id> --evaluate-sealed-once
```

`final/report.json` 会注明 `SYNTHETIC DATA`、`selected_origin`、风险与交易成本。不要把合成数据上的百分比或 Sharpe 解读为投资业绩。

## Qwen 开发实验

`--mode qwen` 固定使用 `qwen3.5-27b`，默认网关是 `https://ai-gateway.theunifiedcare.com/v1`。正常研究须显式提供数据清单和全部四个切分日期；自有数据在脚本模式下也需要四个日期，不会自动套用演示数据的位置。先预检数据类型、时间区间、各区间样本数和 token 预算，再启动模型调用。

下面的清单路径和日期是命令格式示例，请按你的数据文件和实际交易日修改。预检只读取和校验本地输入，**不创建运行目录、不联网，也不需要 API key 或干净工作树**。

```bash
.venv/bin/python -m pip install -e '.[llm]'
quant_data_args=(
  --dataset-manifest data/my-market/manifest.json
  --train-end 2018-12-28
  --dev-start 2019-01-04
  --dev-end 2022-12-23
  --holdout-start 2023-01-03
)
.venv/bin/python -m quant_research.run --mode qwen \
  "${quant_data_args[@]}" --preflight-only
```

实际调用前，在当前终端通过环境变量提供 `OPENAI_API_KEY`，不要把密钥写进命令行、配置或 Git。真实调用要求工作树已提交且干净；每题默认预算为 40,000 token，启动时会再次显示运行摘要。账本 `TOKENS` 是资源计量，**不是现金价格**。

```bash
.venv/bin/python -m quant_research.run --mode qwen \
  "${quant_data_args[@]}" --budget-tokens 40000
```

仅为检查真实网关与合成演示的接线时，可以明确允许模型消耗：

```bash
.venv/bin/python -m quant_research.run --mode qwen \
  --allow-synthetic-paid-demo --preflight-only
.venv/bin/python -m quant_research.run --mode qwen \
  --allow-synthetic-paid-demo
```

这两种数据路径都会保留来源标签；合成演示调用不构成真实市场实证。自有日收盘价须有带许可和 SHA-256 的 `quant_dataset/1` 清单。首版不处理退市、动态成分股、复权核查、真实成交与市场冲击，所以真实市场数据结果也只能作为研究演示。完整数据格式、方法与限制见 [量化研究 MVP 文档](docs/quant-research-mvp.md)。

## 验证

```bash
.venv/bin/python -m unittest tests.test_quant_data tests.test_quant_splits \
  tests.test_quant_backtest tests.test_quant_signals tests.test_quant_harness \
  tests.test_quant_core_compat tests.test_quant_provenance \
  tests.test_quant_run tests.test_quant_finalize \
  tests.test_quant_etf_data tests.test_quant_etf_study \
  tests.test_quant_etf_report tests.test_quant_etf_case -v
```

脚本角色验证 Agent 工程闭环，ETF 案例验证固定规则在历史数据上的表现；当前没有可以据此报告的 Qwen 量化效果或现金节省。
