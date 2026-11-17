# Quant Research MVP: reproducible factor experiments｜可复现因子实验

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

This is a research demonstration, not trading advice or evidence of live returns. The first milestone evaluates **one fixed universe at daily frequency**. A candidate is a signal function that ranks the assets; the system either rejects it or retains it with a complete experiment record. Positive returns are not a completion criterion. A key-free scripted run and a separate one-shot finalizer are implemented. A real Qwen ETF experiment has also completed: none of the three tested candidates beat the original signal, so the Agent retained the original five-session momentum. See the [audited results and negative results](qwen-etf-case/report.md).

## Data and timing contract

- Input is a user-supplied CSV with `date,symbol,close` columns, plus a manifest recording its source, redistribution permission, price-adjustment basis and SHA-256. The repository may contain deterministic synthetic fixtures, but no unlicensed market-data copy. The first version rejects duplicate, missing, non-positive or non-finite prices and unequal per-symbol date grids. A fixed universe is a **demonstration limitation**, not a survivorship-free market backtest.
- Every candidate sees price history only through signal date `t`. A signal calculated after the `t` close can first change the portfolio at the `t+1` close; its first measured return is from the `t+1` close to the `t+2` close. There is no same-close execution. Fees and slippage are charged on absolute weight changes.
- Dates split chronologically into train, development and sealed final evaluation. The trusted parent validates the full CSV at load time, but neither the model nor the candidate worker receives sealed future prices during development; the final period is **scored** only after the candidate and selection rule are frozen. If a future-return label spans `H` sessions, adjacent selection windows require at least `H` sessions of separation. Earlier prices may warm up features; later labels and fitted statistics may not flow backwards.

The current label horizon is **two sessions** (`t` to `t+2`); the synthetic workflow leaves four sessions between its train/dev and dev/holdout result windows. The current signal functions are rule-based and do not fit parameters on the train window; that window is reserved for historical context and future train-only fitting.

A dataset manifest has this shape (replace the source, permission statement and digest with your actual data):

```json
{
  "schema": "quant_dataset/1",
  "csv": "prices.csv",
  "csv_sha256": "64 lowercase hexadecimal characters",
  "source": "origin and acquisition date",
  "license": "permission or restriction statement",
  "price_basis": "close adjustment and corporate-action policy",
  "universe_policy": "fixed_demo"
}
```

The CSV header is exactly `date,symbol,close`, sorted by date, with a complete price for every symbol on every date. `fixed_demo` is a deliberately narrow policy name: it does **not** establish point-in-time membership or survivorship-free results.

## Explicit data choice and preflight

- User-supplied data requires all four trading-date boundaries: `--train-end`, `--dev-start`, `--dev-end` and `--holdout-start`. This applies to both scripted and Qwen runs. The synthetic demonstration's position-based split is never silently applied to a supplied market dataset.
- Qwen runs on supplied data also require `--dataset-manifest` and a matching `--protocol`. The protocol fixes the dataset hash, model, date boundaries, fees and budgets. A model call against generated synthetic prices is available only with the explicit `--allow-synthetic-paid-demo` flag; it remains a synthetic engineering demonstration.
- Add `--preflight-only` to inspect the data label, chronological windows, sample counts and token budget. Preflight validates local inputs without creating a run, contacting the gateway, requiring an API key or requiring a clean Git checkout. Actual research prints the same summary before model calls; real Qwen calls additionally require an API key in the process environment and a clean committed checkout.

Copyable commands for a supplied dataset are in the [repository README](../README.md) and [Qwen experiment guide](qwen-etf-comparison.md). Replace the example paths and dates with the supplied data's path and trading dates, and register a matching protocol. Pricing is in test-only resource tokens, not cash.

## Implemented workflow

1. A deterministic loader, reference signals and cost-aware backtester, plus future-data, timing and malformed-input tests.
2. `quant_research.run` prepares an isolated `quant_run/1`, invokes the existing experiment tree, ledger and library through `QuantHarness`, and freezes the highest accepted development node only after the run passes admission checks. `--mode scripted` is a zero-model engineering test; `--mode qwen` is a real-model dev entry point. No model is called in the scripted example.
3. Inputs are sealed before research: the Agent config, prices, all four role prompts, hard-cap rules, triggers, original task signal and task profile. The run verifies those hashes again before writing a selection. Unknown model usage, a runner error or changed inputs produces a failed terminal record and failure report, retains partial evidence, and prevents selection and sealed evaluation.
4. `quant_research.finalize_cli --evaluate-sealed-once` checks candidate, score, dataset, manifest, input and ledger hashes and rejects ineligible dev runs before consuming an attempt. A separate scorer then opens the sealed interval. It creates an exclusive attempt marker before scoring and refuses any repeat, including after a scoring failure. It is never called from `evaluation.orchestrate.execute_task`, which scores holdout after each old-style task.
5. The run directory retains every executed node, including rejected and failed candidates, and reports four disjoint token buckets. Unknown usage is flagged, with the total set to null in failure evidence; it cannot qualify for sealed evaluation. The test-only token price table cannot support a cash-cost claim.

The report shows every executed candidate, including failures; the dataset, code, configuration and ledger hashes; net return, risk, turnover, wall time and model usage. Unknown model usage remains unknown, not zero.

The complete input-sealing protocol applies to new runs. Older records stay unchanged; a run without the required input hashes cannot be finalized by the stricter evaluator. Do not reconstruct or overwrite old evidence to make it pass.

## Current evidence and remaining work

The fixed-rule historical ETF case and the real Qwen experiment are published separately. In the Qwen experiment, the selected original signal returned **+6.07%** over the 61-day sealed interval, versus **−12.47%** for daily equal weight; improvement over the original signal was **0**. The successful dev run recorded **17,791 tokens** across six calls. Four earlier failed attempts have unknown usage, so that figure is not the cost of all attempts.

A point-in-time universe, dividend/split verification, multiple independent periods and stronger comparisons of research methods remain future work. The synthetic holdout validates the one-shot mechanism, not predictive quality; the short historical Qwen observation does not establish stable or live performance.

---

<a id="中文"></a>

## 中文

这是研究演示，不是交易建议或实盘收益证据。首个里程碑评估**固定股票池、日频数据**。候选是对资产排序的信号函数；系统拒绝它，或连同完整实验记录保留它。正收益不是完成标准。无密钥的脚本运行与独立单次 finalizer 已实现，真实 Qwen ETF 实验也已完成：三个已测试候选均未超过原始信号，Agent 最终保留原始 5 日动量。详见[审计结果与负结果](qwen-etf-case/report.md)。

## 数据与时序约定

- 输入是用户提供的 CSV，列为 `date,symbol,close`，并附带记录来源、再分发许可、复权口径与 SHA-256 的 manifest。仓库可包含确定性合成测试数据，但不分发未经许可的行情副本。首版拒绝重复、缺失、非正、非有限价格，以及各标的日期网格不一致的数据。固定股票池是**演示限制**，不代表无幸存者偏差的市场回测。
- 每个候选只看到截至信号日 `t` 的价格历史。在 `t` 收盘后计算的信号，最早于 `t+1` 收盘调整持仓；首次计量收益来自 `t+1` 至 `t+2` 收盘。不允许同收盘成交，手续费与滑点按权重绝对变化计入。
- 日期按时间顺序切为训练、开发和封存最终评估。可信父进程加载时校验完整 CSV，但开发期间模型和候选工作进程都不会收到封存未来价格；只有候选与选择规则冻结后才对最终区间**评分**。如果未来收益标签跨越 `H` 个交易日，相邻选择窗口至少留出 `H` 日间隔。早期价格可用于特征预热，后续标签或拟合统计量不得反向泄露。

当前标签跨度为**两个交易日**（`t` 至 `t+2`）；合成流程在训练／开发和开发／封存收益窗口之间均留出四个交易日。当前信号函数基于规则，不在训练窗口拟合参数；该窗口用于历史上下文及未来仅限训练期的拟合。

数据 manifest 格式如下，请替换为实际数据的来源、许可声明与哈希：

```json
{
  "schema": "quant_dataset/1",
  "csv": "prices.csv",
  "csv_sha256": "64 lowercase hexadecimal characters",
  "source": "origin and acquisition date",
  "license": "permission or restriction statement",
  "price_basis": "close adjustment and corporate-action policy",
  "universe_policy": "fixed_demo"
}
```

CSV 表头必须恰为 `date,symbol,close`，按日期排序，每天每个标的都有完整价格。`fixed_demo` 是刻意限定的策略名称，**不**证明历史时点成分股完整，也不证明消除了幸存者偏差。

## 显式选择数据与预检

- 用户提供的数据必须指定全部四个交易日边界：`--train-end`、`--dev-start`、`--dev-end` 和 `--holdout-start`，脚本与 Qwen 模式均如此。不会把合成演示按位置切分的规则静默套到真实数据上。
- Qwen 使用自有数据时，还必须指定 `--dataset-manifest` 和匹配的 `--protocol`。协议固定数据哈希、模型、日期、费用和预算。仅在显式使用 `--allow-synthetic-paid-demo` 时，才能对生成的合成价格调用模型；这仍是合成工程演示。
- 添加 `--preflight-only` 可查看数据标签、时间窗口、样本数与 token 预算。预检只校验本地输入，不创建运行、不访问网关，也不需要 API key 或干净工作树。实际研究会在模型调用前打印同样的摘要；真实 Qwen 调用还要求进程环境中的 API key 和已提交且干净的工作树。

自有数据的命令示例见[仓库 README](../README.md)和 [Qwen 实验手册](qwen-etf-comparison.md)。示例路径与日期须替换为实际数据路径及交易日，并注册匹配协议。计价使用仅供测试的资源 token，不是现金。

## 已实现流程

1. 确定性数据加载器、参考信号、含交易成本的回测器，以及未来数据、时序和异常输入测试。
2. `quant_research.run` 创建隔离的 `quant_run/1`，通过 `QuantHarness` 调用现有实验树、账本和能力库；只有通过准入检查后，才冻结最高分的 accepted 开发节点。`--mode scripted` 是零模型工程测试，`--mode qwen` 是真实模型开发入口。脚本示例不调用模型。
3. 研究前封存 Agent 配置、价目、四个角色提示词、硬上限、触发词、原始任务信号与任务概要；写入选择记录前再次校验哈希。用量未知、运行错误或输入变更会产生失败终态与失败报告，保留部分证据，并阻止生成选择记录和封存评分。
4. `quant_research.finalize_cli --evaluate-sealed-once` 核对候选、分数、数据集、manifest、输入与账本哈希，在消耗机会前拒绝不合格的开发运行。独立评分器随后打开封存区间，评分前创建排他尝试标记，成功或失败都拒绝重复。该入口不会从旧式逐任务后评估 holdout 的 `evaluation.orchestrate.execute_task` 调用。
5. 运行目录保留每个已执行节点，包括被拒绝和失败的候选，并报告四个互斥 token 桶。未知用量会标记为未知，失败证据中的总量为 null，不能通过封存准入。测试 token 价目不能支持现金费用结论。

报告展示全部已执行候选及失败、数据／代码／配置／账本哈希、净收益、风险、换手、耗时和模型用量。未知用量保持未知，不按零处理。

完整输入封存协议适用于新运行。旧记录保持原样；缺少必要输入哈希的运行不能通过更严格的 finalizer。不要为使旧证据通过而补造或覆盖记录。

## 当前证据与后续工作

固定规则 ETF 历史案例与真实 Qwen 实验分别发布。Qwen 实验最终保留的原始信号在 61 日封存期净收益为 **+6.07%**，每日等权为 **−12.47%**；相对原始信号的改进为 **0**。成功开发运行的六次调用记录 **17,791 tokens**。此前四次失败的用量未知，因此该数字不是全部尝试的总成本。

历史时点股票池、股息／拆股核验、多个独立区间及更充分的研究方法比较仍待完成。合成封存评分验证的是单次机制，不是预测质量；短期历史 Qwen 观察也不能证明长期稳定或实盘表现。
