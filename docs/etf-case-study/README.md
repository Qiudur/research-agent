# Real ETF case study: development selection failed out of sample

[English](#english) · [中文](#中文)

<a id="english"></a>

This study tests whether fixed rules can consistently outperform daily equal weighting across different years. **It used no Qwen or other model calls and does not establish the Agent's quantitative research ability.**

## Results

The preregistered selection rule chose 126-session momentum from seven candidates. Its net excess returns in the three development windows were −0.58, −10.37 and +13.51 percentage points: a worst-window result of −10.37 percentage points and a mean of +0.86 percentage points. All seven candidates had negative excess returns in their worst development window.

After selection was frozen, the holdout covered 501 return observations from 2018-01-04 to 2019-12-31. The figures below are historical returns over the whole interval after fees and slippage.

| Strategy | Net period return | Annualized return | Maximum drawdown | Net excess over equal weight |
| --- | ---: | ---: | ---: | ---: |
| Daily equal weight (8 ETFs) | 14.06% | 6.84% | 12.90% | +0.00 pp |
| Fixed 63-session momentum (Top 2) | 16.25% | 7.87% | 10.73% | +2.19 pp |
| Development-selected 126-session momentum (Top 2) | -0.49% | -0.25% | 14.92% | -14.56 pp |

The selected rule lagged equal weight by **14.56 percentage points** in the holdout and did not sustain its development performance. The fixed 63-session momentum rule was a preregistered comparator; it did not replace the selected rule after the holdout was viewed. These results do not support a claim that development selection wins consistently. This holdout has been opened; future model comparisons need a new protocol and an unopened final interval.

![Holdout net equity](holdout-net-equity.svg)

![Holdout drawdown](holdout-drawdown.svg)

## Inspect the evidence

- [Full research report, readable on GitHub](report.md): all development windows, negative results, three comparison lines and source declarations.
- [Static HTML](index.html): download and open locally; charts are inline, with no CDN or model calls.
- [Original study JSON](study.json): 30 window series containing 15,039 daily records of positions, returns, turnover and equity.
- [Development selection frozen before holdout](dev_selection.json) and [holdout attempt record](holdout_attempt.json): published byte for byte so the SHA chain can be checked.
- [Frozen protocol](protocol.json), [dataset manifest](dataset/manifest.json), [reproduction and validation record](reproduction.json), and [file checksums](SHA256SUMS).

Published JSON files are exact copies of the run artifacts. Evidence paths use stable logical labels; actual file paths were used only for local verification during execution. The raw price CSV is not distributed with the repository, so this dataset manifest is an evidence manifest. Loading it directly will fail without the price file; download the data as described below first.

## Reproduce from source

From the repository root, using Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m quant_research.etf_case
```

The command downloads the fixed Kaggle v2 historical snapshot and writes a new `runs/etf-case-*/report/` directory. Data downloads require network access but no key. Reuse existing data with `--dataset-manifest runs/etf-case-data-v1/manifest.json`. Each run must use a fresh output directory.

A fresh download changes the acquisition timestamp, so its manifest SHA may differ. Check the source-data and normalized-price hashes against this case. New selection and study files also have different SHAs because their timestamps differ; compare the numerical evidence separately. Verify the published files from this directory with `shasum -a 256 -c SHA256SUMS`.

## Method and limitations

The fixed universe is EEM, EFA, GLD, IEF, IWM, QQQ, SPY and TLT. The years 2008–2011 provide historical feature context; 2012–2017 is divided into three development windows. Selection ranks candidates by worst-window net excess return, then mean net excess return, then candidate ID. The final evaluation scores only the three preregistered holdout lines. Signals are generated after the t close, traded at the t+1 close and realize returns at t+2. Each window starts from cash, with fees of 2 bps and slippage of 3 bps charged on traded value; the equal-weight benchmark also rebalances daily.

The data is a 2020 Yahoo/yfinance adjusted-price snapshot supplied by a Kaggle uploader. A fixed universe of surviving ETFs has survivorship bias, and retrospective adjustments do not provide a point-in-time guarantee. Assumed closing fills and fixed costs do not establish achievable fills, capacity or market impact. The uploader declares CC0, but upstream redistribution rights have not been independently verified, so raw prices remain local. See [data sources](../etf-data-sources.md).

The case records 112 passing quantitative tests. An independent arithmetic check recomputed all returns, drift-adjusted turnover and metrics from the CSV and recorded positions, without rerunning signals, and found agreement. This verifies consistency between the implementation and evidence; it does not prove strategy effectiveness.

---

<a id="中文"></a>

# 真实 ETF 研究案例：开发期选优没有通过样本外验证

这次研究检验固定规则能否在不同年份稳定超过每日等权。**它没有调用 Qwen 或其他模型，也不代表 Agent 的量化能力。**

## 结果

按预先提交的规则，7 个候选中入选的是 126 日动量：三个开发窗口的净超额分别为 −0.58、−10.37、+13.51 个百分点，最差窗口 −10.37 个百分点，平均 +0.86 个百分点。全部 7 个候选的最差开发窗口超额都为负。

冻结后，2018-01-04 至 2019-12-31 共 501 个收益观测日的结果如下。以下是整个区间扣除手续费和滑点后的历史回测收益，不是实盘结果。

| 策略 | 区间净收益 | 年化收益 | 最大回撤 | 相对等权净超额 |
| --- | ---: | ---: | ---: | ---: |
| 每日等权（8 ETF） | 14.06% | 6.84% | 12.90% | +0.00 pp |
| 固定 63 日动量（Top 2） | 16.25% | 7.87% | 10.73% | +2.19 pp |
| 开发期选出的 126 日动量（Top 2） | -0.49% | -0.25% | 14.92% | -14.56 pp |

入选方案在封存期落后等权 **14.56 个百分点**，未能延续开发期表现。固定 63 日动量是协议预先列出的对照，不是看完封存结果后替换的赢家。这组结果不支持“开发期选优能稳定胜出”的结论；本次封存期已经打开，未来模型对照须使用新的协议和未打开的最终区间。

![封存期净值](holdout-net-equity.svg)

![封存期回撤](holdout-drawdown.svg)

## 查看证据

- [完整研究报告（GitHub 可直接阅读）](report.md)：所有开发窗口、全部负结果、三线对照及来源声明。
- [静态 HTML](index.html)：下载后本地打开；图表内联，无 CDN、无模型调用。
- [原始研究结果 JSON](study.json)：30 条窗口序列，保留 15,039 条逐日持仓、回报、换手与净值记录。
- [封存前的开发选择](dev_selection.json)与[封存尝试记录](holdout_attempt.json)：按原字节公开，SHA 链可核验。
- [冻结协议](protocol.json)、[数据清单](dataset/manifest.json)、[复现与验证记录](reproduction.json)、[文件校验和](SHA256SUMS)。

公开 JSON 是运行产物的原样副本。证据路径使用稳定逻辑标签，实际文件路径只在运行时用于本地核验。原价格 CSV 不随仓库分发，因此此处的数据清单是证据清单；直接加载它会缺少价格文件，请先按下文下载。

## 从源码复现

在仓库根目录、Python 3.11 环境执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m quant_research.etf_case
```

命令会下载固定的 Kaggle v2 历史快照并输出新的 `runs/etf-case-*/report/`。需要网络下载数据，无需密钥；已有数据可用 `--dataset-manifest runs/etf-case-data-v1/manifest.json` 复用。每次必须使用新输出目录。

新下载会改变清单中的获取时间，因此 manifest SHA 可能不同；源数据和规范化价格的哈希应与本案例核对。新运行的选择文件与 study 也因时间戳不同而有不同 SHA，数值证据应分别比较。公开文件可在本目录运行 `shasum -a 256 -c SHA256SUMS` 核验。

## 方法与限制

固定股票池为 EEM、EFA、GLD、IEF、IWM、QQQ、SPY、TLT；2008—2011 年仅提供历史特征上下文，2012—2017 年分三个开发窗口。按最差窗口净超额、平均净超额、候选 ID 顺序选优；最终只评预先规定的三条封存线。信号在 t 收盘后生成、t+1 收盘交易、t+2 实现收益；每个窗口从现金起步，手续费 2 bps、滑点 3 bps 按交易金额计入，等权基准也每日再平衡。

数据是 2020 年的 Yahoo/yfinance 调整价快照，来自 Kaggle 上传者。固定的存续 ETF 池有幸存者偏差，事后复权不能提供历史时点数据保证；固定收盘成交与成本无法证明真实成交、容量或市场冲击。上传者声明 CC0，但上游再发布权利未独立核验，所以原行情只留在本地。详见 [数据来源](../etf-data-sources.md)。

112 项量化测试通过；独立算术核对按 CSV 与记录持仓复算全部回报、漂移换手和指标，未重跑信号，结果一致。它验证实现和记录的一致性，不构成策略有效性证明。
