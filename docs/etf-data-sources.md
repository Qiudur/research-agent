# Historical ETF data sources｜固定 ETF 历史数据来源

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

The assets and dates for the fixed-rule historical case were fixed before running strategies: SPY, QQQ, IWM, EFA, EEM, TLT, IEF and GLD, from 2008-01-01 through 2019-12-31. These surviving ETFs form a demonstration universe, not a complete point-in-time investable universe; the case does not eliminate survivorship bias.

## Public source and snapshot

Data comes from [Kaggle Stock Market Dataset v2](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/versions/2), uploaded by jacksoncrow. Public metadata lists the last update as 2020-04-02 and daily observations through 2020-04-01. The source description says historical prices were obtained from Yahoo Finance through yfinance: `Close` is split-adjusted, while `Adj Close` is adjusted for dividends and splits.

The project uses `Adj Close`, retaining the precision of every original row and emitting the engine's `date,symbol,close` format. These are adjusted prices in a 2020 snapshot, not corporate-action data reconstructed as of each historical date. They do not establish full point-in-time data guarantees. They support a research-workflow demonstration and evaluation of fixed candidates, not evidence of executable trading performance.

Downloading requires no API key, login or payment. It uses the public metadata API and the fixed-version single-file GET endpoint, for example:

```text
https://www.kaggle.com/api/v1/datasets/view/jacksoncrow/stock-market-dataset
https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset/etfs%2FSPY.csv?datasetVersionNumber=2
```

HEAD and GET may behave differently on this service; the downloader uses GET. Each response is limited to 5 MB, with a 30-second timeout per request and up to three attempts for transient errors only. Changes to the remote interface cause an error, never a fallback to synthetic data. The downloader checks that public metadata still identifies v2; if the uploader changes the current version, it requires a source review instead of silently accepting the change.

## Local reproduction

Run from the repository root:

```bash
python -m quant_research.etf_data --output-dir runs/etf-case-data-v1
```

The output directory must not already exist, so earlier research evidence cannot be overwritten. On success, the command returns `runs/etf-case-data-v1/manifest.json`, which can be passed directly to `quant_research.data.load_dataset()`. It records the fixed assets, requested and actual date windows, row counts, snapshot version, SHA-256 of metadata and every source CSV, and SHA-256 of the normalized CSV. Raw responses remain in `raw/` for local checks.

Only dates observed for all eight assets within the window are retained. There is no forward fill, backward fill or interpolation. The manifest records dates missing for each asset, dates removed by the intersection and the resulting dropped observations per asset. This check uses dates present in the source, without an independent exchange-calendar check: a date missing from every source file cannot be detected by intersection. Duplicate dates, incorrect columns, non-positive or non-finite adjusted prices fail immediately. A failed directory may retain downloaded source material, but no loadable success manifest is written; choose a new directory for a retry.

## Rights and redistribution

The Kaggle uploader declares **CC0: Public Domain**. This is the uploader's declaration only. The project has not independently verified upstream Yahoo Finance rights or redistribution permission, so the declaration does not establish that permission has been verified. Raw and normalized market CSV files remain in the Git-ignored local `runs/` directory and are not redistributed in the repository. The repository publishes download code, source links, data hashes and derived strategy-research results, so users can reproduce the case after checking the applicable data terms themselves.

---

<a id="中文"></a>

## 中文

真实案例的标的与时间窗口在运行策略前固定：SPY、QQQ、IWM、EFA、EEM、TLT、IEF、GLD，2008-01-01 至 2019-12-31。这是一组用于演示的存续 ETF，不代表历史时点可投资全集，也不能声称消除了幸存者偏差。

## 公开来源与快照

数据来自 [Kaggle Stock Market Dataset v2](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset/versions/2)，上传者为 jacksoncrow。公开元数据声明最后更新时间为 2020-04-02，覆盖截至 2020-04-01 的日线。来源说明写明：历史价格通过 yfinance 从 Yahoo Finance 获取；`Close` 为拆股调整价，`Adj Close` 为股息与拆股调整价。

本项目选择 `Adj Close`，保留原文件每一行的数值精度，输出为引擎使用的 `date,symbol,close`。这是 2020 年快照中的调整价，不是按每个历史时点还原的公司行动数据，不能据此声称具有完整的 point-in-time 数据保证。它适合展示研究流程和固定候选验证，不适合证明交易可实现性。

下载不需要 API key、登录或付费，调用公开元数据 API 与固定版本的单文件 GET 接口，例如：

```text
https://www.kaggle.com/api/v1/datasets/view/jacksoncrow/stock-market-dataset
https://www.kaggle.com/api/v1/datasets/download/jacksoncrow/stock-market-dataset/etfs%2FSPY.csv?datasetVersionNumber=2
```

HEAD 与 GET 在此服务可能表现不同；下载器使用 GET。每个响应最多 5 MB，每次网络请求超时 30 秒，仅对暂时性错误最多尝试 3 次。远端接口发生变化时会报错，绝不转为合成数据。下载器检查公开元数据仍标识 v2；若上传者更新当前版本，会要求先复核来源，而不静默接受变化。

## 本地复现

在仓库根目录运行：

```bash
python -m quant_research.etf_data --output-dir runs/etf-case-data-v1
```

输出目录必须不存在，以免覆盖已有研究证据。成功后返回 `runs/etf-case-data-v1/manifest.json`，可直接传入 `quant_research.data.load_dataset()`。其中记录固定标的、请求窗口、实际窗口、行数、快照版本、元数据与每份源 CSV 的 SHA-256，以及规范化 CSV 的 SHA-256。原始响应保存在 `raw/` 供本地核对。

日期只取八个标的在窗口内实际观测日期的交集，不前向填充，不后向填充，也不插值。不同标的缺失的日期、交集丢弃的日期和各标的因此丢弃的观测数量写入 manifest。这个检查基于来源实际出现的日期，不包含独立交易所日历验证：若所有文件共同漏掉一天，交集检查无法发现。重复日期、错误列、非正数或非有限调整价会立即失败。失败目录可能保留已下载的原始材料，但不会生成可加载的成功 manifest；重试时请选择新目录。

## 权利边界

Kaggle 上传者声明许可证为 **CC0: Public Domain**；这仅是上传者的声明。本项目没有独立核实 Yahoo Finance 上游权利或再分发许可，不能把这个声明等同于完成授权核验。因此原始和规范化的市场 CSV 仅保存于 Git 忽略的本地 `runs/`，不随仓库再分发。仓库公开下载代码、来源链接、数据摘要哈希和派生策略研究结果，使使用者能在自行确认数据条款后复现。
