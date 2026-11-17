# v0.1.0 — Auditable quant research MVP

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

This release packages the factor-research workflow, development selection, independent one-shot finalizer, token ledger, published evidence, and English-first bilingual documentation. It supports **macOS with `sandbox-exec` and Python 3.11**. The core runtime has no third-party dependencies; the `llm` extra is optional for real model calls. The commands below require neither an API key nor market-data downloads.

### Validation

A fresh remote clone of commit `f57fef8ceec352cebc45f0e1e424a60847adfa2b`, installed in a new virtual environment with Python **3.11.16**, passed **113 core quant tests**, with no failures, errors, or skips. A new synthetic scripted run completed development and its single finalizer evaluation with **zero model calls**. The 267 runtime/configuration/package-definition files matched this release preparation; all **38 Qwen evidence hashes** and **17 fixed-ETF checksums** were verified. See [the reproduction record](reproduction.json) for environment details and checks.

The [macOS CI workflow](https://github.com/Qiudur/research-agent/actions/workflows/quant-regression.yml) runs the core tests and synthetic smoke check on pushes, pull requests, and manual dispatches. Local equivalent checks passed; hosted CI has not yet been verified in this validation record.

### Reproduce from a fresh clone

Run on macOS with Python 3.11 available:

```bash
git clone --branch codex/quant-research-mvp https://github.com/Qiudur/research-agent.git
cd research-agent
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -p 'test_quant*.py' -v
.venv/bin/python -m quant_research.run \
  --project-root . --mode scripted --run-dir runs/v010-synthetic
.venv/bin/python -m quant_research.finalize_cli \
  --run-dir runs/v010-synthetic --evaluate-sealed-once
```

Use a new run directory for every repetition; the finalizer refuses a second attempt. This creates only a new synthetic engineering demonstration, without repeating either published market holdout. If a host-specific package index lacks a build dependency, explicitly use public PyPI with that host configuration disabled:

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  .venv/bin/python -m pip install --index-url https://pypi.org/simple -e .
```

### Research findings and scope

The [real Qwen experiment](../../qwen-etf-case/report.md) recorded **17,791 tokens across six successful calls**. All three executed candidates were rejected; selection retained `n0000`, the original five-session momentum signal. Improvement over that original signal was **zero**. Four earlier failures have unknown usage, so 17,791 tokens is not the total cost of all attempts. The [fixed-rule ETF case](../../etf-case-study/README.md) is reported separately. Both market holdouts have already been scored; they are not fresh validation sets for future tuning.

A project-wide code license has not been selected. See [license status](../../../LICENSE_STATUS.md) for the separate code, dataset, and dependency terms. The [project README](../../../README.md) explains the workflow, results, and limitations.

---

<a id="中文"></a>

## 中文

本版本交付因子研究流程、开发期选优、独立单次 finalizer、token 账本、公开证据，以及英文在前的双语文档。支持 **macOS、`sandbox-exec` 和 Python 3.11**。核心运行时没有第三方依赖；真实模型调用所需的 `llm` 扩展可选安装。下列命令不需要 API key，也不下载行情。

### 验证记录

从远端重新克隆提交 `f57fef8ceec352cebc45f0e1e424a60847adfa2b`，使用 Python **3.11.16** 建立全新虚拟环境后，**113 项核心量化测试**全部通过，没有失败、错误或跳过。新的合成数据脚本运行完成开发及唯一一次 finalizer 评分，**模型调用为零**。267 份运行代码、配置和包定义文件与本次发布准备一致；**38 项 Qwen 证据哈希**及 **17 项固定 ETF 校验和**全部核对通过。环境与检查明细见[复现记录](reproduction.json)。

[macOS CI](https://github.com/Qiudur/research-agent/actions/workflows/quant-regression.yml) 在推送、PR 和手动触发时执行核心测试及合成流程检查。本地等价检查已通过；这份验证记录尚未确认远端 CI 结果。

### 从干净克隆复现

在已安装 Python 3.11 的 macOS 上运行：

```bash
git clone --branch codex/quant-research-mvp https://github.com/Qiudur/research-agent.git
cd research-agent
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -p 'test_quant*.py' -v
.venv/bin/python -m quant_research.run \
  --project-root . --mode scripted --run-dir runs/v010-synthetic
.venv/bin/python -m quant_research.finalize_cli \
  --run-dir runs/v010-synthetic --evaluate-sealed-once
```

每次重复须使用新的运行目录，finalizer 会拒绝第二次尝试。此命令只创建新的合成工程演示，不重跑任何已公开的市场封存期。若本机私有包索引缺少构建依赖，可禁用该主机配置并显式使用公共 PyPI：

```bash
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  .venv/bin/python -m pip install --index-url https://pypi.org/simple -e .
```

### 研究结果与范围

[真实 Qwen 实验](../../qwen-etf-case/report.md)成功运行的六次调用记录 **17,791 tokens**。三个已执行候选均被拒绝，最终保留 `n0000` 原始 5 日动量，相对原始方案的改进为 **零**。此前四次失败的用量未知，所以 17,791 不是所有尝试的总成本。[固定规则 ETF 案例](../../etf-case-study/README.md)单独报告。两个市场封存期均已评分，不能作为未来调参的新验证集。

本项目尚未选择整体代码许可证；代码、数据和依赖的独立条款见[许可状态](../../../LICENSE_STATUS.md)。[项目 README](../../../README.md)说明完整流程、结果及限制。
