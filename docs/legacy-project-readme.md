# Core architecture and historical research index

[English](#english) · [中文](#中文)

<a id="english"></a>

## English

The research engine supplies the experiment tree, resource budgeting, versioned capability library, task scoring and audit trail used by the current quant research workflow. This page collects the core architecture references and earlier studies.

### Architecture and technical references

| Component | Responsibility | Reference |
| --- | --- | --- |
| `research_agent/` | Planning, execution, diagnosis and review; experiment-tree selection, budget checks and replay records. | [Schema and interface contract](接口约定.md) |
| `library/` | Versioned experiences, citations, credibility updates, frozen snapshots and verified skill loading. | [Skill layer](../skill_layer/README.md) |
| `harness.py`, `tabular_harness.py` | Code-speedup and tabular task loading, development scoring and isolated candidate execution. Real scoring requires macOS `sandbox-exec`. | [Schema and interface contract](接口约定.md) |
| `ledger.py`, `evaluation/` | Four disjoint token buckets, manifests, staged runs, practice-only self-research, validation admission and one-shot holdout evidence. | [Evaluation module](../evaluation/README.md) |
| `dashboard/` | Read-only export and local display of run, ledger, tree and library evidence. | [Dashboard](../dashboard/README.md) |

The [pilot](../pilot/README.md) documents the early timing and cross-run experience-loop checks. Its scripted roles and TEST accounting verify engineering behavior; they do not establish autonomous model learning or cash savings.

The current market-data workflow uses `quant_research/` with a separate development admission and one-shot finalizer. See [Quant Research MVP](quant-research-mvp.md) and the [project README](../README.md).

### Historical research and evidence

These reports describe the earlier speedup/tabular task-bank studies. Their tasks, scoring rules and evaluation periods differ from the current ETF studies; their numbers must not be presented as current quant results. Historical reports retain their original scope and limitations.

- [Real-model development pilot](真实模型试点.md): configurations, all attempts and negative results.
- [Practice-only automatic self-research](真实模型自研究试验.md): candidate comparisons and proposal costs.
- [Cross-task reevaluation](真实模型迁移复评.md): transfer results, failed calls and score-normalization limitations.
- [Concurrent development benchmark](真实模型并发基准.md): observed wall time and comparison limits.
- [Ten-practice-task benchmark](A-十题练习试验.md): the default Qwen configuration across three repetitions.
- [Frozen-bank internal token research](内部Token正式研究报告.md) and [partial formal-batch evidence](evidence/formal-batch-v2-partial.md): the validation rejection and interrupted exam are reported explicitly. There is no complete exam comparison or final break-even claim.
- [Speedup calibration](提速朴素成本校准.md), [tabular calibration](表格朴素成本校准.md) and [fixed aggregation rules](成本校准合并规则.md): measured resource accounting and retained failures.

The historical `TOKENS` units measure resources, not cash or official competition scores. Unknown usage remains unknown. Generated runs may be local and Git-ignored; the reports identify the available evidence and its limits.

---

<a id="中文"></a>

## 中文｜核心架构与历史研究索引

研究引擎提供了当前量化研究流程沿用的实验树、资源预算、版本化能力库、任务评分与审计链。本页汇总核心架构说明与早期研究。

### 架构与技术入口

| 组件 | 职责 | 说明 |
| --- | --- | --- |
| `research_agent/` | 规划、执行、诊断与复盘，以及实验树选优、预算检查和回放记录。 | [Schema 与接口约定](接口约定.md) |
| `library/` | 版本化经验、引用、可信度更新、冻结快照和技能校验加载。 | [技能层](../skill_layer/README.md) |
| `harness.py`、`tabular_harness.py` | 提速与表格题加载、开发期评分和隔离候选执行。真实评分需要 macOS `sandbox-exec`。 | [Schema 与接口约定](接口约定.md) |
| `ledger.py`、`evaluation/` | 四个互斥 token 桶、manifest、分阶段运行、仅练习题自研究、验证准入与单次 holdout 证据。 | [评估模块](../evaluation/README.md) |
| `dashboard/` | 只读导出和本地展示运行、账本、实验树及能力库证据。 | [看板](../dashboard/README.md) |

[早期 pilot](../pilot/README.md) 记录了真实计时与跨运行经验闭环检查。脚本角色和 TEST 计量验证工程行为，不证明模型自主学习或现金节省。

当前行情研究使用 `quant_research/`，具有独立的开发准入与单次 finalizer。详见[量化研究 MVP](quant-research-mvp.md)和[项目 README](../README.md)。

### 历史研究与证据

以下报告属于早期提速／表格题库研究。其任务、评分规则及评估区间与当前 ETF 研究不同，不能把其中数字作为当前量化结果；历史报告保留各自的范围与限制。

- [真实模型开发试点](真实模型试点.md)：配置、全部尝试与负结果。
- [仅练习题的自动自研究](真实模型自研究试验.md)：候选对比与提案成本。
- [跨题迁移复评](真实模型迁移复评.md)：迁移结果、失败调用和分数归一化限制。
- [并发开发基准](真实模型并发基准.md)：墙钟观察与比较边界。
- [十道练习题基准](A-十题练习试验.md)：默认 Qwen 配置的三遍运行。
- [冻结题库内部 token 研究](内部Token正式研究报告.md)及[正式批次部分证据](evidence/formal-batch-v2-partial.md)：明确记录验证拒绝与考试中止，没有完整考试对比或最终回本结论。
- [提速校准](提速朴素成本校准.md)、[表格校准](表格朴素成本校准.md)与[预先固定的合并规则](成本校准合并规则.md)：实测资源计量和保留的失败。

历史 `TOKENS` 单位计量资源，不是现金价格或官方比赛成绩；未知用量保持未知。生成的运行目录可能仅保存在本地并被 Git 忽略，各报告说明可用证据及其限制。
