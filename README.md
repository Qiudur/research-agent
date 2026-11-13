# Research Agent · A 线

这是《分工文档》中 A 线的 Python 实现：同一模型、四种角色、规则调度、实验树、预算闸门、可版本回放的能力库。任务与评分由 B 的 harness 提供，价格与账本由 C 的 ledger 提供；A 不读取 bank.json 或 hidden/。

## 当前状态

A 侧核心实现与 49 项本地测试已完成；A 的整条任务线**尚未验收完成**。目录中没有 B 的真实 harness/赛题、C 的真实 ledger/价格/manifest，也没有可付费调用的正式任务配置。因此尚未跑真实研究闭环，尚未根据真实题目调优提示词、证明分数提升或成本下降。把“代码实现完成”与“赛题效果完成”分开看；后者需要 B/C 接线后继续。

## 已实现

- research_agent/model.py、roles.py、config/prompts/：Responses API 模型接入、角色推理强度、结构化输出、互斥 token 记账。
- research_agent/scheduler.py、tree.py、runner.py：假设排序、按最佳节点扩展、每四次探索一次、最多三次修复、dev 接受规则、诊断与复盘、90% 预算闸门、任务结束总结。
- library/：经验检索、可信度、条件收窄、版本快照、冻结和历史版本只读。
- research_agent/cli.py、replay.py：运行一个任务、按保存的种子与能力库版本回放节点。
- docs/A-决策记录.md：无人值守实施中做出的决定与复盘点。

## 本地验证

Python 3.11+；测试仅用标准库，不需要 API Key、真实赛题或付费调用：

    python3 -m unittest discover -s tests -v

安装 live 模型客户端时：

    python3 -m pip install -e '.[llm]'

运行前需有 B 的 harness 模块、C 的 ledger 模块与 manifest、真实 config/prices.json、任务预算、模型名和 B/C 提供的起点 dev score/1 文件。环境中设置真实模型所需 API 凭据；不要把 Key 写入仓库。运行单个任务：

    python3 -m research_agent.cli run --run-dir runs/<run_id> --task-id <task_id> --baseline-json <baseline_score.json>

按实验编号重新评分：

    python3 -m research_agent.cli replay --run-dir runs/<run_id> --task-id <task_id> --node-id n0007

CLI 不负责创建题包、猜价格、生成基线或调用 holdout。正式任务完成后的 holdout 和 task_end 事件由 C 的评估编排执行。当前目录还没有 B/C 的生产实现，因此本仓库测试的是 A 与其接口替身的端到端行为，不代表已有比赛成绩。

## B/C 对接

见 [A-接口对接清单](docs/A-接口对接清单.md)；外部 schema 以 [接口约定 v1](docs/接口约定.md) 为准。
