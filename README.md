# Research Agent · A/B/C 最小闭环

这是《分工文档》中 A/B/C 最小闭环的 Python 实现：同一模型、四种角色、规则调度、实验树、预算闸门、可版本回放的能力库；B 提供任务与评分沙箱，C 提供价格账本、报价、holdout 和报告。A 不读取 bank.json 或 hidden/。

## 当前状态

A/B/C 最小闭环已接通：B 的真实 macOS sandbox 评分器、C 的四桶账本/报价/holdout/finalize 与 A 的模型角色和主循环已通过端到端测试。A/B/C 测试共 80 项，另有技能层 6 项、看板 5 项通过。正式研究闭环仍**未验收完成**：当前没有真实模型凭据、正式模型价格、正式预算，也只有 1 道 speedup 练习题，因此不能宣称正式比赛成绩或成本下降。

已新增一个独立的[真实计时试验](pilot/首次试验报告.md)：用实际代码和随机输入测得距离矩阵候选约 4.3 倍提速，并让 A 的主循环记录接受节点。该试验的角色输出是固定的，未调用模型或测量模型成本，不能代替上述真实研究闭环验收。

另有[跨运行经验联调](pilot/自进化联调报告.md)：第一轮写入经验，第二轮检索、引用并根据实测结果更新可信度。两轮使用同一已知候选与固定角色逻辑，因此只验证自进化机制的接线，不证明自主发现或经验带来效果提升。

## 已实现

- research_agent/model.py、roles.py、config/prompts/：Responses API 模型接入、角色推理强度、结构化输出、互斥 token 记账。
- research_agent/scheduler.py、tree.py、runner.py：假设排序、按最佳节点扩展、每四次探索一次、最多三次修复、dev 接受规则、诊断与复盘、90% 预算闸门、任务结束总结。
- library/：经验检索、可信度、条件收窄、版本快照、冻结和历史版本只读。
- research_agent/cli.py、replay.py：运行一个任务、按保存的种子与能力库版本回放节点。
- docs/A-决策记录.md：无人值守实施中做出的决定与复盘点。
- harness.py、harness_worker.py、tasks/：B 的任务加载、macOS 沙箱、随机测速、正确性检查、holdout 门禁和防作弊标记；正式任务 `budget.money` 保持 `null`。
- ledger.py、evaluation/：C 的 manifest/价格哈希、四桶 token 计价、开工报价、单次 holdout、task_end/run_end、可追溯报告、三遍比较和准入闸门。
- dashboard/、library/skills.py、skill_layer/：只读离线看板，以及需要跨两个任务实证后才能晋升的技能层。

## 本地验证

Python 3.11+；核心测试不需要 API Key 或真实价格：

    python3 -m unittest discover -s tests -v
    python3 -m unittest discover -s skill_layer/tests -v
    python3 -m unittest discover -s dashboard/tests -v

安装 live 模型客户端时：

    python3 -m pip install -e '.[llm]'

正式运行需有真实 `config/prices.json`、任务预算、模型名、API 凭据，以及 C 创建的 manifest 和 B 提供的起点 dev score/1 文件；不要把 Key 写入仓库。A CLI 运行单任务：

    python3 -m research_agent.cli run --run-dir runs/<run_id> --task-id <task_id> --baseline-json <baseline_score.json>

按实验编号重新评分：

    python3 -m research_agent.cli replay --run-dir runs/<run_id> --task-id <task_id> --node-id n0007

CLI 不负责创建题包、猜价格、生成基线或调用 holdout；C 的 `evaluation.orchestrate.execute_task` 负责把 B dev、A 主循环、回放封存和一次 holdout 串起来。无 API 凭据时可用 `tests/fixtures/prices_test_only.json` 与 `test_only_budget` 做 TEST-only 联调，不能把它当作真实花费。

## B/C 对接

见 [A-接口对接清单](docs/A-接口对接清单.md)；外部 schema 以 [接口约定 v1](docs/接口约定.md) 为准。
