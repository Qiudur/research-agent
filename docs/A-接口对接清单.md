# A 线交接与待接线事项

## B · 任务与评分

- 提供 harness.load_task/run_script/score，签名及 TaskView、RunResult、ScoreResult 按《接口约定 v1》。A 只读返回的 workspace/、profile.json、训练样例或 original.py，不接触 hidden/ 和 bank.json。
- 提供每道题起点 n0000 的真实 dev score/1 文件，供 A CLI 的 --baseline-json 使用。v1 的 TaskView 不含基线分数，而实验树必须存起点分数；这是最重要的接口缝隙。
- 明确 run_script(task_id, node_dir, ["solution.py"]) 是否可作为候选的导入/运行检查。A 当前先调用它，再调用 score(..., "dev")；评分失败或正确性失败会算一次修复机会。
- 评分器负责隔离 hidden、网络和越界写入，重新随机输入及作弊检测；A 不自行实现第二套评分器。
- 当前实现：根目录 `harness.py` / `harness_worker.py`，已有 `spd-distance-matrix` 练习题；macOS `sandbox-exec` 下已实测正确性、测速、holdout 门禁和 `reads_hidden` / `writes_outside_node` / `network` 标记。正式题库仍未达到每个场景 12 题，表格场景尚未接入。

## C · 账本与评估

- 提供 ledger.Ledger.open(...)/llm_call(...)/emit(...)/spent()/elapsed() 和 Usage 的 v1 接口。价格缺失应直接报错，绝不折算为 0。
- 在运行前创建 manifest.json，指定任务、模型、配置、价格文件、能力库根目录和冻结状态。填写真实价格与任务预算。A 读取 hard_caps.json，不允许自研究版修改。
- 开工报价由 C 的报价模块基于 A 的 opening_plan.json 或 TaskRunner.on_initial_plan 回调提供的 predicted、expected_progress 和历史预估误差生成 quote 事件。回调在首个实验前触发；当前 `evaluation.quote` 已接入 A CLI，A 不自行伪造报价。
- A 结束后由 C 调用 holdout，并写 task_end。考试期的经验草稿只追加到任务目录 experience_drafts.jsonl，不改冻结库；C 可据此在报告中说明考试时学到了什么。
- 用 A 产出的 tree.jsonl、账本、能力库 changelog 和 replay 报告检验报告里的每个数字。
- 当前实现：`ledger.py`、`evaluation/manifest.py`、`quote.py`、`finalize.py`、`orchestrate.py`、`report.py` 和 `compare.py`；TEST-only fixture 的 A+B+C 端到端链路已通过，正式价格和模型成本仍留空。

## 集成验收

1. ~~B/C 各交一个最小真实任务包和最简账本模块，A CLI 从 manifest 跑通至少一个 dev 实验。~~ 已完成：真实 B sandbox + C TEST-only ledger + A 主循环端到端通过。
2. ~~检查 n0000 真基线、B 的 RunResult/ScoreResult、C 的四桶 token 价格、预算和耗时与 A 记录一致。~~ 已完成于 TEST-only fixture；正式价格仍待提供。
3. 已完成接受节点回放、holdout 单次门禁和冻结库测试；失败节点回放有 A 单测覆盖，仍需用正式 B 题包复核。
4. 未完成：正式题库、价格和模型到位后，冻结代码/题库/能力库，再按 M3/M4 做三遍预演和正式考试。
