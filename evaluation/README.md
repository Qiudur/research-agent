[English](#english) · [中文](#中文)

<a id="english"></a>

# Evaluation module (C track)

## Internal research with a chosen token budget

When a task has no official cash budget, authorized internal research can use an isolated copy of the frozen task bank: select a model that can be deployed locally, a token budget per task, and a four-bucket metering table (`currency=TOKENS`, `test_only=false`). Under this protocol, `budget.money` is measured in token pricing units, not cash; `TOKENS` is a metering label, not an official competition result. Formal internal research requires `unit_tokens=1` and all four bucket rates equal to 1; otherwise the preflight, batch and self-research entry points reject it. Only then is recorded consumption the sum of ordinary input, cached input, visible output and reasoning output. Retain the existing task-bank freeze, code freeze, one-shot validation/exam rules and audit gates. A `test_only=true` development pilot does not qualify as formal internal research, and the original `tasks/bank.json` and `config/prices.json` must not be overwritten. Reports and dashboards identify internal research and state that its units are neither cash nor official scores.

## Formal operation commands

First provide the actual model name, a complete non-TEST price table in `config/prices.json`, and a positive budget for every task; freeze the task bank and commit the code. Install the project's `llm`, `speedup` and `tabular` dependencies, and set `OPENAI_API_KEY` on a machine with macOS `sandbox-exec`. These commands accept formal inputs only, with no fake TEST-data option; use a fresh directory name each time:

```bash
export MODEL='actual-model-name'
export CODE_COMMIT="$(git rev-parse HEAD)"
python3 -m evaluation.cli auto-research --model "$MODEL" --prices config/prices.json --code-commit "$CODE_COMMIT" --study-dir batches/auto-v1 --rounds 3
export TUNED_CONFIG="$(python3 -c 'import json; from pathlib import Path; from evaluation.self_research import verify_auto_selected; print(verify_auto_selected(json.loads(Path("batches/auto-v1/auto_self_research_report.json").read_text())))')"
python3 -m evaluation.cli preflight --model "$MODEL" --prices config/prices.json --code-commit "$CODE_COMMIT" --tuned-config "$TUNED_CONFIG" --exam
python3 -m evaluation.cli batch --model "$MODEL" --prices config/prices.json --code-commit "$CODE_COMMIT" --tuned-config "$TUNED_CONFIG" --batch-dir batches/formal-v1 --exam
python3 -m evaluation.cli figures --batch-dir batches/formal-v1 --auto-study-dir batches/auto-v1 --output batches/formal-v1/study.html
```

For practice and a single validation attempt only, omit `--exam` from the first batch command. This entry point does not open another batch to repeat validation and add an exam after the validation marker has been consumed. Preflight does not modify files: it returns exit code 1 and `blockers` when prerequisites are missing, or 2 for execution errors. Without a real model, prices, budgets or frozen task bank, stop at preflight. TEST-only smoke checks inject test roles and prices through the Python API; the formal CLI cannot launch them.

## Self-research on practice tasks

`evaluation.self_research.evaluate_candidates` accepts named `agent/1` configurations, each of which may reference its own prompt files; a comparison of multiple candidates must include the `default` control. Each candidate runs independently three times on the practice partition in fixed task order. The result includes all audited `run_report/1` records, costs per repetition, mean score, mean spending, score-to-spending ratio and the selected candidate. Each candidate configuration and prompt file has a SHA-256 record; `verify_selected` rechecks the hashes before validation admission. This entry point opens practice packages only, never validation or exam tasks; critical price, bank and hard-cap files are checked before and after each repetition.

`evaluation.self_research.auto_evaluate_candidates` starts from `default` and automatically proposes up to eight rounds of candidates. Each model response may contain only bounded `agent/1` settings and short strategy appendices for the four roles; the original prompts remain unchanged prefixes. Proposal calls use `ModelClient`. A separate `proposal_ledger/1` records the four token buckets, price-table SHA, source-prompt SHA, response status and any reported usage from failed calls. Candidate ranking includes that candidate's proposal cost; total research cost also includes rejected candidates and failed calls. Output uses `auto_self_research/1`: `selection` identifies the best **generated** candidate, while `overall_best_on_practice` also records the overall winner including `default`. Even if `default` scores better in practice, a generated candidate must pass validation admission before it can enter the exam. If there is no valid generated candidate, `selection=null` and the workflow stops at practice. Use `verify_auto_selected` to check the selected version. Each call's `max_output_tokens` must be between 128 and 4096.

Before any calls, new studies write `paired-scenario-quality-v1` into `auto_plan.json` as `selection_policy`. Pair each practice task with `default` on the same task and repetition: tabular tasks use raw holdout AUC, while speedup tasks use holdout scores clipped to [0,1]. Average the three paired differences per task, then average within each scenario. Selection first compares the worse of the two scenario improvements, then the mean improvement across both scenarios, then audited total research cost. The existing `mean_score` and `score_per_currency` remain diagnostic fields and no longer select winners in new studies. Independent audits use the old policy for historical reports without `selection_policy`; historical conclusions are not rewritten.

Every model call has `usage_status=reported` or `unknown`. A response without complete token usage is a failure even if it contains text; unknown usage is not zero cost. Once usage is unknown, the current task immediately stops further model requests and does not retry that call. A completed run containing unknown usage has both `cost_complete=false` and `exact_token_accounting=false` in its run report and is ineligible for self-research selection. On an interrupted failure, inner `self_research_report.json` and outer `auto_self_research_report.json` preserve `status=partial`, completed runs, failures, a known-cost lower bound and sequence numbers for unknown usage, with `selection=null`. The original exception is still raised; partial reports cannot admit a candidate to validation or the exam.

Automatic research has no overall monetary cap, but each inner task still has a positive budget and a 90% hard gate; it must not be described as having unbudgeted inner tasks. Formal self-research requires real prices, API credentials and a frozen bank. TEST-only runs use a development manifest and TEST prices and establish only that the mechanism works. The selected configuration must still pass the batch workflow's single validation admission before entering a formal exam.

## Batch orchestration across three repetitions

`evaluation.batch.execute_batch` accepts `project_root`, an exclusive `batch_dir`, B's harness, per-task `baseline_solutions`, per-configuration `agent_configs`, `prices_path`, `model`, `code_commit`, and optional `run_validation`, `run_exam` and `test_only_budget`. It returns `batch_report/1`; `reports` are grouped by `practice`, `validation` and `exam`. Each item is a `run_report/1` verified against the ledger, experiment tree and holdout, ready for the figures module. Each run also saves `report.json`; the batch directory saves `plan.json` and `batch_report.json`.

Each configuration runs repetitions 1–3, starting from an empty library and practicing in bank order each time. The practice libraries of `naive/default` do not enter the exam, preserving the empty-library controls. The `default_lib/tuned_lib` libraries are separately copied into frozen validation and exam snapshots. Validation first compares `default_lib` with `tuned_lib`; a lower score or no reduction in mean cost excludes `tuned_lib` from the exam. Formal validation and exam each use a global one-shot attempt marker for the same bank version; failures are not automatically rerun.

TEST-only mode permits scripted roles and temporary budgets, but the underlying manifest must have `phase=dev`. A batch report's `logical_phase` describes workflow position only, not a formal validation or exam result. Formal mode rejects TEST prices, scripted roles, unfrozen or incomplete banks, missing budgets and missing API credentials. A formal exam also requires four configurations with distinguishable settings. Each formal repetition checks that Git HEAD equals `code_commit` and that the checkout is clean. Store outputs in ignored `batches/` or `runs/`. This entry point does not perform self-research's uncapped exploration with formal prices.

Figure generation reads exam reports from the same batch's `batch_report`, re-audits all six validation runs and recalculates admission. Admission yields four configurations × three repetitions; rejection yields the other three configurations × three repetitions, explicitly states that the tuned version did not enter the exam and does not calculate its break-even point. The command is:

```bash
python3 -m evaluation.figures --batch-dir batches/<batch> --auto-study-dir batches/<self-study> --output batches/<batch>/study.html
```

`auto-study-dir` re-audits every practice `run_report` and the `proposal_ledger`, verifies that the selected self-research configuration matches the tuned validation/exam manifests in path and hash, and includes failed proposal costs in break-even calculations. TEST development results carry a prominent watermark and cannot be presented as formal exam results.

`ledger.py` implements `ledger/1`: one append-only ledger per task or run, with four disjoint token buckets: `input`, `cached_input`, `output` and `reasoning`. A missing price table or mismatched hash raises an error.

`evaluation.manifest.create_manifest` seals the tasks, configuration, model, prices, library version and hard cap before a run starts. `evaluation.quote.make_quote_callback` hooks into A's `TaskRunner.on_initial_plan` and writes a `quote` before the first experiment; where historical predicted/actual pairs exist, it corrects by median error and records the basis. Batch history uses only complete audited reports from earlier tasks in the same run and scenario, avoiding mixtures across configurations or repetitions. The first task has no history and retains its uncalibrated original quote.

`evaluation.orchestrate.execute_task` is the single-task A+B+C entry point: B supplies the development baseline and scoring, A runs the main loop, and C receives the quote, one-shot holdout, `task_end`, sealed replay and cost reconciliation. Holdout runs only within the brief `RA_EVAL=1` window when C calls B; candidate processes do not inherit the variable. `finish_run` writes `run_end` and `ended_ts` only when every task has completed exactly once.

`evaluation.report` extracts task/run reports with sequence-numbered evidence from the ledger, experiment tree and holdout files. `evaluation.compare.compare_configurations` implements the original three-repetition criterion: each configuration has repetitions 1, 2 and 3, all three differences point in the same direction, and the mean gap exceeds within-configuration variation before it reports `better`; otherwise it reports `no_clear_difference`.

`evaluation.finalize.finish_task` evaluates a private copy of the best node's `solution.py` (`holdout_candidate/<node>/`) on the holdout exactly once. Before and after scoring, it verifies that the original file still exists and its SHA-256 matches the sealed record. Scorer exceptions, missing or changed files, or invalid responses from B produce `holdout_score.json` with `score: null` and `task_end.holdout_failed: true`, without crashing the run; the one-shot marker remains and no retry occurs. The baseline (`n0000`) receives a separate reference score on the same holdout in `baseline_holdout.json`; `task_end` includes `baseline_holdout_score`, `baseline_holdout_raw` and `holdout_raw_lift`. Task reports from `evaluation.report` include `holdout_failed`, `holdout_raw_lift`, `budget_violation` and `budget_excess`; run reports aggregate `holdout_failures` and `budget_violations`. In formal token mode, exceeding a task budget is recorded as a violation; inconsistent ledger units or an invalid budget cause rejection. `evaluation.comparable.comparable_reports` excludes holdout-failed tasks from all runs in the same comparison group and lists the exclusions. `admit_complete` returns a claim of `incomplete` for missing runs (the comparison logic in `compare.py` is unchanged). `execute_batch` isolates individual run failures in `failed_runs/skipped_runs` within `batch_report.json` while other runs continue; it always writes a final batch report with `status: complete|partial`.

`evaluation.admission.admit_configuration` is the validation admission gate: a candidate must score at least as well as the default in each of three paired runs and have lower mean cost to return `admitted`. It is separate from score-only `compare_configurations`, avoiding admission of a higher-scoring but more expensive configuration as a replacement.

Local `tests/fixtures/prices_test_only.json` and `test_only_budget` are only for integration checks without API credentials. Their currency is `TEST`, and manifests permit only `dev`. The root `config/prices.json` remains unfilled and cannot support formal runs. Formal exams require C to provide real prices, the model and budgets, and freeze the library.

---

<a id="中文"></a>

# C 线评估模块

## 自选 token 预算的内部研究口径

当题目未给官方现金预算时，可以在隔离副本建立经授权的内部冻结题库研究：选定可本地部署模型、单题 token 预算和四桶计量表（`currency=TOKENS`、`test_only=false`）。此协议下 `budget.money` 按 token 计价单位解释，不代表现金；`TOKENS` 仅是计量标签，不是官方比赛成绩。内部正式研究必须将 `unit_tokens` 设为 1，且四桶费率都设为 1，否则预检、批量与自研究入口拒绝；此时记录的消耗才是普通输入、缓存输入、可见输出、推理输出四桶之和。实验需保留原有题库冻结、代码冻结、一次性验证/考试和审计门禁；不能把 `test_only=true` 的 dev 试点当作内部正式研究，也不能覆盖原始 `tasks/bank.json` 或 `config/prices.json`。报告和看板会明确写出内部研究标识、非现金与非官方声明。

## 正式操作命令

先填入真实模型名、config/prices.json 的完整非 TEST 价目、各题正预算，冻结题库并提交代码；安装项目的 llm、speedup、tabular 依赖，在有 macOS sandbox-exec 的机器设置 OPENAI_API_KEY。以下命令只接受正式输入，不提供 TEST 假数据选项；目录名应每次唯一：

    export MODEL='实际模型名'
    export CODE_COMMIT="$(git rev-parse HEAD)"
    python3 -m evaluation.cli auto-research --model "$MODEL" --prices config/prices.json --code-commit "$CODE_COMMIT" --study-dir batches/auto-v1 --rounds 3
    export TUNED_CONFIG="$(python3 -c 'import json; from pathlib import Path; from evaluation.self_research import verify_auto_selected; print(verify_auto_selected(json.loads(Path("batches/auto-v1/auto_self_research_report.json").read_text())))')"
    python3 -m evaluation.cli preflight --model "$MODEL" --prices config/prices.json --code-commit "$CODE_COMMIT" --tuned-config "$TUNED_CONFIG" --exam
    python3 -m evaluation.cli batch --model "$MODEL" --prices config/prices.json --code-commit "$CODE_COMMIT" --tuned-config "$TUNED_CONFIG" --batch-dir batches/formal-v1 --exam
    python3 -m evaluation.cli figures --batch-dir batches/formal-v1 --auto-study-dir batches/auto-v1 --output batches/formal-v1/study.html

若只想执行练习与一次验证，首次 batch 命令可省略 --exam；本入口不会在验证标记已经消耗后另开一批重跑验证以补考试。preflight 不修改文件，准备不足时返回退出码 1 和 blockers；运行错误返回 2。无真实模型、价目、预算或冻结题库时必须停在预检。TEST-only 烟测使用 Python API 注入测试角色和价目，不能从正式 CLI 发起。

## 练习题自研究

evaluation.self_research.evaluate_candidates 接受具名 agent/1 配置，配置可引用各自的提示词文件；多候选比较必须包含 default 对照。每个候选在题库练习层按固定顺序独立跑三遍，返回全部审计过的 run_report/1 和每遍成本、平均分、平均花费、分数/花费比以及选中候选。候选配置与提示词逐文件记录 SHA256，verify_selected 在交给验证集准入前复核哈希。此入口只打开练习题包，不运行验证或考试题；关键价格、题库和硬上限文件在每遍前后核对。

evaluation.self_research.auto_evaluate_candidates 从 default 配置开始，自动提出最多 8 轮候选；每轮模型输出只允许有界 agent/1 设置和四角色短策略附录，原提示词作为不变前缀保留。生成调用走 ModelClient，四桶 token、价目 SHA、源提示词 SHA、响应状态和失败调用可得 token 记入独立 proposal_ledger/1；候选排名将其生成费计入本候选成本，整体研究成本还包括被拒候选和失败调用。输出 auto_self_research/1，selection 是最佳**生成**候选，overall_best_on_practice 同时记录含默认版的总胜者；即使默认版练习表现更好，生成候选仍须过验证准入才可考试。若无有效生成候选，selection 为 null，停止于练习阶段。可用 verify_auto_selected 核验选中版本。每次调用的 max_output_tokens 只允许 128–4096。

新研究在发起调用前把 `paired-scenario-quality-v1` 写入 auto_plan.json 的 selection_policy。每道练习题按同题、同遍与 default 配对：表格题使用原始 holdout AUC，提速题使用截断到 [0,1] 的 holdout 分数；三遍差值先按题平均，再按场景平均。选优先比两个场景中较差的平均改善，再比两场景平均改善，最后比已审计的总研究成本。原有 mean_score 与 score_per_currency 继续保存作诊断，不再决定新研究的胜者。旧报告没有 selection_policy 时，独立审计仍按旧规则复核，历史结论不回写。

每次模型调用的 usage_status 为 reported 或 unknown；未得到完整 token 用量的响应即使有正文也视为失败，未知用量不解释为零成本。调用一旦出现未知用量，当前任务立即停止后续模型请求，不对该调用重试。完整运行若包含未知用量，run_report 的 cost_complete 与 exact_token_accounting 均为 false，不能参加自研究选优。中途失败时，内层 self_research_report.json 和外层 auto_self_research_report.json 保存 status=partial、已完成运行、失败记录、已知成本下界及未知用量序号，selection=null；原异常仍向调用方抛出，不能据部分报告进入验证／考试。

自动研究没有整体货币上限，但每次内层任务仍保留正预算和 90% 硬闸；不能把它表述为“内层任务无预算”。正式自研究须有真实价目、API 凭据和冻结题库；TEST-only 使用 dev manifest 和 TEST 价目，只说明机制可运行。选中配置必须再通过 batch 的一次验证准入才能进正式考试。

## 三遍批量编排

evaluation.batch.execute_batch 接受 project_root、独占的 batch_dir、B 的 harness、每题的 baseline_solutions、每配置的 agent_configs、prices_path、model、code_commit，以及可选的 run_validation、run_exam、test_only_budget。返回 batch_report/1；reports 按 practice、validation、exam 分组，组内每一项都是由账本、实验树和 holdout 核对过的 run_report/1，可直接交给出图模块。每个运行也单独保存 report.json，batch_dir 保存 plan.json 和 batch_report.json。

每个配置重复 1–3 遍，每遍均从空库开始、按题库顺序练习。naive/default 的练习库不带入考试，以保持空能力库对照；default_lib/tuned_lib 的库分别复制为冻结的验证和考试快照。验证集先比较 default_lib 与 tuned_lib；若研究版得分下降或平均成本不降，就不允许 tuned_lib 进入考试。正式验证和考试分别对同一题库版本设全局一次性尝试标记，失败不自动重跑。

TEST-only 模式允许脚本角色与临时预算，但底层 manifest 的 phase 必为 dev；batch_report 的 logical_phase 仅表示流程位置，不能作为正式验证或考试成绩。正式模式拒绝 TEST 价目、脚本角色、未冻结或未完成的题库、缺预算和缺 API 凭据；正式考试还要求四条配置及各自可区分的设置。正式运行每遍检查 Git HEAD 与 code_commit 一致且工作区洁净；运行输出应放在已忽略的 batches/ 或 runs/。自研究的无预算正式价格探索不由此入口完成。

出图从同一批次的 batch_report 读取考试报告，并重审六份验证运行、重新计算准入。准入通过时是四配置×三遍；准入拒绝时是其余三配置×三遍，明确研究版未进考试且不计算其回本点。命令入口是 python3 -m evaluation.figures --batch-dir batches/某批次 --auto-study-dir batches/某自研究 --output batches/某批次/study.html。auto-study-dir 会重审全部练习 run_report 和 proposal_ledger，校验自研究选中配置与验证/考试研究版 manifest 的路径和哈希一致，再把失败提案成本计入回本点。TEST dev 结果带显著水印，不能伪装为正式考试。

`ledger.py` 实现 `ledger/1`：每个任务或运行一个只追加账本，模型 token 分为互斥的 `input`、`cached_input`、`output`、`reasoning` 四桶，价格表缺失或哈希不匹配会直接报错。

`evaluation.manifest.create_manifest` 在运行开始前封存任务、配置、模型、价格、能力库版本和 hard cap。`evaluation.quote.make_quote_callback` 接在 A 的 `TaskRunner.on_initial_plan` 上，在第一个实验前写入 `quote`；有历史预测/实际对时，按中位误差修正并留下 basis。 batch 只取同一次运行中更早、同场景任务的完整审计报告作历史，避免跨配置或不同重复轮次混用；首题没有历史，仍保留未经校准的原始报价。

`evaluation.orchestrate.execute_task` 是 A+B+C 的单任务编排入口：B 提供 dev 基线和评分，A 运行主循环，C 接收报价、单次 holdout、`task_end`、回放封存和成本核对。holdout 只在 C 调用 B 的短暂 `RA_EVAL=1` 窗口内执行，候选进程不会继承该变量。`finish_run` 要求所有任务恰好完成一次后才写 `run_end` 并填 `ended_ts`。

`evaluation.report` 从账本、实验树和 holdout 文件提取带序号证据的任务/运行报告；`evaluation.compare.compare_configurations` 实现原定三遍判定：每个配置恰好重复 1、2、3，三次方向一致且平均差距超过配置内波动才报告 `better`，否则 `no_clear_difference`。

`evaluation.finalize.finish_task` 在最佳节点 `solution.py` 的私有副本（`holdout_candidate/<节点>/`）上做唯一一次 holdout，打分前后核对原文件仍在且 SHA-256 与封存记录一致；评分器抛错、文件缺失或被改、B 返回无效都记入 `holdout_score.json`（`score: null`）并让 `task_end` 带 `holdout_failed: true`，不使运行崩溃，一次性标记不变、不重试。基线（n0000）另在同一份 holdout 上打一次参照分写入 `baseline_holdout.json`，`task_end` 带 `baseline_holdout_score`、`baseline_holdout_raw` 和 `holdout_raw_lift`。`evaluation.report` 的任务报告对应带 `holdout_failed`、`holdout_raw_lift`、`budget_violation`、`budget_excess`，运行报告汇总 `holdout_failures` 与 `budget_violations`；正式 token 模式下单题超预算只记违规，账本单位不一致或预算无效才拒绝。`evaluation.comparable` 的 `comparable_reports` 把 holdout 失败的题从同组所有运行里剔除并列明，`admit_complete` 在运行不齐全时返回 claim 为 `incomplete`（比较逻辑 `compare.py` 不变）。`execute_batch` 把单个运行的失败隔离进 `batch_report.json` 的 `failed_runs`/`skipped_runs`，其余运行继续，批次结束一定写出 batch_report，并带 `status: complete|partial`。

`evaluation.admission.admit_configuration` 是验证集准入闸门：候选配置必须在三次配对运行中每次得分不低于默认配置，且平均成本更低，才返回 `admitted`；它与只比较分数的 `compare_configurations` 分开，避免把更高分但更贵的配置误当成可替换版本。

本地 `tests/fixtures/prices_test_only.json` 与 `test_only_budget` 只用于无 API 凭据的集成联调，货币标为 `TEST`，manifest 强制只允许 `dev`。根目录 `config/prices.json` 保持未填价格，不能用于正式运行。正式考试需要 C 写入真实价格、模型和预算，并把能力库冻结。
