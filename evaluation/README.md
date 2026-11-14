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

`evaluation.admission.admit_configuration` 是验证集准入闸门：候选配置必须在三次配对运行中每次得分不低于默认配置，且平均成本更低，才返回 `admitted`；它与只比较分数的 `compare_configurations` 分开，避免把更高分但更贵的配置误当成可替换版本。

本地 `tests/fixtures/prices_test_only.json` 与 `test_only_budget` 只用于无 API 凭据的集成联调，货币标为 `TEST`，manifest 强制只允许 `dev`。根目录 `config/prices.json` 保持未填价格，不能用于正式运行。正式考试需要 C 写入真实价格、模型和预算，并把能力库冻结。
