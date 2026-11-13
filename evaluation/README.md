# C 线评估模块

`ledger.py` 实现 `ledger/1`：每个任务或运行一个只追加账本，模型 token 分为互斥的 `input`、`cached_input`、`output`、`reasoning` 四桶，价格表缺失或哈希不匹配会直接报错。

`evaluation.manifest.create_manifest` 在运行开始前封存任务、配置、模型、价格、能力库版本和 hard cap。`evaluation.quote.make_quote_callback` 接在 A 的 `TaskRunner.on_initial_plan` 上，在第一个实验前写入 `quote`；有历史预测/实际对时，按中位误差修正并留下 basis。

`evaluation.orchestrate.execute_task` 是 A+B+C 的单任务编排入口：B 提供 dev 基线和评分，A 运行主循环，C 接收报价、单次 holdout、`task_end`、回放封存和成本核对。holdout 只在 C 调用 B 的短暂 `RA_EVAL=1` 窗口内执行，候选进程不会继承该变量。`finish_run` 要求所有任务恰好完成一次后才写 `run_end` 并填 `ended_ts`。

`evaluation.report` 从账本、实验树和 holdout 文件提取带序号证据的任务/运行报告；`evaluation.compare.compare_configurations` 实现原定三遍判定：每个配置恰好重复 1、2、3，三次方向一致且平均差距超过配置内波动才报告 `better`，否则 `no_clear_difference`。

`evaluation.admission.admit_configuration` 是验证集准入闸门：候选配置必须在三次配对运行中每次得分不低于默认配置，且平均成本更低，才返回 `admitted`；它与只比较分数的 `compare_configurations` 分开，避免把更高分但更贵的配置误当成可替换版本。

本地 `tests/fixtures/prices_test_only.json` 与 `test_only_budget` 只用于无 API 凭据的集成联调，货币标为 `TEST`，manifest 强制只允许 `dev`。根目录 `config/prices.json` 保持未填价格，不能用于正式运行。正式考试需要 C 写入真实价格、模型和预算，并把能力库冻结。
