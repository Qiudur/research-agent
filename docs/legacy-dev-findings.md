# Legacy development findings

[English](#english) · [中文](#中文)

<a id="english"></a>

These findings are a historical snapshot from **2026-11-14** of the inherited speedup and tabular evaluation system. Statements about budgets, task-bank contents, implementation state, and unopened holdouts describe that date; they are not the current configuration or evidence of the quant project's performance. The measurements, failures, TEST-only limits, and data-leakage observations are retained for traceability.

## 2026-11-14

### Formal task budgets were not yet set

`tasks/spd-distance-matrix/task.json` kept `budget.money=null`. Local test prices were not competition prices and could not be written into formal task packages or the task bank's original scores.

### Temporary task packages supplied integration-test budgets

Integration tests could point `RA_TASKS_ROOT` at temporarily copied task packages and inject a TEST budget through `evaluation.orchestrate.execute_task(..., test_only_budget=...)`. Temporary control files were written into the run directory; formal task packages were not rewritten.

### The speedup reference used chunked NumPy calculations

The first task's hidden reference used chunked NumPy distance calculations and returned a Python `list[list[float]]`. This avoided distorting the reference ceiling with the original pilot's slow symmetric-loop reference. A single local measurement on 2400 points took approximately 0.27 seconds for the original double loop and 0.056 seconds for the NumPy reference; formal scoring still used medians from multiple alternating measurements.

### The sandbox did not permit a weaker fallback

Scoring and exploration scripts required macOS `sandbox-exec`. A missing command or a non-macOS platform caused an error rather than falling back to AST checks alone. Node processes could not use the network, read `hidden/` or other tasks, or write outside their own node directories.

### Task coverage at the time

The speedup bank had grown to 12 tasks: 5 practice, 2 validation, and 5 exam tasks. `spd-already-vectorized` was a stop-loss trap. Four basic algorithm categories and eight extended categories passed real macOS sandbox development scoring. Regression tests covered correct and incorrect candidates, reference implementations, the holdout gate, and isolation from hidden data and the user's HOME directory.

The original implementation and hidden reference candidate for each speedup task were timed on development data with three fixed seeds. Median raw speedup ratios supplied `baseline_raw` and `ceiling_raw` in `tasks/bank.json`. Per-seed data, platform, and timestamps are retained in [the speedup calibration evidence](B-提速校准.json). The stop-loss task used identical original and reference code, so timing noise could put the measured reference below the baseline; the ceiling was not to be artificially raised.

Formal `naive_cost`, its derived `difficulty`, and task `budget.money` remained unset because formal model credentials, model pricing, and measured task costs were missing. TEST prices or timing results could not substitute for these fields. The bank remained `frozen=false`, and formal stratified evaluation could not be described as complete.

### Public tabular splits and retained failures

New UCI Adult, Mushroom, Online Shoppers, WDBC, and Cleveland Heart task packages used fixed seeds and feature-vector grouping for stratified 60/20/20 splits. Development measurements used sandboxed LightGBM and AutoGluon `medium_quality` with a 180-second limit. Source-archive hashes, feature transformations, split statistics, and reference measurements were retained in each task's `SOURCE.md`, `split_manifest.json`, and `hidden/reference_measurements.json`. Holdout data was generated but not used for calibration.

UCI Spambase and Wine Quality also retained separate packages and measurements, but their AutoGluon development AUC was below LightGBM. Their normalized gap was invalid, so `metric` was not filled and they were excluded from bank-v1. Mushroom's initial eight-feature version reached a perfect development score; that feedback influenced the four-feature definition, restricting it to practice rather than validation or exam use.

### A TEST-only batch smoke check used the real sandbox

`tests/test_real_batch_e2e.py` selected one practice, one validation, and one exam task in a temporary project, with four configurations and three repetitions each. It exercised the real Agent loop, macOS sandbox scorer, batch orchestration, four-bucket ledger, quotes, frozen libraries, admission, one-shot holdouts, `run_report`, and figures. On 2026-11-14, 30 tasks ran locally and the test passed in 56.5 seconds. Role responses and candidate code were explicit `ScriptedRoles` and known reference solutions; token usage and prices were fake TEST data. This established pipeline integration, not autonomous research performance or actual spending.

### Tabular stratification, a leakage trap, and unstable gaps

The proposed formal tabular bank had 12 tasks with measured positive gaps: Adult, Mushroom, Online Shoppers, Phishing Websites, and German Credit for practice; WDBC and Magic Gamma for validation; Bank Marketing, Cleveland Heart, HTRU2, Default Credit, and Australian Credit for the exam. Each task's baseline/ceiling used actual legal LightGBM/AutoGluon AUC on its fixed development split, with details in `hidden/reference_measurements.json`. EEG Eye State, QSAR Biodegradation, Spambase, and Wine Quality had non-positive reference gaps; their failure records were retained and they were not added to the bank.

Online Shoppers had a positive gap of only 0.000133 and Phishing Websites only 0.000011. These single-development differences were too small to support exam-ceiling evidence, so the tasks were limited to practice. Mushroom's feature choice was influenced by earlier development feedback and was likewise restricted to practice. Bank Marketing's `train.csv` retained `duration`, which becomes known only after a call; the task explicitly prohibited it and inference rows omitted it. AUC using only `duration` on the visible training data was 0.819829, demonstrating the trap's predictive power. Legal LightGBM and AutoGluon implementations both dropped `duration`; recalibrated development AUC was still 0.798154 and 0.802219. A candidate that used `duration` failed in real sandbox development scoring. This made Bank Marketing an empirically checked leakage trap; its exam holdout had not been opened at the time.

Formal tabular `score` evaluated three consecutive random seeds in separate sandbox runs, used the median normalized score as `score`, and recorded the population standard deviation as `spread`. A single-development `raw_score` call remained available for bank calibration. This avoided reporting unmeasured variation as zero; nondeterministic candidates could increase scoring cost and time.

---

<a id="中文"></a>

# 历史开发验证记录

以下为继承的提速与表格评估系统在 **2026-11-14** 的历史快照。预算、题库、实现状态及“holdout 尚未打开”等表述均指当时情况，不代表当前配置或量化项目效果。原始测量、失败、TEST-only 限定与数据泄漏观察保留供追溯。

## 2026-11-14

### 正式任务预算保持未定

tasks/spd-distance-matrix/task.json 的 budget.money 保持 null。本机测试价格不是比赛价格，不能写进正式任务包或题库原始分。

### 端到端联调用临时题包覆盖预算

C 的联调可设置 RA_TASKS_ROOT 指向临时复制的题包，并由 evaluation.orchestrate.execute_task(..., test_only_budget=...) 注入 TEST 预算。临时控制文件会写入运行目录；正式题包不被改写。

### 提速参考解采用 NumPy 分块实现

首道题的隐藏参考解使用分块 NumPy 距离计算并返回 Python list[list[float]]，避免原 pilot 的对称循环参考解过慢而造成参考上限失真。实测原始双循环约 0.27 秒、NumPy 参考约 0.056 秒（2400 点，单次本机测量）；正式评分仍以交替多次中位数为准。

### 沙箱无条件拒绝降级

评分和探索脚本要求 macOS sandbox-exec；没有该命令或运行平台不是 macOS 时直接报错，不退化为仅 AST 检查。节点进程不能联网、读取 hidden/ 或其他任务、写出自己的节点目录。

### 当前范围

提速题已扩到 12 道：练习 5、验证 2、考试 5，其中 spd-already-vectorized 是止损埋坑题。四类基础算法和八类扩展算法均通过真实 macOS 沙箱 dev 评分；正确/错误候选、参考解、holdout 门禁、隐藏数据及用户 HOME 读取隔离均有回归测试。

每道提速题的原始实现和隐藏参考候选在三个固定种子上各做 dev 测速，以原始加速比的中位数写入 tasks/bank.json 的 baseline_raw 和 ceiling_raw。逐种子数据、平台与时间见 docs/B-提速校准.json。其中止损题原始与参考代码相同，实测参考分可能因计时噪声略低于基线；不得人为抬高 ceiling。

正式 naive_cost、由其决定的 difficulty、任务 budget.money 仍为空：缺正式模型凭据、模型定价和本题实跑成本。这些字段不能用 TEST 价目或速度代填；题库保持 frozen=false，正式分层评估不能宣称完成。

### 表格公开题的真实拆分和失败记录

新增的 UCI Adult、Mushroom、Online Shoppers、WDBC、Cleveland Heart 题包使用固定种子及特征向量分组的 60/20/20 分层拆分，并以沙箱 LightGBM 与 AutoGluon medium_quality、180 秒上限在 dev 实测。来源归档哈希、特征处理、拆分统计及参考测量逐题保存在 SOURCE.md、split_manifest.json、hidden/reference_measurements.json。holdout 只生成，不用于校准。

UCI Spambase 与 Wine Quality 也保留独立题包和测量记录，但其 AutoGluon dev AUC 低于 LightGBM；归一化 gap 无效，因此 metric 未填，也不纳入 bank-v1。Mushroom 的初版八特征在 dev 达到满分，四特征定义受这一反馈影响，故只可作为练习题，不能用作验证或考试。

### 真实沙箱的 TEST-only batch 烟测

tests/test_real_batch_e2e.py 在临时项目选一题练习、一题验证、一题考试，四配置各三遍，调用真实 A 主循环、B macOS 沙箱评分及 C 的 batch、四桶账本、报价、冻结库、准入、单次 holdout、run_report、figures。2026-11-14 本机实跑 30 个任务，测试 56.5 秒通过。角色响应和候选代码是明确的 ScriptedRoles/已知参考解，token 用量与价格是 TEST 假数据；只能证明管线接通，不能解释为自主研究效果或真实花费。

### 表格题分层、泄漏诱饵与不稳定 gap

表格正式 bank 候选为 12 道实测正 gap：练习 Adult、Mushroom、Online Shoppers、Phishing Websites、German Credit；验证 WDBC、Magic Gamma；考试 Bank Marketing、Cleveland Heart、HTRU2、Default Credit、Australian Credit。每题 baseline/ceiling 是固定 dev split 上合法 LightGBM/AutoGluon 的真实 AUC，明细保存在各题 hidden/reference_measurements.json。EEG Eye State、QSAR Biodegradation、Spambase、Wine Quality 的参考 gap 非正，保留失败记录，不加入 bank。

Online Shoppers 的正 gap 仅 0.000133，Phishing Websites 仅 0.000011；单次 dev 差异过小，不把它们作为考试上限证据，只放练习。Mushroom 的特征选择受前期 dev 反馈影响，也只放练习。Bank 的 train.csv 保留通话后才能知道的 duration，任务明确禁用且推断 rows 不含此列；可见训练集上仅 duration 的 AUC 为 0.819829，说明诱饵有预测力。合法 LightGBM 与 AutoGluon 均先丢弃 duration，重校准 dev AUC 仍为 0.798154 和 0.802219；误用 duration 的候选在真实沙箱 dev 失败。Bank 因此是有实证的考试泄漏陷阱，考试 holdout 仍未打开。

表格正式 score 现在对连续三个随机种子各做一次沙箱评估，以归一化分数中位数为 score、总体标准差为 spread。raw_score 的单次 dev 调用仍供题库校准使用。此改动避免把未测量的波动误报为零；若候选有非确定性，评分成本和时间会相应增加。
