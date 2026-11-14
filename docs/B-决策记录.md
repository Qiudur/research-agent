# B 线决策记录

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
