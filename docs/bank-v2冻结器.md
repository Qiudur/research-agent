# bank-v2 组合与冻结器

此脚本只处理新考试题的 dev-only 朴素成本校准；不会运行模型、沙箱或 holdout。它从冻结的 bank-v1 读取每场景 5 道 practice、2 道 validation 及其既有 naive_cost，并接入每场景 5 道全新 exam 题的三遍校准。原 bank-v1 文件保持不变。bank-v2 的重复 practice/validation 不算新的独立验证，新的考试比较只能依赖十道未使用 exam 题。

## 输入及审计

- --project-root：含冻结 tasks/bank.json（bank-v1）和新十题文件的工作树。表格候选源 task.json 可以尚未定预算，但目标工作树必须先用独立激活步骤把十道新题的预算设为 40,000 token；脚本只容许这一处源→校准副本覆写。
- --speedup-campaign、--tabular-campaign：各含 protocol.json、已结束的 summary.json、project/、15 次逐次账本和 dev 回放封存。校准源工作树仍须可访问。脚本复用原独立审计器，重算四桶、失败已花 token、身份、每题三次尝试和预算上限，并拒绝任何 holdout 文件或事件。naive_cost 是三次全部用量之和除以正常完成次数；少于两次完成则阻断。
- --target-code-commit：拟冻结的正式提交。每个 campaign 的提交与它比较；若 diff 涉及 harness、agent、编排、计价或校准配置，提案记录 runtime_changed 并置 ready=false，须按新代码重新测量。仅文档和测试差异不会阻断。配置及新题源码还须与 campaign 的封存 SHA 一致。
- 新表格题须有 SOURCE.md、split_manifest.json 和 hidden/bank_v2_reference_measurements.json；task 与校准题库里的 AUC 基线/上限一致，差值至少 0.01。正式冻结另要求五份逐题许可确认，按 SOURCE.md SHA 与提案绑定。脚本不读取 holdout.csv 内容。

## 两阶段命令

先在干净的候选工作区运行：

    python -m scripts.propose_bank_v2 propose --project-root <候选工作区> --speedup-campaign <已完成的提速校准目录> --tabular-campaign <已完成的表格校准目录> --target-code-commit <正式目标代码SHA> --output <新建的proposal.json>

输出 bank_v2_proposal/1，包含 24 题、两个 campaign 的逐次审计摘要、代码差异、SHA 和阻断项。每场景 12 题按 (naive_cost, task_id) 排序，前 4 道 easy、中 4 道 medium、后 4 道 hard。若 ready=false，命令退出 1，但保留提案供复盘，不能冻结。

输入协议 schema 为 naive_token_calibration_protocol/1 与 naive_token_calibration_summary/1；每个 campaign 必须记录 model、bank_version、code_commit、tasks、repetitions、预算/输出上限、source_sha256、copied_sha256，summary 须有 ended_utc 和 15 条逐次 attempts。提案 schema bank_v2_proposal/1 的顶层字段为 ready、blocked、source_bank_v1_sha256、target_code_commit、campaigns、license_evidence、tasks；campaigns 分别带 protocol_sha256、summary_sha256、code_drift、五题的逐次成本和任务源码哈希。

许可确认文件的 schema 为 bank_v2_license_review/1：proposal_sha256 是提案文件的 SHA-256；entries 恰有五项，每项包含 task_id、source_md_sha256 和 redistribution_permitted=true。确认人须先读对应 SOURCE.md 与来源许可。五项均与提案和源文件 SHA 匹配后执行：

    {"schema":"bank_v2_license_review/1","proposal_sha256":"<64位摘要>","entries":[{"task_id":"tab-...","source_md_sha256":"<64位摘要>","redistribution_permitted":true}]}

    python -m scripts.propose_bank_v2 freeze --project-root <候选工作区> --proposal <proposal.json> --confirm-proposal-sha <proposal文件SHA256> --license-review <license-review.json> --output-bank <新建的bank-v2.json>

冻结时重新审计所有输入并比较完整提案，以独占新文件方式写出 bank/1、version=bank-v2、frozen=true 的 24 题 JSON 和回执；拒绝覆盖 bank-v1。新文件须安装到另一个干净工作区的 tasks/bank.json，连同新题一起提交并预检。bank-v1 已曝光考试题不得混入。
回执 schema 为 bank_v2_freeze_receipt/1，含 bank_path、bank_sha256、proposal_sha256、source_bank_v1_sha256、target_code_commit 和 task_count=24。

本脚本尚未对真实 bank-v2 做实测；真实表格 campaign 尚未交付。早期提速校准若早于失败计零代码，差异会写入提案并阻断冻结，不能静默视作同协议。
