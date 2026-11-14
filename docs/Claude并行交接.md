# Claude 并行交接（2026-11-14）

## 当前固定边界

- Codex 本窗口独占执行正式工作区 "/Users/qizetao/Desktop/research-agent-formal-20261114-v2" 的一次性 batch；该工作区固定在提交 `bf9de62c52a16893e99d5dd5d93228bdd19a8532`。不要修改该工作区、启动额外模型评分、删除或重跑 bank-v1 的 validation/exam attempt。
- 正式批次目录是该工作区下的 `runs/batch-internal-20261114-v2`；自研究目录是 `runs/auto-internal-20261114-v2`。这两个 `runs/` 目录未进入 Git。不要把凭据、原始运行包或隐藏题数据提交到仓库。
- 本窗口的子 agent 正在独立审计批次并维护 `docs/内部Token正式研究报告.md`。不要编辑这份报告，以免冲突。

## 请 Claude 负责的独立工作

在单独分支/工作树改进 `scripts/generate_roadshow_report.py`：增加可选的正式批次输入，让最终路演页从已审计的 batch/self-research 报告读取实际分数、四桶 token、验证准入与考试线数，并标注同一批次的提交与配置 SHA。批次尚未完成时保留明确的“待真实账本”状态；研究版未通过准入时不显示研究版考试成绩或回本点。保持 TOKENS 非现金、非官方成绩的口径。可复用 `scripts/audit_formal_outputs.py` 的只读校验，避免另写评分逻辑。

仅用合成夹具做单元测试，覆盖准入通过、准入拒绝、缺失/篡改报告三种情况；不要在本机同时运行模型或沙箱评分。交付你的分支名、提交 SHA、测试命令和结果。Codex 会在唯一正式批次完成后核对并合并，再用真实账本重生成展示页。

## 协作方式

请不要直接改共享的 `main`；基于当前 `origin/main` 建独立分支或 worktree。若发现正式批次严重问题，先写简短事实与可复现路径通知用户/本窗口，不要自行重启一次性阶段。
