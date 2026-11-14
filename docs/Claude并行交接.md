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

## Claude 交付（2026-11-14 22:15）

- 分支 `agent/final-report-20261114`（本机同一仓库，未推送），基于 `c53a767`。路演集成以提交 `5ae0934` 为准；`74159a4` 是汇总器，`c105139`、`77adfc1` 是被 `5ae0934` 取代的中间版本。没有改 `main`、正式工作区和 `docs/内部Token正式研究报告.md`。
- 路演生成器新增可选输入。批次完成后在合并后的仓库根目录运行：

      .venv/bin/python scripts/generate_roadshow_report.py \
        --formal-batch-dir /Users/qizetao/Desktop/research-agent-formal-20261114-v2/runs/batch-internal-20261114-v2 \
        --auto-study-dir /Users/qizetao/Desktop/research-agent-formal-20261114-v2/runs/auto-internal-20261114-v2

  运行时先只读调用 `scripts/audit_formal_outputs.py`，审计通过后再用 `scripts/summarize_formal_batch.py` 汇总，写出 `docs/evidence/formal-batch-roadshow.json` 和 `docs/路演汇报.html`，两份文件都需要提交。页面展示验证准入、考试线数、各配置三遍分数、每遍平均 token、四桶合计，以及代码提交、配置和价目表的 SHA-256。三种情况：
  - 没有 `batch_report.json`：保持“待真实账本”。
  - 审计失败：退出码 1，什么都不写。
  - 研究版未准入：不显示研究版考试成绩和回本点。

  之后不带参数重生成时，只读取已提交、审计通过且自洽的证据文件。
- 汇总器（只读，拒绝把输出写进正式工作区）：

      .venv/bin/python scripts/summarize_formal_batch.py --batch-dir <批次目录> --output-dir <正式工作区之外> --research-cost 2094728

  输出中文 Markdown 和 JSON，内容包括：练习、验证、考试三个阶段逐遍的四桶 token、平均归一化分数、失败调用、停止原因和单题最高花费；验证和考试的逐题原始指标；按三遍规则做的三项考试对比；回本点；小参考差（<0.002）题的诊断视图（不改正式成绩）。批次运行中可以加 `--partial`。
- 测试：`.venv/bin/python -m unittest tests.test_roadshow_report tests.test_summarize_formal_batch tests.test_audit_formal_outputs`，35 项通过。因为同机正在做计时，没有跑全套。

## 下一项并行工作（2026-11-14，正式考试收尾中）

- 路演代码和安全修复已合入主线 `a4b6a7a`。请 Claude 在独立分支更新 `README.md` 与 `docs/项目交付与决策.md` 的过时状态：自研究完成、正式验证三遍拒绝研究版、考试仅运行 naive/default/default_lib 三线；待原始 `batch_report.json` 完成并经独立审计后才填写考试分数和成本。
- 不要修改正式工作区、一次性标记、原始运行包、`docs/内部Token正式研究报告.md` 或路演生成器。不要启动第二次评分。请交付分支、提交 SHA、文档差异和验证命令；Codex 将核对合并。
