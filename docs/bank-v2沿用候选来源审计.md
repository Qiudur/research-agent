# bank-v2 沿用 bank-v1 自研究候选：来源封存与审计

bank-v1 的正式考试已经写下一次性标记并中止，不能重跑或补齐。bank-v2 使用全新的考试题，但允许沿用 bank-v1 已保存自研究所选出的 `auto-02` 候选。沿用是来源声明，不是一次新的 bank-v2 自研究。

在 bank-v2 的独立工作区中，先复制原候选的 `agent.json` 和四个角色提示词，保持配置所写的相对路径与原文件字节一致；默认配置及其四提示词也须与原自研究时的文件一致。练习、验证题的 `bank.json` 条目和公开任务文件必须与 bank-v1 一致，考试题 ID 必须与 bank-v1 考试全不重叠。题库还必须满足 `_bank_tasks(..., formal=True)` 的双场景真实 trap、分层、实测基线与预算检查。

完成题库和候选复制后，在 bank-v2 工作区生成封存清单：

```bash
python3 -m scripts.seal_bank_v2_carry_forward \
  --project-root . \
  --source-root /Users/qizetao/Desktop/research-agent-formal-20261114-v2 \
  --source-study-dir runs/auto-internal-20261114-v2 \
  --source-code-commit bf9de62c52a16893e99d5dd5d93228bdd19a8532 \
  --carried-candidate runs/auto-internal-20261114-v2/candidates/auto-02/agent.json \
  --carried-default config/agent.json
```

脚本只读复核原 `auto_plan.json`、自研究报告、提案响应与账本、九个练习 run 的逐题账本和 holdout 分数，然后生成 `config/bank_v2_carry_forward.json`。它不会调用模型、评分器或沙箱。清单记录原工作区、代码提交、题库及价目表 SHA、原研究三份核心文件 SHA，以及复制后的候选和默认配置及四提示词 SHA。生成失败时清单会被删除，不会留下半成品。

**正式运行前必须先提交清单和题库，确认 Git 工作树干净。**`evaluation.cli` 预检会重新审计来源；`execute_batch` 在创建批次目录和一次性阶段标记前再次审计，并将清单 SHA 连同配置、提示词、价目表及其他输入写入 `batch_plan/1.sealed_inputs`。完成后 `scripts.audit_formal_outputs` 会用原 bank-v1 的根目录、版本和提交独立审计旧研究，再以 bank-v2 的提交审计新批次；若清单、来源、公开题文件、候选字节或考试题去重失败，则不宣称完整正式结果。路演生成器要求 `--auto-study-dir` 指向清单中的原 bank-v1 自研究目录。

原研究的 2,094,728 TOKENS 是已发生的历史成本，审计展示其来源，但 bank-v2 回本点保持**不计算**；不得把该旧成本当作 bank-v2 当期投入。全部 TOKENS 是内部资源计量，不是现金价格或官方比赛成绩。
