# 内部 token 研究：冻结到一次性考试

本手册只适用于可本地部署的 qwen3.5-27b 经授权网关调用、每题自选 40,000 个 token、四桶各 1 单位/token 的研究。它不是本机推理、现金价格或官方比赛成绩。仅总指挥在同一台评分 Mac 上执行校准汇总、冻结、验证和考试；其他 agent 只做隔离的 practice 或静态审计。同机不要并跑其他计时评分。

## 1. 校准审计，先于任何题库修改

在未冻结且干净的仓库根目录，确认提速与表格两个完整 campaign 已交付：各 12 题、每题三遍，包含 protocol.json、summary.json、project/runs 全部账本与封存结果，不能只有汇总。下面两个变量必须指向完整 campaign 目录，不能指向 summary.json；输出目录必须是尚不存在的 runs/ 子目录。

    PY="$PWD/.venv/bin/python"
    SPEEDUP_CAMPAIGN='/absolute/path/to/speedup-campaign'
    TABULAR_CAMPAIGN='/absolute/path/to/tabular-campaign'
    AUDIT_DIR="$PWD/runs/naive-cost-audit-$(date -u +%Y%m%d-%H%M%S)"
    git status --short
    "$PY" -m scripts.aggregate_naive_costs --project-root "$PWD" --speedup-campaign "$SPEEDUP_CAMPAIGN" --tabular-campaign "$TABULAR_CAMPAIGN" --output-dir "$AUDIT_DIR"

仅当命令退出码为 0、输出 ready=true 且 audit.json 无 blocked 时继续。汇总器核验源/副本 SHA、四桶账本、停止原因、dev 封存与零 holdout；少于两次正常完成的题必须重新做一轮有明确原因的新 campaign，不能删掉失败样本。提议在 bank_proposal.json，成本是观察 token，不是货币。

## 2. 检查、冻结、提交

冻结工具会现场重跑两个 campaign 的审计，并检查提议与当前 bank 的 SHA、24 个唯一题目、每场景 4/4/4 难度、qwen3.5-27b 与 40,000 token 预算；只会填 bank 的 naive_cost/difficulty/frozen 和各 task.json 的 budget.money。先预检，再实际应用；中途不要改动题包、价格或代码。

    "$PY" -m scripts.freeze_internal_bank --project-root "$PWD" --proposal "$AUDIT_DIR/bank_proposal.json" --speedup-campaign "$SPEEDUP_CAMPAIGN" --tabular-campaign "$TABULAR_CAMPAIGN" --check-only
    "$PY" -m scripts.freeze_internal_bank --project-root "$PWD" --proposal "$AUDIT_DIR/bank_proposal.json" --speedup-campaign "$SPEEDUP_CAMPAIGN" --tabular-campaign "$TABULAR_CAMPAIGN"
    git diff --check
    git diff --stat
    git add tasks/bank.json tasks/spd-*/task.json tasks/tab-*/task.json
    git commit -m 'Freeze measured internal token bank'
    CODE_SHA="$(git rev-parse HEAD)"
    git status --short
    FORMAL_ROOT="../research-agent-formal-$(date -u +%Y%m%d-%H%M%S)"
    git worktree add --detach "$FORMAL_ROOT" "$CODE_SHA"
    cd "$FORMAL_ROOT"
    test "$(git rev-parse HEAD)" = "$CODE_SHA"
    test -z "$(git status --porcelain)"

预期只有 bank.json 加 24 个 task.json 发生数据变动；若不是，先核对差异。正式运行必须使用与干净 Git HEAD 相同的完整 40 位 CODE_SHA。建议在该提交上建立专用、干净的 formal worktree，以免队友在主工作区的编辑打断长跑。Python 虚拟环境路径 PY 在切换工作区前已保存为绝对路径。

## 3. 安全注入与预检

在 zsh 交互终端，可用不回显的 read 将密钥临时放入当前进程环境；自动化时等价地从可信密钥源通过无回显 stdin 注入子进程。不要把密钥写入命令参数、shell history、.env、仓库、日志或报告；也不要用 env、set、调试回显打印环境。授权网关地址是非秘密，但须精确匹配。

    umask 077
    export MODEL='qwen3.5-27b'
    export OPENAI_BASE_URL='https://ai-gateway.ailab.jiuan.com/v1'
    read -rs 'OPENAI_API_KEY?Gateway API key (hidden): '
    print
    export OPENAI_API_KEY
    "$PY" scripts/preflight.py --model "$MODEL"

缺少或错误的 OPENAI_BASE_URL、缺密钥、题库/价格不合规、沙箱不可用或工作区不洁净，都必须在一次性验证前解决。正式 TOKENS 路径逐次限制输出至多 2,048 token，请求前预留题目预算的 30%（40,000 对应 12,000）；SDK 自动重试关闭。预留不是最终硬上界，完跑后仍要审计逐题实耗。

## 4. 只在十道 practice 题做自研究

以下只使用冻结 bank 中的全部 practice 题；三遍同候选并发，候选轮次顺序不变。先用一轮快速探路；若候选无效或质量不稳，只能在新的 study_dir 继续 practice 研究，不能用 validation/exam 调参。study_dir 必须此前不存在。

    AUTO_STUDY="$PWD/runs/auto-internal-$(date -u +%Y%m%d-%H%M%S)"
    "$PY" -m evaluation.cli auto-research --project-root "$PWD" --model "$MODEL" --prices config/prices.json --code-commit "$CODE_SHA" --default-config config/agent.json --study-dir "$AUTO_STUDY" --rounds 1 --max-output-tokens 2048 --workers 3

读取 auto_self_research_report.json 的 selection、overall_best_on_practice、三遍分数/token 和所有 proposal_attempts。selection 为 null 时停在练习阶段；不要消耗一次性验证。若有候选，先验证选中配置和四角色提示词 SHA。此命令仅输出配置路径，不输出密钥：

    TUNED_CONFIG=$("$PY" - "$AUTO_STUDY/auto_self_research_report.json" <<'PY'
    import hashlib, json, sys
    from pathlib import Path
    from evaluation.self_research import verify_auto_selected
    report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    path = verify_auto_selected(report)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['selection']['config_sha256']
    print(path)
    PY
    )

选中候选即使在 practice 不如默认版，也不能改写原报告；先决定是否继续新的 practice-only 轮次。进入验证前不再改代码、题库、配置或提示词；若必须改代码，则以新提交重新跑自研究，不能混用旧证据。

## 5. 一次性验证与考试

先以选中配置做只读预检，确认 ready=true、模型/价格/提交 SHA 正确、同一 bank_version 没有 validation/exam attempt 标记。此后 batch --exam 是一次性操作：练习四线各三遍、验证只比较两条练过的库线各三遍，按配对规则准入；考试按准入结果执行四线或三线各三遍。八路仅让独立模型请求重叠，真实沙箱评分仍串行，尚不能承诺八路的稳定提速倍数。

    "$PY" -m evaluation.cli preflight --project-root "$PWD" --model "$MODEL" --prices config/prices.json --code-commit "$CODE_SHA" --default-config config/agent.json --naive-config config/naive.json --tuned-config "$TUNED_CONFIG" --exam
    BATCH_DIR="$PWD/runs/batch-internal-$(date -u +%Y%m%d-%H%M%S)"
    "$PY" -m evaluation.cli batch --project-root "$PWD" --model "$MODEL" --prices config/prices.json --code-commit "$CODE_SHA" --default-config config/agent.json --naive-config config/naive.json --tuned-config "$TUNED_CONFIG" --batch-dir "$BATCH_DIR" --exam --workers 8

验证若拒绝候选，程序会跳过研究版考试，仍保留朴素版、默认空库、默认练过库三线考试；报告必须说明拒绝原因，不能计算研究版回本点。任何阶段失败都保留原目录和全局 attempt 标记，不删除、不在同一 bank_version 重考。修复后如需新考试，必须按协议重建新版本并重新冻结。

## 6. 审计与图表

batch_report.json、各 run 的 report.json、manifest、账本、回放和一次性标记是事实来源。先检查完整性与每题不超过 40,000 token；超支如实记为协议违例，不抹平或用零覆盖。图表要引用同一批次与同一自研究 selection；figures 会重算准入并核对配置/提示词及研究成本。

    "$PY" - "$BATCH_DIR/batch_report.json" <<'PY'
    import json, sys
    from pathlib import Path
    batch = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    over = [(phase, run['run_id'], task['task_id'], task['total_cost'])
            for phase, runs in batch['reports'].items() for run in runs
            for task in run['tasks'] if task['total_cost'] > 40000]
    print({'admission': batch['admission'], 'over_budget': over,
           'tuned_exam_skipped': batch['tuned_exam_skipped']})
    raise SystemExit(bool(over))
    PY
    "$PY" -m evaluation.cli figures --batch-dir "$BATCH_DIR" --auto-study-dir "$AUTO_STUDY" --output "$BATCH_DIR/figures.html"

最后清除本交互终端中的秘密变量：

    unset OPENAI_API_KEY

最终报告统一称“自选 token 预算的内部研究”。展示真实模型分数、四桶 token、失败与负结果、验证准入及三/四线考试；不得称现金节省或官方比赛成绩。
