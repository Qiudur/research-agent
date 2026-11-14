# Research Agent · A/B/C 研究系统

这是按《分工文档》实现的可审计研究系统：A 负责规划、写解、实验树、预算和能力库；B 提供两类任务与隔离评分；C 负责价格账本、练习/验证/考试编排、自研究、报告和看板。真实模型的 dev token 试点已经运行，正式四线比较尚未执行。TEST 价目、脚本角色和已知参考候选只用于机制联调，不能当作自主提升或真实省钱结论。

## 已交付

- B 题库有 12 道提速题和 12 道表格二分类题，各按练习 5、验证 2、考试 5 分层；速度题有止损陷阱，Bank Marketing 考试题有通话时长泄漏诱饵。原始/参考代码、LightGBM/AutoGluon 基线与上限均在 dev 实测，失败或弱差距的候选题保留测量但不冒充考试题。随机正确性、沙箱文件/网络隔离和新种子评分有回归测试。
- C 的 execute_batch 运行四种配置各三遍：每遍空库练习；验证集只准入分数不降且平均花费更低的研究版；验证与考试对同一题库版本各设一次性尝试标记。正式模式核验题库、价目、预算、Git 提交与洁净工作区，并拒绝 TEST 注入。
- C 的自动自研究只看练习题，受限生成设置和提示词附录，逐候选三遍评分。任务花费和提案模型调用的四桶 token（含失败调用）分别入账，所有候选和源文件有哈希；选中配置仍须通过一次验证准入。
- 报告从账本、实验树和单次 holdout 生成。准入通过时四线图从同一批次的 12 份考试报告绘制分数范围、逐题走势、报价误差和成本；准入未通过时生成三线报告并明确研究版未进考试。回本点包括全部自研究运行及提案费用，未准入时不宣称回本。TEST 数据有显著水印。只读看板可离线导出，也可在 localhost 每秒刷新并分屏展示。
- A 的朴素对照配置使用同一通用提示词、全程高推理和完整实验历史；默认配置按角色分档。能力库支持版本回放、冻结、跨任务证据门槛和技能按需加载。

## 验证

从本仓库根目录运行；真实 B 评分必须在提供 sandbox-exec 的 macOS 上运行。表格依赖已在 Python 3.11 环境核对，并需安装 macOS libomp；请使用同一环境执行测试与正式命令，避免系统 Python 3.13 缺少表格依赖。

    python3.11 -m venv .venv
    .venv/bin/python -m pip install -e '.[llm,tabular,speedup]'
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python -m unittest discover -s skill_layer/tests -v
    .venv/bin/python -m unittest discover -s dashboard/tests -v
    .venv/bin/python scripts/preflight.py --model <正式模型名>

2026-11-14 在 Python 3.11 虚拟环境中，主测试 151 项、技能层 6 项、看板 7 项均通过；24 道入库题的原始方案逐题通过真实沙箱 dev 评分，证据见 docs/B-全题dev烟测.json。wheel 已构建并从外部工作目录实测评分。真实 B 沙箱加 TEST 价目、脚本角色的三阶段四线端到端测试在 tests/test_real_batch_e2e.py；它跑 30 个 A 主循环任务和每任务一次 holdout，明确不构成真实模型效果。A 的真实计时和跨运行经验试验见 pilot/README.md。每个 TEST 运行的 manifest.phase 均为 dev，不会冒充正式考试。

## 真实模型 dev 试点

选择可本地部署的 qwen3.5-27b，经授权网关进行真实模型调用。每题自定 40,000 个四桶合计 token 预算，常规扩展阈值 36,000，当前每次调用前预留 12,000（早期试验为 5,000），墙钟 900 秒。价目表 config/prices_token_pilot.json 的币种为 TOKENS、四桶权重均为 1、test_only=true；它只计量资源，不是现金价格。运行脚本 scripts/run_token_pilot.py 将练习题复制进被 Git 忽略的 runs 目录，不修改原题，并留下 manifest、账本、实验树、单次 holdout、回放上下文和报告。密钥仅通过进程环境临时注入。实验配置、所有尝试和负结果见 docs/真实模型试点.md。

    .venv/bin/python -m scripts.run_token_pilot
    .venv/bin/python -m scripts.run_token_pilot --task tab-mushroom-edibility --agent-config config/agent_token_pilot_v2.json
    .venv/bin/python -m scripts.run_token_pilot --task tab-german-credit --agent-config config/naive_token_pilot.json --config-label naive
    .venv/bin/python -m scripts.run_token_self_research

最后一条命令在被 Git 忽略的独立项目副本上，只取一道提速和一道表格练习题，各候选重复三次，并由真实模型提出至多一轮受限调参。C 报告标记 TEST ONLY，但此试验的 TEST 单位严格等于网关报告的四桶 token 总和，不能解释为现金成本或正式验证/考试。输入哈希、提案账本和所有运行留在 `runs/token-self-research-*`。

真实模型自研究已跑通两轮、各 6 次运行和 12 次任务。首轮候选分数下降且 token 花费上升；第二轮候选平均每遍省 34.5% token，但平均分下降约 2.2%，一个提速节点重放超容差。均未进入正式验证/考试；配置、结果和审计见 `docs/真实模型自研究试验.md`。

## 正式货币运行

若要报告人民币或美元成本，需要核实模型的四桶真实计价并填 config/prices.json；给每道正式题定预算，在朴素配置上实测 naive_cost 后定难度，核对两场景 5/2/5 和陷阱，然后冻结 bank 版本、提交并保持工作区洁净。比赛目前未指定模型或题目预算，所以 token 试点采用自选预算，不能冒充比赛给定价格或正式成绩。不要把试点价目或凭据写成真实货币价格。具体决策和未完成项见 docs/项目交付与决策.md。

    export MODEL='比赛提供的实际模型名'
    # OPENAI_API_KEY 由安全渠道注入环境变量，不写入仓库或命令历史。
    COMMIT=$(git rev-parse HEAD)
    .venv/bin/python -m evaluation.cli auto-research --project-root . --study-dir batches/auto-v1 --model "$MODEL" --prices config/prices.json --code-commit "$COMMIT"
    TUNED_CONFIG=$(.venv/bin/python -c 'import json; from pathlib import Path; from evaluation.self_research import verify_auto_selected; print(verify_auto_selected(json.loads(Path("batches/auto-v1/auto_self_research_report.json").read_text())))')
    .venv/bin/python -m evaluation.cli preflight --project-root . --model "$MODEL" --prices config/prices.json --code-commit "$COMMIT" --tuned-config "$TUNED_CONFIG" --exam
    .venv/bin/python -m evaluation.cli batch --project-root . --batch-dir batches/formal-v1 --model "$MODEL" --prices config/prices.json --code-commit "$COMMIT" --tuned-config "$TUNED_CONFIG" --exam
    .venv/bin/python -m evaluation.cli figures --batch-dir batches/formal-v1 --auto-study-dir batches/auto-v1 --output batches/formal-v1/study.html

上述命令读取并核验自研究生成的最佳调参配置；即使默认版在练习题上总体更好，也让最佳生成版接受一次验证准入，拒绝后只报告其余三条正式考试线。若所有提案均无效，selection 为空，必须先停下检查提案失败记录，不启动验证/考试。图表会交叉验证研究版与选中配置的哈希。正式验证/考试一旦开始就不会自动重跑，结果怎样就如实报告。详细参数见 evaluation/README.md。

正式成绩出来后，现场演示、录屏和答辩按 docs/现场演示与答辩.md 核对证据并填入数字；该提纲当前不含任何伪造的正式结果。

运行时可查看只读看板：

    python3 -m dashboard.serve --run-dir runs/<run_id> --port 8765

## 当前限制

授权令牌已在本轮真实 dev 试点中临时使用，没有写入仓库或运行产物。正式货币价目未确认；bank 的 naive_cost、difficulty 仍待相同模型的朴素对照实测，frozen=false。因此正式自研究、验证、考试及真实货币成本下降数字都尚未完成。网关对 max_output_tokens 的行为不能保证推理 token 的绝对上限；预留闸门只降低超预算风险，账本会记录每次实际消耗。自动自研究没有总体金额上限，但内部每道题仍沿用 A 的正预算和 90% 扩展阈值。正式录屏和答辩中的实测结论须等考试完成后制作。
