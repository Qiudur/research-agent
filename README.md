# 自进化 Research Agent 系统

让 Research Agent 研究自己。

这是按《分工文档》实现的可审计研究系统：A 的规划、写解、实验树、预算和能力库，B 的两类任务与隔离评分，以及 C 的账本、阶段编排、自研究和报告，均已通过工程回归与真实沙箱联调。bank-v1 已根据真实 dev 成本校准冻结。首次正式内部 token 练习批次未产生有效候选；修复后的 v2 正在独立工作区运行，验证与考试尚未启动。正式账本的 TOKENS 是资源计量，不能视作现金价格或官方比赛成绩；已核验结果和待办见 [内部 token 正式研究报告](docs/内部Token正式研究报告.md)。

路演使用自包含的 [仓库路演汇报 HTML](docs/路演汇报.html)。GitHub Actions 每小时第 17 分钟检查已提交的仓库更新；只有源码或文档变化时才重生成并提交报告。可手动运行 python3 scripts/generate_roadshow_report.py；页面支持方向键翻页和打印。报告中的实测数字从已提交的证据文档提取，正式结果以原始审计记录为准。

下一阶段候选方向见 [自进化云端路由实验方案](docs/自进化云端路由实验.md)：让 Qwen API 充当“本地模型替身”，在练习题上研究何时值得升级调用 Claude API 或 OpenAI API 模型。多模型路由及对应真实云端实验尚未实现，不能把现有 Qwen 试点算作该结果。

## 已交付

- B 题库有 12 道提速题和 12 道表格二分类题，各按练习 5、验证 2、考试 5 分层；速度题有止损陷阱，Bank Marketing 考试题有通话时长泄漏诱饵。原始/参考代码、LightGBM/AutoGluon 基线与上限均在 dev 实测，失败或弱差距的候选题保留测量但不冒充考试题。随机正确性、沙箱文件/网络隔离和新种子评分有回归测试。
- C 的 execute_batch 运行四种配置各三遍：每遍空库练习；验证集只准入分数不降且平均花费更低的研究版；验证与考试对同一题库版本各设一次性尝试标记。正式模式核验题库、价目、预算、Git 提交与洁净工作区，并拒绝 TEST 注入。
- C 的自动自研究只看练习题，受限生成设置和提示词附录，逐候选三遍评分。任务花费和提案模型调用的四桶 token（含失败调用）分别入账，所有候选和源文件有哈希；选中配置仍须通过一次验证准入。
- C 可在同一阶段并行执行独立运行：batch 的 --workers 接受 1–8，自研究的 --workers 接受 1–3，默认均为 1。各运行内部任务仍按原顺序推进；练习、验证、考试之间及自研究候选轮次之间保留屏障。模型调用可以重叠，真实沙箱执行和评分串行，能力库设置用 ContextVar 隔离线程。一次真实模型 dev 基准观察到较短墙钟，稳定加速和正式效果仍需复测；详见 [真实模型并发基准](docs/真实模型并发基准.md)。不同 campaign 或队友在同一机器上没有跨进程评分串行保证；表格评分和提速脚本会争用 CPU、影响计时，正式计时应隔离机器或集中评分。
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

截至 2026-11-14，修复提交 bf9de62 的主回归 203 项通过；技能层 6 项、看板 7 项也已通过。24 道入库题的原始方案逐题通过真实沙箱 dev 评分，证据见 docs/B-全题dev烟测.json。wheel 已构建并从外部工作目录实测评分。真实 B 沙箱加 TEST 价目、脚本角色的三阶段四线端到端测试在 tests/test_real_batch_e2e.py；它跑 30 个 A 主循环任务和每任务一次 holdout，明确不构成真实模型效果。A 的真实计时和跨运行经验试验见 pilot/README.md。每个 TEST 运行的 manifest.phase 均为 dev，不会冒充正式考试。

## 真实模型 dev 试点

选择可本地部署的 qwen3.5-27b，经授权网关进行真实模型调用。每题自定 40,000 个四桶合计 token 预算，常规扩展阈值 36,000，当前每次调用前预留 12,000（早期试验为 5,000），墙钟 900 秒。价目表 config/prices_token_pilot.json 的币种为 TOKENS、四桶权重均为 1、test_only=true；它只计量资源，不是现金价格。运行脚本 scripts/run_token_pilot.py 将练习题复制进被 Git 忽略的 runs 目录，不修改原题，并留下 manifest、账本、实验树、单次 holdout、回放上下文和报告。密钥仅通过进程环境临时注入。实验配置、所有尝试和负结果见 docs/真实模型试点.md。

    .venv/bin/python -m scripts.run_token_pilot
    .venv/bin/python -m scripts.run_token_pilot --task tab-mushroom-edibility --agent-config config/agent_token_pilot_v2.json
    .venv/bin/python -m scripts.run_token_pilot --task tab-german-credit --agent-config config/naive_token_pilot.json --config-label naive
    .venv/bin/python -m scripts.run_token_self_research
    .venv/bin/python -m scripts.run_token_self_research --default-only

不带 --default-only 的 run_token_self_research 命令在被 Git 忽略的独立项目副本上，只取一道提速和一道表格练习题，各候选重复三次，并由真实模型提出至多一轮受限调参。C 报告标记 TEST ONLY，但此试验的 TEST 单位严格等于网关报告的四桶 token 总和，不能解释为现金成本或正式验证/考试。输入哈希、提案账本和所有运行留在 `runs/token-self-research-*`。

真实模型自研究已跑通两轮、各 6 次运行和 12 次任务。首轮候选分数下降且 token 花费上升；第二轮候选平均每遍省 34.5% token，但平均分下降约 2.2%，一个提速节点重放超容差。均未进入正式验证/考试；配置、结果和审计见 `docs/真实模型自研究试验.md`。

第二轮候选又在未参与提案的距离矩阵和 Adult Income 两道 practice 题上做了三遍真实模型迁移复评：平均分提高但 token 花费增加，十二个最佳节点重放全通过。Adult Income 的微小 dev 参考差使 holdout 归一化均分为负，暴露了 C 原先负分/成本排序的缺陷；修复后非正分先按质量、再按成本排序。原始报告保留，详见 `docs/真实模型迁移复评.md`。

## 内部 token 正式研究

可在正式命令中给 auto-research 增加 --workers 3、给 batch 增加 --workers 8；这只调整独立重复的并发数，不改变阶段顺序或一次性验证/考试规则。--default-only 是真实模型三遍默认配置的并发基准入口，不运行候选提案；已用该入口完成一次三路真实模型 dev 基准：460.778 秒，对照旧串行约 1,318.138 秒；这是非受控观察，不是稳定倍数或正式比赛效果，详见 [真实模型并发基准](docs/真实模型并发基准.md)。

bank-v1 已按两组真实 dev 校准冻结：24 题共 72 次尝试，69 次正常完成、3 次提前停止；四桶合计 1,573,026 token，校准未打开 holdout。每题自选 40,000 token 预算，按预先固定的规则写入 naive_cost 和 difficulty。提速与表格明细分别见 [提速朴素成本校准](docs/提速朴素成本校准.md)、[表格朴素成本校准](docs/表格朴素成本校准.md)；完整只读审计与提议保存在 Git 忽略的私有 `runs/naive-cost-audit-20261114-1041-agent/`。此账本计量 token 资源，不是现金价格或官方比赛成绩；正式运行步骤见 [内部 token 正式运行手册](docs/内部Token正式运行手册.md)。

成员甲另一台机器交付的表格题独立复测已核验并单列为[队友甲表格成本独立复核](docs/队友甲表格成本独立复核.md)，附逐次[JSON 摘要](docs/队友甲表格成本独立复核.json)。因运行代码与网关不同，且一笔失败调用未返回用量，该组数字没有覆盖已冻结题库。

正式操作在独立、干净的 worktree 中按 [内部 token 正式运行手册](docs/内部Token正式运行手册.md) 执行；该手册给出 qwen3.5-27b、授权网关、`runs/` 目录、`--workers 3/8`、哈希预检及一次性考试的完整命令。当前 v2 practice 已在独立工作区运行，勿从主工作区另起同一批次。

正式批次读取并核验自研究生成的最佳调参配置；即使默认版在练习题上总体更好，也让最佳生成版接受一次验证准入，拒绝后只报告其余三条正式考试线。若所有提案均无效，selection 为空，必须先停下检查提案失败记录，不启动验证/考试。图表会交叉验证研究版与选中配置的哈希。正式验证/考试一旦开始就不会自动重跑，结果怎样就如实报告。详细参数见 evaluation/README.md。

正式成绩出来后，现场演示、录屏和答辩按 docs/现场演示与答辩.md 核对证据并填入数字；该提纲当前不含任何伪造的正式结果。

运行时可查看只读看板：

    python3 -m dashboard.serve --run-dir runs/<run_id> --port 8765

## 当前限制

bank-v1 已根据两组真实 dev 证据冻结：24 题、72 次尝试中 69 次正常完成、3 次提前停止，合计消耗 1,573,026 token，未打开校准 holdout；失败消耗已计入朴素成本。首次正式 practice 批次只完成默认版三遍，两轮提案均被拒；修复后的 v2 practice 正在运行。一次性验证和考试尚未执行，因此暂无正式四线效果或回本点结论。当前 TOKENS 账本仅用于自选预算的内部研究；网关现金价目未核实，不能宣称现金节省或官方比赛成绩。网关对 max_output_tokens 的行为不能保证推理 token 绝对上限；每次调用预留预算，最终仍按四桶账本核查实际消耗。正式录屏与答辩须引用考试后的审计证据。

## 冻结后下一步

1. 等待独立工作区正在进行的 v2 十题 practice 自研究，并审计三遍分数、token、候选配置与提示词哈希；操作顺序见[内部 token 正式运行手册](docs/内部Token正式运行手册.md)。
2. 仅在练习阶段选定候选后做只读预检，启动同一 bank-v1 的唯一一次三遍验证准入。
3. 按准入结果执行三线或四线考试，核对每题预算、回放和一次性标记，再生成图表与报告。
