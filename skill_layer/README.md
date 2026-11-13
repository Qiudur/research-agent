# 技能层（M4）

公开接口位于 `library.skills`，不修改现有 `Library` 的存储 schema：

```python
from library.skills import promote_candidates, promote_to_skill, load_skill_verified, demote_skill

groups = promote_candidates(lib)                 # 例如 [["exp-0001"]]
skill_id = promote_to_skill(
    lib, groups[0],
    body="# 操作步骤\n...",
    summary="一句话简介",
)
text = load_skill_verified(lib, skill_id)        # 只有选中技能后才加载全文
```

晋升必须同时满足 `config/agent.json` 的 `skill_min_credibility` 与 `skill_min_tasks`。这里的“在不同题上有效”不是只数 `sources` 或 `effect.wins`：每次胜出须在能力库变更记录中确有 `record_use` 的胜出增量，并且对应 `runs/<run_id>/tasks/<task_id>/tree.jsonl` 的节点最终为 `accepted`、引用该经验、无作弊标记、分数提升超过波动。至少两道不同 `task_id` 才过门槛；同一道题重复两遍不算。缺少实验树或历史能力库快照的旧记录不够格。

`promote_candidates` 只列单条合格经验，避免未经验证地把不同条件的经验合并。`promote_to_skill` 需要人明确提供正文和简介；不会从固定脚本或模型输出中擅自生成。晋升通过 `Library.put(op="promote")` 写入版本快照和含源节点的变更证据，正文保存在 `skills/skl-XXXX.md`，入口记录 SHA256。平时 `Library.brief()` 只展示简介，`load_skill_verified()` 才读取并核对全文哈希。

显式降级需要原因和证据。降级的技能不再被检索或加载；之后三次无效使用的退休逻辑仍由 `Library.record_use()` 按既有门槛执行。技能数量超过 `max_skills` 时，淘汰可信度最低的可用技能。冻结库会拒绝晋升或降级。

```bash
python3 -m unittest discover -s skill_layer/tests -v
```

限制：旧 pilot 的经验只有一个任务上的一次使用，因此不会晋升。证据验证依赖本地实验树和能力库版本文件的完整性；正式报告仍须配合 manifest、回放封存和 B/C 的验证集准入。此模块不证明技能可带来跨题提升，也不包含 M5 的工具代码晋升。
