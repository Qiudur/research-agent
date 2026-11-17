[English](#english) · [中文](#中文)

<a id="english"></a>

# Skill layer (M4)

The public interface is in `library.skills`; the existing `Library` storage schema is unchanged:

```python
from library.skills import promote_candidates, promote_to_skill, load_skill_verified, demote_skill

groups = promote_candidates(lib)                 # e.g. [["exp-0001"]]
skill_id = promote_to_skill(
    lib, groups[0],
    body="# Steps\n...",
    summary="One-sentence summary",
)
text = load_skill_verified(lib, skill_id)        # Load full text only after selecting a skill
```

Promotion must meet both `skill_min_credibility` and `skill_min_tasks` in `config/agent.json`. “Effective on different tasks” is not just a count of `sources` or `effect.wins`: each win needs a winning `record_use` increment in the library change history. Its node in `runs/<run_id>/tasks/<task_id>/tree.jsonl` must ultimately be `accepted`, cite the experience, have no cheating flags and show a score gain above measured variation. At least two distinct `task_id` values are required; two repetitions of the same task do not count. Historical records missing an experiment tree or library snapshot are ineligible.

`promote_candidates` lists only individually qualified experiences, avoiding unverified merging of experiences with different conditions. `promote_to_skill` requires a human to supply the body and summary explicitly; it does not generate them autonomously from a fixed script or model output. Promotion uses `Library.put(op="promote")` to write a versioned snapshot and change evidence containing the source nodes. The body is stored in `skills/skl-XXXX.md`, with a SHA-256 in the entry. Normally, `Library.brief()` shows only summaries; `load_skill_verified()` reads the full text and checks its hash.

Explicit demotion requires a reason and evidence. Demoted skills are no longer retrieved or loaded. Retirement after three subsequent unsuccessful uses remains the responsibility of `Library.record_use()` under the existing thresholds. If the number of skills exceeds `max_skills`, the available skill with the lowest credibility is evicted. A frozen library rejects both promotion and demotion.

```bash
python3 -m unittest discover -s skill_layer/tests -v
```

Limitations: the historical pilot's experience has only one use on one task, so it is not promoted. Evidence verification depends on the integrity of local experiment trees and library-version files; formal reports still require manifests, sealed replays and B/C validation admission. This module does not demonstrate gains across tasks and does not include M5 promotion of tool code.

---

<a id="中文"></a>

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
