[English](#english) · [中文](#中文)

<a id="english"></a>

# Run dashboard (read-only)

Generate a single HTML page from `run/1`, `ledger/1`, `node/1` and `libchange/1` files for offline viewing. Input files in the run directory are not modified. The page has no network dependencies and does not read `hidden/`, candidate source code or prompt bodies.

```bash
python3 -m dashboard.export \
  --run-dir runs/<run_id> \
  --output /tmp/research-agent-dashboard.html
```

If the library lives outside the run directory, add `--library-root <path>`. To export the completed historical pilot:

```bash
python3 -m dashboard.export \
  --run-dir runs/pilot-20261113-154027-6770 \
  --output /tmp/research-agent-pilot-dashboard.html
```

The dashboard uses the last complete experiment-tree snapshot for each `node_id`. It displays the experiment tree, development-score trend, budget/spending, model tokens and cache usage, three timing categories, quotes and task-end reconciliation, library status/changes and ledger events. Monetary amounts are shown only when the price-file SHA-256 matches the manifest, all four price categories exist for the model, and the ledger provides valid spending evidence; otherwise they are “unknown.” Test prices are labeled as tests. Historical pilots without a manifest can display their trees, but the local control budget `1.0` is not presented as real money.

Read-only verification:

```bash
python3 -m unittest discover -s dashboard/tests -v
```

A live, local, read-only dashboard can refresh every second. Each request rereads the ledger and experiment tree without writing to the run directory:

```bash
python3 -m dashboard.serve --run-dir runs/<run_id> --port 8765
```

To compare two existing runs side by side (empty library on the left, trained library on the right):

```bash
python3 -m dashboard.serve --run-dir runs/<empty_library_run> --other-run-dir runs/<trained_library_run> --port 8765
```

Open the local address printed by the server in a browser. The service listens only on localhost; the side-by-side view is a live demonstration and is not written into formal research reports. A screen recording must still be made from an actual demonstration run.

---

<a id="中文"></a>

# 运行看板（只读）

从 `run/1`、`ledger/1`、`node/1` 和 `libchange/1` 文件生成可离线打开的单个 HTML 页面。输入运行目录中的文件不会被修改，页面没有网络依赖，也不读取 `hidden/`、候选源码或提示词正文。

```bash
python3 -m dashboard.export \
  --run-dir runs/<run_id> \
  --output /tmp/research-agent-dashboard.html
```

若能力库不在运行目录下，可加 `--library-root <path>`。导出已完成的旧 pilot：

```bash
python3 -m dashboard.export \
  --run-dir runs/pilot-20261113-154027-6770 \
  --output /tmp/research-agent-pilot-dashboard.html
```

看板按 `node_id` 取实验树最后一份完整快照，显示实验树、dev 走势、预算/花费、模型 token 与缓存、耗时三块、报价和任务结束对账、能力库状态/变更与账本事件。金额只有在 manifest 指向的价格文件 SHA256 匹配、该模型四类价格完整，且账本存在有效花费证据时才显示；否则为“未知”。测试价目会标为测试。没有 manifest 的旧 pilot 可以查看实验树，但不会把本地控制预算 `1.0` 当作真实钱。

只读校验：

```bash
python3 -m unittest discover -s dashboard/tests -v
```

运行中的只读看板可在本机每秒刷新一次；所有请求重新读取账本与实验树，运行目录不会被写入：

    python3 -m dashboard.serve --run-dir runs/<run_id> --port 8765

用两个已存在的运行分屏展示（左侧空库、右侧练过的库）：

    python3 -m dashboard.serve --run-dir runs/<empty_library_run> --other-run-dir runs/<trained_library_run> --port 8765

浏览器打开本机显示的地址。服务只监听 localhost；分屏是现场演示，不会写入正式研究报告。录屏仍需在实际演示运行后制作。
