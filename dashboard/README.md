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

当前是**导出式看板**：重新运行命令即可刷新数据；不是 Web 服务，也不包含原计划的并排实时演示或录屏。
