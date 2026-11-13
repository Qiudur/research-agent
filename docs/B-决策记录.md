# B 线决策记录

## 2026-11-14

### 正式任务预算保持未定

tasks/spd-distance-matrix/task.json 的 budget.money 保持 null。本机测试价格不是比赛价格，不能写进正式任务包或题库原始分。

### 端到端联调用临时题包覆盖预算

C 的联调可设置 RA_TASKS_ROOT 指向临时复制的题包，并由 evaluation.orchestrate.execute_task(..., test_only_budget=...) 注入 TEST 预算。临时控制文件会写入运行目录；正式题包不被改写。

### 提速参考解采用 NumPy 分块实现

首道题的隐藏参考解使用分块 NumPy 距离计算并返回 Python list[list[float]]，避免原 pilot 的对称循环参考解过慢而造成参考上限失真。实测原始双循环约 0.27 秒、NumPy 参考约 0.056 秒（2400 点，单次本机测量）；正式评分仍以交替多次中位数为准。

### 沙箱无条件拒绝降级

评分和探索脚本要求 macOS sandbox-exec；没有该命令或运行平台不是 macOS 时直接报错，不退化为仅 AST 检查。节点进程不能联网、读取 hidden/ 或其他任务、写出自己的节点目录。

### 当前范围

当前题库先提供一道可真实评分的提速练习题。表格题依赖和正式 LightGBM/AutoGluon 分数未在本机实测前不写入题库，避免把替代基线冒称正式基线。
