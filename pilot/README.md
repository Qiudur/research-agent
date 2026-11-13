# 第一次真实计时试验

本目录只用于先验证 A 能否把一个代码提速候选交给真实的正确性检查和计时器。候选代码由一个子 agent 在只看到 task.json、original.py 与 profile.json 的情况下编写；A 的 TaskRunner 通过固定角色响应调用它。这里没有 live ModelClient 调用，也没有真实 token 成本。

运行方式：

    python3 -m pilot.run --size 2500 --repetitions 7

对已生成的运行使用新种子复测：

    python3 -m pilot.confirm --run-dir runs/<pilot_run_id> --size 2500 --repetitions 7

输入是 2,500 个二维点；原始实现和候选交替计时，各跑 7 次取中位数。检查空输入、单点、重复点、负坐标及新的随机输入。隐藏参考解是独立写出的对称双循环，原定作归一化参照；候选实际比它还快，因此本试验只用原始实现与候选的**原始加速比**作效果结论，不把归一化分数解释成可达上限。

本目录中的 task.json 的 money=1.0 和固定角色预测只是让现有 A 调度器运行的本地控制参数，不是真实价格或预算。评分器在子进程中运行候选，但尚无生产级文件/网络沙箱、防作弊检查或题库隔离。因此它证明的是一次真实计算提速与 A 主循环的接线，不证明多任务泛化、模型成本下降或正式比赛效果。

首次运行的逐次计时、随机种子与源码 SHA256 归档在 [results/first-run.json](results/first-run.json)；完整的运行树、账本和回放仍保存在本地被 Git 忽略的 runs/pilot-20261113-154027-6770/ 中。
