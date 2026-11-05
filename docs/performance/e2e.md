# 端到端训练基准

端到端基准覆盖模型层全链路：数据加载、forward、backward、sparse update 和 dense optimizer，产物是 samples/s 与 ms/step。

入口：

```bash
python3 -m tools.benchmarks.e2e.custom.cli --help
```

运行前先按[快速运行](../quickrun.md)准备好 PS 和训练进程；机器与拓扑要求见 [分布式训练](../dev/distributed.md)。

结果口径：

- 报告 samples/s 与 p50 ms/step，必要时拆出 sparse / lookup / dense 分项；
- 历史结果与环境说明见[性能报告](reports.md)；
- 三层指标的含义与不可比性见[测试范围与指标](methodology.md)。
