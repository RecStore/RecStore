# 参数服务器与传输基准

PS/network 层只测参数服务器与客户端之间的传输，不含模型训练：

- 支持 GRPC、BRPC、RDMA 三种传输；
- 可本机运行，也可用显式 IP 列表跑跨机拓扑；
- 指标是 keys/s 与 p50/p99 延迟，读写分别统计。

入口：

```bash
python3 tools/benchmarks/run_benchmark_ps.py --help
```

RDMA 的设备、GID、QP 与 lane 参数见[RDMA 实现](../parameter_server/rdma.md)；拓扑配置
见[分布式训练](../dev/distributed.md)。三层指标口径见[测试范围与指标](methodology.md)。
