# 命令行与环境变量

常用入口都可以用 `--help` 查看当前参数：

| 入口 | 用途 |
| --- | --- |
| `build/bin/ps_server` | 启动参数服务器，`--config_path` 指定 JSON 配置，`-local_shard_id` 指定分片 |
| `build/bin/benchmark_kv_engine` | storage-only 基准，`--engine_class`、`--index_type`、`--value_store_type` 等 |
| `tools/benchmarks/run_benchmark_ps.py` | PS/network 基准 runner |
| `tools/benchmarks/run_kvengine_compare.py` | KVEngine 对比 runner |
| `tools/benchmarks/e2e/custom/cli.py` | 端到端基准入口 |

```bash
./build/bin/ps_server --help
python3 tools/benchmarks/run_benchmark_ps.py --help
```

常用环境变量：

| 变量 | 作用 |
| --- | --- |
| `RECSTORE_CONFIG` | 测试 runner 读取的配置文件路径 |
| `RECSTORE_REPORT_MODE` / `RECSTORE_REPORT_LOCAL_SINK` / `RECSTORE_REPORT_JSONL_PATH` | 性能上报模式与本地落盘（见[Profiling](../performance/profiling.md)） |
| `CPUPROFILE` | 开启 gperftools CPU profiling |

配置字段本身见[配置字段参考](../config.md)；RDMA 环境变量见[RDMA 实现](../parameter_server/rdma.md)。
