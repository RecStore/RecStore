# 命令行与环境变量

常用入口都可以用 `--help` 查看当前参数：

| 入口 | 用途 |
| --- | --- |
| `build/bin/ps_server` | 启动参数服务器，`--config_path` 指定 JSON 配置，`-local_shard_id` 指定分片 |
| `build/bin/petps_server` | RDMA 参数服务器入口，配置与线程数走 flags |
| `build/bin/benchmark_kv_engine` | storage-only 基准，`--engine_class`、`--index_type`、`--value_store_type` 等 |
| `tools/benchmarks/run_benchmark_ps.py` | PS/network 基准 runner |
| `tools/benchmarks/run_kvengine_compare.py` | KVEngine 对比 runner |
| `tools/benchmarks/e2e/custom/cli.py` | 端到端基准入口 |

```bash
./build/bin/ps_server --help
python3 tools/benchmarks/run_benchmark_ps.py --help
```

## 常用编译参数

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `-DCMAKE_BUILD_TYPE=Release` | — | 基准与对外数字只认 Release；Debug 仅用于排查 |
| `-DUSE_PERF_REPORT=ON` | `OFF` | 编译期打开性能埋点（会带来几个百分点的开销） |
| `-DRECSTORE_ENABLE_THIRD_PARTY_ENGINES=ON` | `OFF` | 编入 FASTERKV / HPS 等外部存储引擎 |
| `-DENABLE_CUDA=OFF` | `ON` | 无 GPU 环境做纯 CPU 构建 |
| `-DUSE_BRPC_CPU_PROFILER=ON` | `OFF` | 打开 bRPC 内置 CPU profiler |
| `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` | — | 新版 CMake 兼容旧 `cmake_minimum_required` |

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release -j
```

仓库里 `build/` 与 `build_release/` 是两种常用目录；测试与 runner 默认找 `build/bin/...`，用 `build_release` 时记得让仓库根的 `build -> build_release` 软链保持有效。

## 常用环境变量

| 变量 | 作用 |
| --- | --- |
| `RECSTORE_CONFIG` | 配置文件路径；未设置时从当前目录逐级向上找 `recstore_config.json` |
| `RECSTORE_REPORT_MODE` / `RECSTORE_REPORT_LOCAL_SINK` / `RECSTORE_REPORT_JSONL_PATH` | 性能上报模式与本地落盘（见 [Profiling](../performance/profiling.md)） |
| `RECSTORE_ENABLE_GPU_CACHE` | 打开 GPU cache 路径 |
| `RECSTORE_LOG` / `RECSTORE_LOG_EVERY_MS` | 日志级别与限流间隔 |
| `CPUPROFILE` | 开启 gperftools CPU profiling |

??? note "RDMA 常用环境变量"

    | 变量 | 作用 |
    | --- | --- |
    | `RECSTORE_RDMA_CONTROL_PLANE_HOST` / `RECSTORE_RDMA_CONTROL_PLANE_PORT` | client 连 server 控制面的地址，默认取配置文件里的 server |
    | `RECSTORE_RDMA_GET_RESPONSE_MODE` | `direct_sg` 或 `staging_copy`；按 key 批量大小选择，选错会走到慢路径 |
    | `RECSTORE_RDMA_VALIDATE_ROUTING` | 打开后校验 key 路由，仅用于排查 |
    | `RECSTORE_RDMA_WIRE_TRACE` / `RECSTORE_RDMA_WIRE_TRACE_TAG` | 线上 trace 采集 |
    | `RECSTORE_RDMA_WAIT_TIMEOUT_MS` | 等待完成超时 |
    | `RECSTORE_RDMA_RC_NAMESPACE` | 多进程共享同一套 RC 资源时的命名空间 |
    | `RS_DEMO_RDMA_FABRIC` | 端到端 runner 用的 fabric，格式 `device:port:gid_index:mode`，如 `mlx5_0:1:0:ib` |

    设备与 GID 的默认值和选择规则见 [RDMA 参数](../parameter_server/rdma.md)。

配置字段本身见[配置字段参考](../config.md)；配置文件放在哪、怎么被读到见[配置指南](../usage/configuration.md)。
