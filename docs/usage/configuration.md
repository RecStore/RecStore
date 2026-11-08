# 配置指南

RecStore 只用一个 JSON 配置文件。本页回答"配置放哪、怎么被读到、常见场景改什么"；字段含义统一查[配置字段参考](../config.md)，本页不重复字段表。

## 1. 配置文件放在哪

默认文件名 `recstore_config.json`，仓库根目录已经带了一份可直接运行的。读取顺序与 `src/ps/ps_server.cpp`、`tools/config/recstore_config_path.py` 一致：

1. 命令行 `ps_server --config_path <file>`；文件不存在时忽略并继续向下回落（`--brpc_config_path` 仍然可用，但只在 `--config_path` 为空时生效，部分 benchmark runner 还在用它）。
2. 环境变量 `RECSTORE_CONFIG=<file>`；测试 runner 和 `model_zoo/rs_demo` 都走这条。
3. 从当前工作目录逐级向上查找 `recstore_config.json`。
4. 仓库根目录的 `recstore_config.json`。

四处都没有时启动直接失败，报 `recstore_config.json not found`。服务端与客户端必须读同一份配置，`distributed_client` 尤其是这样。

## 2. 三种常见场景

| 场景 | 关键改动 | 参考 |
| --- | --- | --- |
| 单机跑通 | 保持默认：`ps_type=GRPC`、`num_shards=2`、`127.0.0.1` 上 15000/15001 两个端口 | [快速运行](../quickrun.md) |
| 多机多分片 | 两台机器共享同一份 `distributed_client`（`num_shards` / `hash_method` / `servers` 必须一致），`servers` 换成真实 IP，服务端 `host` 改成 `0.0.0.0` 或 LAN IP | [分布式训练](../dev/distributed.md) |
| 换存储后端 | 只改 `cache_ps.base_kv_config` 的 `index` / `value` / `capacity` | [KVEngineComposite](../storage/composite_kvengine.md) |
| RDMA 数据面 | `cache_ps.ps_type` 写 `RDMA`，服务端换成 `petps_server`（`ps_server` 只分发 GRPC/BRPC） | [RDMA 参数](../parameter_server/rdma.md) |

## 3. 顶层块

| 块 | 谁读 | 内容 |
| --- | --- | --- |
| `cache_ps` | 参数服务器、框架客户端 | `ps_type`、`num_threads`、`num_shards`、`servers`、`base_kv_config`（`max_batch_keys_size` 已废弃，仅保留兼容） |
| `distributed_client` | 计算层客户端 | `num_shards`、`hash_method`、`max_keys_per_request`、`servers` |
| `client` | 单机客户端 | 单分片的 `host` / `port` / `shard` |
| `report_API` | 可选 | 性能上报接收端 |
| `hugectr` / `hierkv` | 可选 | 第三方适配层配置 |

## 4. 配置文件、命令行与环境变量的分工

配置文件管"拓扑和存储"；命令行和环境变量管"这一次运行怎么起"：`--local_shard_id` 选本机分片，`RECSTORE_CONFIG` 选配置文件，`CPUPROFILE` 开 profiling。所有入口见[命令行与环境变量](../api/cli.md)。

## 5. 排错

- `Unknown ps_type`：`ps_server` 只接受 `GRPC` / `BRPC`；RDMA 走 `petps_server`，`LOCAL_SHM` 只用于测试与 benchmark。
- 服务端起来了但客户端连不上：先对齐 `cache_ps.servers` 与 `distributed_client.servers` 的 `host`/`port`/`shard`；跨机时服务端 `host` 不能还是 `127.0.0.1`。
- 表建出来读不到值：检查 `capacity` 与 `default_value_size_hint` 是否够用，以及 index/value 组合是否在 [KVEngineComposite](../storage/composite_kvengine.md) 的支持矩阵里。
