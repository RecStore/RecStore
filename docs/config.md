# RecStore 配置文档

## 配置文件结构

RecStore 配置采用 JSON 格式，位于根目录。当前仓库默认包含以下顶层配置块：`cache_ps`、`distributed_client`、`client`、`report_API`，以及按需启用的 `hugectr`、`hierkv`。

## 1. cache_ps 配置

`cache_ps` 配置用于参数服务器（Parameter Server）端，定义服务器的运行参数和底层存储引擎。

### 1.1 服务器基础配置

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ps_type` | string | 是 | 参数服务器类型，可选 `"GRPC"` 或 `"BRPC"` |
| `max_batch_keys_size` | integer | 是 | 单次批量请求的最大键数量 |
| `num_threads` | integer | 是 | 服务器工作线程数 |
| `num_shards` | integer | 是 | 分片数量，应等于 `servers` 数组长度 |
| `servers` | array | 是 | 服务器节点配置数组 |

作用位置：通信协议选择见 `src/framework/op.cc`、`src/ps/grpc/grpc_ps_server.cpp` 和 `src/ps/brpc/brpc_ps_server.cpp`。同时也可通过统一入口可执行程序 `src/ps/ps_server.cpp` 按 `ps_type` 自动选择 GRPC 或 bRPC；批量限制与线程数在 `src/ps/grpc/grpc_ps_client.h` 及 `src/storage/kv_engine/base_kv.h` 中生效；分片与 servers 列表用于分布式路由。

### 1.2 servers 数组配置

每个服务器节点包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器监听端口 |
| `shard` | integer | 是 | 分片编号，从 0 开始 |

???+ tip "多机环境"
    默认配置中 `host` 为 `127.0.0.1`，仅允许本地访问。在多机场景下，必须修改为 `0.0.0.0` 或本机对外的局域网 IP (LAN IP)，以允许远程客户端连接。

### 1.3 base_kv_config 配置

`base_kv_config` 定义底层键值存储引擎的配置，由 `src/storage/kv_engine/engine_selector.h` 的 `ResolveEngine` 解析 `engine_type`；具体字段校验在对应引擎构造时完成。

#### 通用必填字段（非 HYBRID 模式）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `engine_type` | string | 否 | 省略时默认为 `KVEngineComposite`；全部引擎见 [basekv.md](./storage/basekv.md) |
| `capacity` | integer | 是 | 预估存储条目数 |
| `index` | object | Composite | 索引配置，见 [composite_kvengine.md](./storage/composite_kvengine.md) |
| `value` | object | Composite | 值存储配置，见 [composite_kvengine.md](./storage/composite_kvengine.md) |

作用位置：字段最终进入 `BaseKVConfig.json_config_`，在 `src/storage/kv_engine/engine_selector.h` 中推导引擎类型，并在 `src/storage/kv_engine/base_kv.h` 及对应引擎实现中用于容量预分配与值大小约束。

???+ note "注意"
    Composite 配置请使用嵌套 `index` / `value`，不要混用已废弃的顶层 `index_type`、`value_type`、`value_memory_management` 等字段。组件类型与 allocator 明细见 [composite_kvengine.md](./storage/composite_kvengine.md)。

#### value 配置结构

| 字段 | 类型 | 适用范围 | 说明 |
|------|------|----------|------|
| `value.type` | string | 全部 | `DRAM_VALUE_STORE`、`SSD_VALUE_STORE` 或 `TIERED_VALUE_STORE` |
| `value.default_value_size_hint` | integer | 全部 | 单条 value 的默认字节数提示，替代旧的顶层 `value_size` |
| `value.path` | string | `DRAM_VALUE_STORE`、`SSD_VALUE_STORE` | DRAM 可为空或 `/dev/shm` 路径；SSD 必须是非空文件路径 |
| `value.dram_allocator` | object | `DRAM_VALUE_STORE`、`TIERED_VALUE_STORE` | DRAM 分配器配置，常用字段为 `type`、`capacity_bytes`；`CONCURRENT_SLAB_MEMORY_POOL` 可加 `size_classes` 覆盖尺寸类；`DRAM_VALUE_STORE` 不允许 `dram_allocator.path` |
| `value.ssd_allocator` | object | `SSD_VALUE_STORE`、`TIERED_VALUE_STORE` | SSD 分配器配置，常用字段为 `type`、`capacity_bytes`、`min_block_size`、`max_block_size`、`io`；`SSD_VALUE_STORE` 的文件路径放在 `value.path`，不放在 `ssd_allocator.path` |
| `value.ssd_allocator.path` | string | `TIERED_VALUE_STORE` | TIERED 模式的 SSD 层文件路径 |

#### 引擎类型选择

`base::ResolveEngine` 读取 `engine_type`，缺省为 `KVEngineComposite`。`KVEngineComposite` 根据 `index.type` 和 `value.type` 组装组件：

- `index.type` 取值：`DRAM_EXTENDIBLE_HASH` / `DRAM_UNORDERED_MAP` / `DRAM_PET_HASH` / `SSD` / `SSD_EXTENDIBLE_HASH`
- `value.type` 取值：`DRAM_VALUE_STORE` / `SSD_VALUE_STORE` / `TIERED_VALUE_STORE`
- 旧字段 `path`、`index.io.file_path`、`value.ssd_allocator.io.file_path` 会被拒绝

仓库根目录 `recstore_config.json` 的**默认 KV 组合**为：

| 字段 | 默认值 |
|------|--------|
| `engine_type` | `KVEngineComposite`（可省略） |
| `index.type` | `DRAM_PET_HASH` |
| `value.type` | `DRAM_VALUE_STORE` |
| `value.dram_allocator.type` | `CONCURRENT_SLAB_MEMORY_POOL` |

引擎推导实现位置：`src/storage/kv_engine/engine_selector.h`

## 2. distributed_client 配置

`distributed_client` 配置用于分布式客户端，实现多分片参数服务器的访问。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `num_shards` | integer | 是 | 分片总数，应与 `cache_ps.num_shards` 一致 |
| `hash_method` | string | 是 | 哈希方法：`"city_hash"` 或 `"simple_mod"` |
| `max_keys_per_request` | integer | 否（默认 500） | 单次请求最大键数量 |
| `servers` | array | 是 | 服务器节点配置，结构同 `cache_ps.servers` |

作用位置：分片数与哈希方法在 `src/ps/brpc/dist_brpc_ps_client.cpp` 与 `src/ps/grpc/dist_grpc_ps_client.cpp` 的 `GetShardId`、`PartitionKeys` 中决定路由；`max_keys_per_request` 限制单分片请求大小；`servers` 列表在 `InitializeClients` 中生成各分片客户端。

## 3. client 配置

`client` 配置用于单节点客户端，直接连接单个参数服务器。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `host` | string | 是 | 服务器主机地址 |
| `port` | integer | 是 | 服务器端口 |
| `shard` | integer | 是 | 目标分片编号 |

作用位置：单节点客户端在 `src/ps/grpc/grpc_ps_client.cpp` 读取以上字段创建 gRPC 通道并标识分片，在 `src/ps/grpc/grpc_ps_client.h` 中保存元数据。

???+ tip "动态覆盖"
    运行时可用 `client.set_ps_config(host, port)` 覆盖配置文件里的 `host` 和 `port`。接口说明见 [计算层接口](./calc/interfaces.md)。

## 4. report_API 配置

`report_API` 是顶层配置项，用于指定性能 report 的 HTTP 上报入口。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `report_API` | string | 否 | `http://127.0.0.1:8081/report` | `report()` 异步上报接口地址，通常由该服务写入 ClickHouse 并供 Grafana 展示 |

作用位置：`report_API` 由 `src/base/report/report_client.cpp` 读取；当该字段缺失时会回退到默认地址 `http://127.0.0.1:8081/report`。

## 5. hugectr 配置

`hugectr` 顶层配置用于 HugeCTR 入口的运行时后端选择。它不会修改 `CommonOp` 语义，而是只影响 `src/framework/hugectr/op_hugectr.cc` 内部的 backend 分发。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `backend` | string | 否 | `recstore` | HugeCTR 运行时后端，可选 `recstore` 或 `hierkv` |

当前行为：

- `recstore`：沿用现有 host staging + RecStore OP 路径
- `hierkv`：切到 HugeCTR 专用 HierKV backend 适配层

## 6. hierkv 配置

`hierkv` 顶层配置仅在 `hugectr.backend = "hierkv"` 时使用，用于初始化 HugeCTR 专用 HierarchicalKV backend。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `max_capacity` | integer | 是 | HierarchicalKV 最大容量 |
| `max_hbm_for_vectors` | integer | 是 | 向量可使用的 HBM 上限，单位字节 |
| `dim` | integer | 是 | embedding 维度 |
| `device_id` | integer | 否 | 可选 CUDA 设备号 |
| `reserved_key_start_bit` | integer | 否 | 可选保留 key 起始 bit，范围 0-62 |

注意：

- 这组配置当前只进入 HugeCTR 侧适配层，不会回流到 `CommonOp`
- 如果选择 `hierkv` 但缺少上述必填字段，HugeCTR 入口会直接抛出配置错误

## 配置示例

???+ example "完整配置示例"

    ```json
    {
        "cache_ps": {
            "ps_type": "GRPC",
            "max_batch_keys_size": 65536,
            "num_threads": 32,
            "num_shards": 2,
            "servers": [
                {
                    "host": "127.0.0.1",
                    "port": 15000,
                    "shard": 0
                },
                {
                    "host": "127.0.0.1",
                    "port": 15001,
                    "shard": 1
                }
            ],
            "base_kv_config": {
                "engine_type": "KVEngineComposite",
                "capacity": 40000000,
                "index": {"type": "DRAM_PET_HASH"},
                "value": {
                    "type": "DRAM_VALUE_STORE",
                    "default_value_size_hint": 512,
                    "path": "/dev/shm/recstore_kv/value",
                    "dram_allocator": {
                        "type": "CONCURRENT_SLAB_MEMORY_POOL",
                        "capacity_bytes": 20480000000
                    }
                }
            }
        },
        "distributed_client": {
            "num_shards": 2,
            "hash_method": "city_hash",
            "max_keys_per_request": 500,
            "servers": [
                {
                    "host": "127.0.0.1",
                    "port": 15000,
                    "shard": 0
                },
                {
                    "host": "127.0.0.1",
                    "port": 15001,
                    "shard": 1
                }
            ]
        },
        "client": {
            "host": "127.0.0.1",
            "port": 15000,
            "shard": 0
        },
        "report_API": "http://127.0.0.1:8081/report"
    }
    ```

??? example "DRAM 索引 + SSD 值配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "capacity": 2000000,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "SSD_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "path": "/data/recstore/value_pages.db",
                    "ssd_allocator": {
                        "type": "SSD_BUDDY",
                        "capacity_bytes": 256000000,
                        "min_block_size": 128,
                        "max_block_size": 65536,
                        "io": {
                            "type": "IOURING",
                            "queue_depth": 512,
                            "base_offset_bytes": 4096
                        }
                    }
                }
            }
        }
    }
    ```

    `"engine_type": "KVEngineComposite"`

??? example "SSD 索引 + SSD 值配置"
    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "capacity": 2000000,
                "index": {
                    "type": "SSD_EXTENDIBLE_HASH",
                    "path": "/data/recstore/index_pages.db",
                    "io": {
                        "type": "IOURING",
                        "queue_depth": 512,
                        "base_offset_bytes": 0
                    }
                },
                "value": {
                    "type": "SSD_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "path": "/data/recstore/value_pages.db",
                    "ssd_allocator": {
                        "type": "SSD_BUDDY",
                        "capacity_bytes": 256000000,
                        "min_block_size": 128,
                        "max_block_size": 65536,
                        "io": {
                            "type": "IOURING",
                            "queue_depth": 512,
                            "base_offset_bytes": 4096
                        }
                    }
                }
            }
        }
    }
    ```

    推导引擎类型：`KVEngineCCEH`

??? example "HYBRID 混合模式配置"

    ```json
    {
        "cache_ps": {
            "base_kv_config": {
                "capacity": 2000000,
                "index": {"type": "DRAM_EXTENDIBLE_HASH"},
                "value": {
                    "type": "TIERED_VALUE_STORE",
                    "default_value_size_hint": 128,
                    "dram_allocator": {
                        "type": "PERSIST_LOOP_SLAB",
                        "capacity_bytes": 10737418240,
                        "path": "/dev/shm/recstore/tiered_dram"
                    },
                    "ssd_allocator": {
                        "type": "SSD_BUDDY",
                        "capacity_bytes": 107374182400,
                        "min_block_size": 128,
                        "max_block_size": 65536,
                        "path": "/data/recstore/tiered_value_pages.db",
                        "io": {
                            "type": "IOURING",
                            "queue_depth": 512,
                            "base_offset_bytes": 4096
                        }
                    }
                }
            }
        }
    }
    ```

    推导引擎类型：`KVEngineHybrid`

??? example "在 CI 中的配置"

    Github Actions 服务器环境限制 CPU 使用，同时无显卡支持，在 ci 配置脚本中使用：

    ```shell
    jq '.cache_ps.base_kv_config.capacity = 512
        | .cache_ps.max_batch_keys_size = 128
        | .cache_ps.num_threads = 4
        | .distributed_client.max_keys_per_request = 32
        | .cache_ps.base_kv_config.index = {"type": "DRAM_EXTENDIBLE_HASH"}
        | .cache_ps.base_kv_config.value = {
            "type": "SSD_VALUE_STORE",
            "default_value_size_hint": 128,
            "path": "/tmp/recstore/value_pages.db",
            "ssd_allocator": {
              "type": "SSD_BUDDY",
              "capacity_bytes": 65536,
              "min_block_size": 128,
              "max_block_size": 65536,
              "io": {
                "type": "IOURING",
                "queue_depth": 512,
                "base_offset_bytes": 4096
              }
            }
          }'
    ```

    来配置DRAM 索引 + SSD 值配置。

## 配置文件使用

配置文件通常保存为 `recstore_config.json`，可通过以下方式读取：

```cpp
std::ifstream config_file("recstore_config.json");
nlohmann::json config;
config_file >> config;

// 服务器端使用
auto cache_ps = std::make_unique<CachePS>(config["cache_ps"]);

// 分布式客户端使用
auto dist_client = std::make_unique<DistributedBRPCParameterClient>(config);

// 单节点客户端使用
auto client = std::make_unique<GRPCParameterClient>(config["client"]);
```
