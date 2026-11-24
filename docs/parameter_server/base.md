# PS Client / Server

## 概述

基础模块定义参数服务器的核心抽象接口和数据结构，为不同协议实现提供统一的编程接口。

## 核心接口

### BaseParameterServer

服务器抽象基类，所有具体实现（gRPC、bRPC）继承此类。

| 方法 | 说明 |
|------|------|
| `Init(const json& config)` | 初始化服务器配置 |
| `Run()` | 启动服务器主循环（阻塞） |

### BasePSClient

客户端抽象基类，定义参数操作接口。

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `GetParameter` | keys, values | int | 同步获取参数 |
| `PutParameter` | keys, values | int | 设置参数初值 |
| `UpdateParameter` | table_name, keys, grads | int | 更新参数 |
| `InitEmbeddingTable` | table_name, config | int | 初始化嵌入表 |
| `AsyncGetParameter` | keys, values | int | 异步获取参数 |
| `PrefetchParameter` | keys | uint64_t | 预取参数，返回 ID |
| `IsPrefetchDone` | prefetch_id | bool | 检查预取是否完成 |
| `WaitForPrefetch` | prefetch_id | void | 等待预取完成 |
| `GetPrefetchResult` | prefetch_id, values | bool | 获取预取结果 |
| `Command` | command | void | 执行控制命令 |

## 数据结构

### ParameterCompressItem

参数压缩存储格式：

| 字段 | 类型 | 说明 |
|------|------|------|
| key | uint64_t | 参数键 |
| dim | int | 参数维度 |
| embedding | float[] | 可变长度数据 |

相关类型：
- `ParameterPack` - 压缩打包工具
- `ParameterCompressor` - 压缩器
- `ParameterCompressReader` - 解压读取器

### EmbeddingTableConfig

嵌入表配置结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| num_embeddings | uint64_t | 嵌入数量 |
| embedding_dim | uint64_t | 嵌入维度 |

### PSCommand

系统控制命令枚举：

| 命令 | 说明 |
|------|------|
| CLEAR_PS | 清空参数服务器 |
| RELOAD_PS | 重新加载参数 |
| LOAD_FAKE_DATA | 加载测试数据 |
| DUMP_FAKE_DATA | 读出指定字节数的测试数据 |
| SAVE_CHECKPOINT | 全量存档一个分片（gRPC / bRPC） |
| LOAD_CHECKPOINT | 从存档恢复一个分片（gRPC / bRPC） |

## 存档与恢复 {#checkpoint-restore}

`SAVE_CHECKPOINT` / `LOAD_CHECKPOINT` 的 payload 是 `arg1[0]` 路径加 `arg2[0]` metadata JSON；metadata 必须含非空的 `checkpoint_id` 和对象型 `identity`，服务端再补上 `optimizer_layout` 与 `shard_id` 后交给引擎。`optimizer_layout` 是当前的键布局版本（`kCheckpointOptimizerLayout`）：优化器 tag 或键编码一改就递增，旧版本写的分片在加载时被拒绝，不会把 optimizer slot 当成缺失行重新初始化。加载只在全新分片上成立：不能有未存档的更新、不能已绑定过另一个 identity、引擎必须为空。metadata 逐字节比较，所以存档和加载要用同一份 identity；跨 `optimizer_layout` 的存档需要先在新二进制上重新存档，或者从头训练。

| 能力 | 现状 |
|------|------|
| gRPC / bRPC | `KVClient.save_checkpoint()` / `load_checkpoint()` 逐分片 fan-out |
| RDMA | 数据面没有 checkpoint 命令；`Command` 会报错，不会假装成功 |
| `KVEnginePetKV` | 明确不支持（共享内存不持久） |
| 其他引擎 | 全量快照：持引擎写锁 dump 所有 key，参数与 optimizer slot 一起落盘 |

自动化的最小闭环由 `recstore.checkpoint` 提供：`save_sparse_checkpoint()` 把分片写进 `root/<checkpoint_id>/` 的临时目录，落完后写 `manifest.json`（checkpoint_id / step / identity / 分片文件名与大小），最后整目录改名为正式版本；读端只认带 manifest 的目录，因此半截存档不会被当成可用版本。`PeriodicSparseCheckpoint.maybe_save(step)` 在步边界按间隔存档并保留最近 N 份。

服务端启动恢复读取 `cache_ps.checkpoint`：

| 字段 | 说明 |
|------|------|
| `restore_dir` | 存档根目录（取 step 最大的已提交版本）或单个版本目录；空表示不恢复 |
| `on_missing` | `fail`（默认）没有 manifest 即启动失败；`start_empty` 空分片启动 |

恢复在 RPC 服务开始监听前完成，失败即启动失败。多机部署时 `restore_dir` 必须是所有分片都能读到的共享目录。跨分片快照不是同一个逻辑步：每个分片各自持写锁 dump，加载只保证每片内部一致。

### 迁移旧存档 {#checkpoint-migrate}

带 `optimizer_layout` 之前的存档加载会被拒绝。`checkpoint_migrate`（源码 `src/ps/checkpoint/`）把旧版本目录重写成当前布局，输出到新目录，原件不动：

```bash
checkpoint_migrate --input <存档根目录或版本目录> --output <新目录> \
  --optimizer AdamW --table-id 7 [--shards N] [--hash city_hash]
```

- **重新分片**：客户端按参数键的 hash 选分片，AdamW 的 tag 变了行就换分片，所以工具重排记录而不是逐文件拷贝。`--hash` 必须与该次训练 `distributed_client.hash_method` 一致；`--shards` 只在存档缺分片或换分片数时需要。
- **AdamW 的 tag**：从 role 改成 `table_id * 4 + role`，`--table-id` 必填——旧 tag 没记录表号。step 是服务端本地键，工具给每个新分片各写一份，并要求旧分片之间的 step 值一致，不一致直接拒绝。
- **只适用于单张 AdamW 表的存档**（#411 之后的保证）。更早的多表存档当时就已经互相覆盖，工具无法从文件里识别。
- 输出目录必须不存在；manifest 最后写，中断的目录读端会忽略。`--no-verify` 跳过回读校验。

SGD / RowWiseAdagrad 的键不变，迁移等价于补 metadata 标记；AdamW 涉及键搬迁，迁移后先用新二进制实际加载一次再清理原存档。

## 文件组织

| 文件 | 说明 |
|------|------|
| `base_ps_server.h` | 服务器抽象基类 |
| `base_client.h` / `.cc` | 客户端抽象基类 |
| `parameters.h` / `.cpp` | 参数数据结构 |
| `config.h` / `.cc` | 配置管理 |
| `cache_ps_impl.h` | 缓存实现 |
| `Postoffice.h` / `.cc` | 进程间通信 |
| `shard_manager.h` | 分片管理 |
