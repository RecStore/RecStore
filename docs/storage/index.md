# RecStore 存储层

存储层为参数服务器提供本地键值语义：`uint64_t key -> embedding 字节`。请求由 `CachePS` 经 `BaseKV` 接口下发；具体落盘/落内存策略由 `engine_type` 选择的 KV 引擎决定。

## 在系统中的位置

```mermaid
graph TD
    Client[PSClient / Trainer] --> RPC[brpc / gRPC / RDMA / LocalShm]
    RPC --> PS[ParameterService]
    PS --> CachePS[CachePS]
    CachePS --> BaseKV[BaseKV]
    BaseKV --> E1[KVEngineComposite]
    BaseKV --> E2[KVEnginePetKV]
    BaseKV --> E3[KVEngineFasterKV / HPS ...]
    E1 --> Index[Index]
    E1 --> VS[ValueStore]
    VS --> Mem[memory/ allocators]
    VS --> IO[IOBackend]
```

- **参数服务**（`src/ps/`）：协议、分片、压缩、优化器；不直接操作 `Index` / `ValueStore`。
- **KV 引擎**（`src/storage/kv_engine/`）：实现 `BaseKV`，对外隐藏索引与 value 布局。
- **可组合组件**（`src/storage/index/`、`value_store/`、`io_backend/`、`allocator/`）：主要由 `KVEngineComposite` 组装；也可被测试或后续引擎单独使用。
- **外部与遗留适配**（`src/storage/external/`、`src/storage/nvm/pet_kv/`）：FasterKV、HugeCTR HPS、PetKV 等整包实现。

引擎选择与 `BaseKV` 接口说明见 [basekv.md](./basekv.md)。默认生产路径为 **`KVEngineComposite`**（`DRAM_PET_HASH` + `DRAM_VALUE_STORE` + `CONCURRENT_SLAB_MEMORY_POOL`，见根目录 `recstore_config.json`），组件与配置见 [composite_kvengine.md](./composite_kvengine.md)。

## 代码目录

| 目录 / 文件 | 职责 |
|-------------|------|
| `src/ps/base/cache_ps_impl.h` | 从 `cache_ps.base_kv_config` 构造 `BaseKV` |
| `src/storage/kv_engine/base_kv.h` | `BaseKV` / `BaseKVConfig` 抽象 |
| `src/storage/kv_engine/engine_selector.h` | `ResolveEngine`：`engine_type` 解析 |
| `src/storage/kv_engine/engine_composite.h` | 默认组合引擎 |
| `src/storage/kv_engine/engine_petkv.h` | PetKV 引擎 |
| `src/storage/kv_engine/kv_engine_register.cc` | 链接 IO 后端等注册单元 |
| `src/storage/index/` | `Index`：key → value handle |
| `src/storage/value_store/` | `ValueStore`：handle → 字节 |
| `src/storage/io_backend/` | 页级 SSD 读写（io_uring / SPDK） |
| `src/storage/allocator/ssd/` | SSD slab / buddy |
| `src/memory/allocators/` | DRAM `MallocApi`（PersistLoop、R2 等） |
| `src/storage/external/` | FasterKV、HPS 适配 |

服务端 JSON 字段总表见 [config.md](../config.md) 的 `base_kv_config` 一节。

## CachePS 与 BaseKV

`CachePS` 在构造时读取 `cache_ps` 配置：

```cpp
BaseKVConfig kv_config;
kv_config.num_threads_ = config["num_threads"];
kv_config.json_config_ = config["base_kv_config"];
auto resolved = base::ResolveEngine(kv_config);
base_kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
    resolved.engine, resolved.cfg));
```

- `num_threads_` 传入引擎，供批量与组件内部并行使用。
- `json_config_` 原样交给具体引擎；**字段合法性由各引擎及其子组件构造时检查**。
- 省略 `engine_type` 时，`ResolveEngine` 默认 `KVEngineComposite`。

典型调用链：

| CachePS 操作 | BaseKV 方法 |
|--------------|-------------|
| 单条读写 | `Put` / `Get` |
| 批量 embedding | `BatchPut` / `BatchGet` |
| 表初始化 / 优化器 | `optimizer_->Init(..., base_kv_.get())` |
| 清空 | `clear()` |

## KVEngineComposite 读写路径（默认）

写入（`Put` / `BatchPut`）：

1. `CachePS` 收到 key 与 float 数据，调用 `BaseKV::Put` 或 `BatchPut`。
2. `KVEngineComposite` 在 `ValueStore` 上 `AllocAndWrite` 新 slot。
3. `Index::Put` 更新 key 的 handle，返回旧 handle。
4. 对旧 handle 调用 `ValueStore::Retire`（延迟回收）。

读取（`Get` / `BatchGet`）：

1. `Index` 解析 key → handle；未命中为 `kValueHandleNone`，对外表现为空值。
2. 若 `ValueStore::DirectPtr` 可用（DRAM），`BatchGet` 可返回指向底层存储的 `ConstArray<float>` 视图。
3. 否则 `Read` / `BatchRead`（SSD 或 tiered 的 SSD 部分）拷贝到线程局部 buffer。

`BatchPut` 对同一批次内重复 key 会退回逐条 `Put`，避免覆盖顺序不确定。其他引擎的语义以各自实现为准，见 [basekv.md](./basekv.md)。

## 数据生命周期与容量语义

RecStore 的存储层不是持久化数据库。当前实现的 backing 分两类：

| 组件 | 实际 backing | 跨进程共享 | 进程退出后保留 |
|------|--------------|:----------:|:--------------:|
| `DRAM_VALUE_STORE`、`KVEnginePetKV`、所有 DRAM allocator | `mmap(MAP_PRIVATE \| MAP_ANONYMOUS)`（`AnonyDramShmFile`） | 否 | 否 |
| `SSD_VALUE_STORE`、SSD 索引、`SSD_SLAB` / `SSD_BUDDY` | 打开或创建普通文件（`O_RDWR \| O_CREAT \| O_DIRECT`） | 是 | 文件保留，但无一致性保证 |
| `TIERED_VALUE_STORE` | DRAM 部分同第一行；SSD 部分同第二行 | 部分 | 仅 SSD 文件 |

- 配置里的 `value.path` / `dram_allocator.path` 只是 mmap 的名字，**不会创建文件**，不要把它当持久化路径。
- SSD 文件不会自动删除，但写入没有 checkpoint/fsync 语义；把文件内容当成可恢复状态不受支持。
- PetKV 的 `path` 下同样是匿名 DRAM mmap（medium 固定为 `DRAM`）。`Persistence=true` 会调用
  `clflushopt_range`，但 `clflushopt()` 当前第一行就是 `return;`，所以 **flush 是空操作**，
  PetKV 不能当持久内存使用。
- 槽位只做分配和释放，没有在介质之间迁移 key 的后台任务。

容量方面：

- 覆盖写先分配新 slot，再 `Index::Put`，旧 handle 交给 `Retire`；瞬时容量需求是旧值 + 新值。
- `TIERED_VALUE_STORE` 在 DRAM 超过 `high_watermark_ratio` 后写 SSD；写入后不再迁移。
- 目标介质分配失败时，`KVEngineComposite` 的 `Put` / `BatchPut` / `BulkLoad` 走 `LOG(FATAL)`，
  进程终止；没有淘汰、降级或返回错误的路径。

## 文档索引

| 文档 | 内容 |
|------|------|
| [basekv.md](./basekv.md) | `BaseKV` 接口、工厂创建、全部 `engine_type` 与外部引擎配置 |
| [composite_kvengine.md](./composite_kvengine.md) | `KVEngineComposite` 的 Index / ValueStore / IO / allocator 与扩展指南 |
| [ssd_layout.md](https://github.com/RecStore/RecStore/blob/main/src/storage/ssd_layout/ssd_layout.md) | 静态 SSD 布局的使用方式、算法与 artifact 格式 |
| [config.md](../config.md) | `recstore_config.json` 中 `base_kv_config` 字段 |

## 验证与基准

| 范围 | 命令 / 位置 |
|------|-------------|
| Composite 组合 | `ctest -R test_kvengine` |
| 外部引擎 | `test_external_kv_engine` / `KVEngineExternalEngineTest`（需对应 CMake 宏） |
| SSD IO | `test_io_backend` |
| YCSB 对比 | `.agents/skills/rs-benchmark-kvengine/SKILL.md` |
