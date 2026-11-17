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

`CachePS` 把 `cache_ps.base_kv_config` 原样交给引擎：`num_threads_` 供内部并行，其余字段由引擎与组件在构造时校验；省略 `engine_type` 时默认 `KVEngineComposite`。

| CachePS 操作 | BaseKV 方法 |
|--------------|-------------|
| 单条读写 | `Put` / `Get` |
| 批量 embedding | `BatchPut` / `BatchGet` |
| 表初始化 / 优化器 | `optimizer_->Init(..., base_kv_.get())` |
| 清空 | `clear()` |

构造代码、`ResolveEngine` 行为与接口语义见 [basekv.md](./basekv.md)；默认组合的读写顺序见 [composite_kvengine.md](./composite_kvengine.md)。

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
