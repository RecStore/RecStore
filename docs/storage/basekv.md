# BaseKV 与 KV Engine

`BaseKV` 是 RecStore 存储层对参数服务器的统一 KV 抽象。`CachePS` 只依赖该接口；索引在 DRAM 还是 SSD、是否 tiered、是否走第三方库，均由 `engine_type` 选择的实现决定。

存储层总览见 [index.md](./index.md)。**默认实现**为可组合的 `KVEngineComposite`，其 `index` / `value` / allocator 配置见 [composite_kvengine.md](./composite_kvengine.md)。

## 接口语义

```cpp
class BaseKV {
 public:
  virtual void Get(uint64_t key, std::string& value, unsigned tid) = 0;
  virtual void Put(uint64_t key, const std::string_view& value, unsigned tid) = 0;
  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid) = 0;
  virtual void BatchPut(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>>* values,
                        unsigned tid);
  virtual bool ApplySgdUpdateFlat(...);  // 部分引擎实现
  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void* value);
  virtual void clear();
  // Exists, DebugInfo, Util, 协程 BatchGet/BatchPut 等见 base_kv.h
};
```

| 方法 | 约定 |
|------|------|
| `Get` | 未命中时清空 `value` |
| `BatchGet` | 未命中项为空的 `ConstArray<float>`；DRAM 可能返回底层内存视图，**不得跨调用长期持有** |
| `Put` / `BatchPut` | `KVEngineComposite` 支持变长 value；FasterKV / HPS 等常为固定 `value_size` |
| `BulkLoad` | 需配置固定 value 字节数（Composite 用 `value.default_value_size_hint`） |
| `tid` | 保留给调用方；Composite 主要依赖组件内部同步，勿假设 `tid` 全局唯一 |

## 配置与创建

`BaseKVConfig` 含 `num_threads_` 与 `json_config_`（即 `cache_ps.base_kv_config` 对象）。

```cpp
auto resolved = base::ResolveEngine(kv_config);
base_kv_.reset(base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(
    resolved.engine, resolved.cfg));
```

`ResolveEngine`（`engine_selector.h`）读取 `engine_type`（缺省为 `KVEngineComposite`）并校验引擎名称。`index` / `value` 嵌套字段的校验在引擎或 `ValueStore` / `Index` 构造时完成。

已移除 `external_engine_type`；配置中若仍出现该字段会抛出 `invalid_argument`。

### 已知 `engine_type`

| `engine_type` | 实现 | 配置形态 | 文档 |
|---------------|------|----------|------|
| `KVEngineComposite`（默认） | `engine_composite.h` | 嵌套 `index` + `value` | [composite_kvengine.md](./composite_kvengine.md) |
| `KVEnginePetKV` | `engine_petkv.h` | 顶层 `path`、`capacity`、`value_size`、`value_capacity` | 下文 |
| `KVEngineFasterKV` | `external/fasterkv/` | 顶层 `capacity`、`value_size`；可选 `fasterkv` 块 | 下文 |
| `KVEngineHPSHashMap` | `external/hps/` | 顶层 `capacity`、`value_size`；无 `path` | 下文 |
| `KVEngineHPSRocksDB` | `external/hps/` | 同上 + 必填 `path` | 下文 |
| `KVEngineDETable` | `external/detable/` | 顶层 `capacity`、`value_size`；可选 `detable` 块（`library_path`、`block_size`） | — |

---

## KVEngineComposite（摘要）

省略 `engine_type` 或显式写 `KVEngineComposite` 时使用嵌套配置，字段与完整示例见
[composite_kvengine.md](./composite_kvengine.md) 的配置骨架一节。

常用顶层字段：

| 字段 | 说明 |
|------|------|
| `capacity` | 容量估计；`DRAM_PET_HASH` 直接用于表大小 |
| `index.type` | 见 [composite_kvengine.md](./composite_kvengine.md) |
| `value.type` | `DRAM_VALUE_STORE` / `SSD_VALUE_STORE` / `TIERED_VALUE_STORE` |
| `value.default_value_size_hint` | `BulkLoad` 固定 value 大小；`Put` 可变长 |

路径与 allocator 细则、组合矩阵、扩展 Index/ValueStore 的步骤均在 [composite_kvengine.md](./composite_kvengine.md)。

---

## KVEnginePetKV

基于 `base::PetMultiKV` 的共享内存 KV，面向**固定** `value_size` 与 RDMA 内存注册（`RegisterPMAddr`）。

| 字段 | 说明 |
|------|------|
| `path` | shm 根目录（必填） |
| `capacity` | key 容量估计 |
| `value_size` | 每条 value 字节数（必填） |
| `value_capacity` | value 区域总字节预算（必填） |
| `shard_num` | 分片数（默认 16） |

```json
{
  "engine_type": "KVEnginePetKV",
  "path": "/dev/shm/pet_kv",
  "capacity": 1000000,
  "value_size": 512,
  "value_capacity": 536870912,
  "shard_num": 16
}
```

`BatchGet` 行为受 gflag `prefetch_method` 控制：`0` 逐 key `Get`，`1` 走 `PetMultiKV::BatchGet`。

---

## KVEngineFasterKV

Microsoft FASTER 单表适配；由 FASTER 内部管理。

| 字段 | 说明 |
|------|------|
| `capacity` | 表容量 |
| `value_size` 或 `value.default_value_size_hint` | 固定 value 大小；`Put` 必须等长 |
| `path` | 工作目录 hint；SSD 时可用于推导 log 路径 |
| `fasterkv.storage` | `memory`（默认，`NullDisk`）或 `ssd`（`FileSystemDisk`） |
| `fasterkv.log_path` | SSD hybrid log；缺省时可为 `{path}/fasterkv-log` |
| `fasterkv.hlog_memory_bytes` | 可选内存 log 容量 |
| `fasterkv.mutable_fraction` | (0, 1] |
| `fasterkv.read_cache_bytes` | 大于 0 时启用 read cache（按 FASTER 页对齐） |

```json
{
  "engine_type": "KVEngineFasterKV",
  "path": "/tmp/fasterkv_data",
  "capacity": 1000000,
  "value_size": 512,
  "fasterkv": {
    "storage": "ssd",
    "log_path": "/data/fasterkv/hlog",
    "hlog_memory_bytes": 1073741824,
    "mutable_fraction": 0.9,
    "read_cache_bytes": 268435456
  }
}
```

构建需启用对应 CMake 目标；测试见 `test_external_kv_engine` / `RECSTORE_TEST_ENABLE_FASTERKV_ENGINE`。

---

## KVEngineHPSHashMap / KVEngineHPSRocksDB

HugeCTR `DatabaseBackend` 适配，固定 float 对齐的 `value_size`。

| 字段 | `HPSHashMap` | `HPSRocksDB` |
|------|:------------:|:------------:|
| `path` | 可选 | **必填** |
| `capacity` | 容量 hint | 同左 |
| `value_size` | 必填 | 必填 |
| `max_batch_size` | 默认 65536 | 同左 |
| `num_threads` | 并发（或由 `CachePS.num_threads` 推导） | 同左 |
| `table_name` | 默认 `default` | 同左 |
| `rocksdb_path` | — | 可选，默认 `{path}/hps_rocksdb` |

```json
{
  "engine_type": "KVEngineHPSRocksDB",
  "path": "/data/rocksdb",
  "capacity": 1000000,
  "value_size": 512
}
```

纯内存对比：

```json
{
  "engine_type": "KVEngineHPSHashMap",
  "capacity": 1000000,
  "value_size": 512
}
```