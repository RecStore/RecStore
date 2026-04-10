# PS 模块：gRPC / bRPC 与分布式接口说明


---

## 1. Proto 与包名

| 项 | gRPC | bRPC |
|----|------|------|
| 文件 | `proto/ps.proto` | `proto/ps_brpc.proto` |
| `package` | `recstoreps` | `recstoreps_brpc` |
| 生成服务 | `grpc::Service`（`ps.grpc.pb.h`） | `cc_generic_services = true`，同步 `Rpc` 方法（`ps_brpc.pb.h`） |



```protobuf
service ParameterService {
  rpc GetParameter(GetParameterRequest) returns (GetParameterResponse);
  rpc Command(CommandRequest) returns (CommandResponse);
  rpc PutParameter(PutParameterRequest) returns (PutParameterResponse);
  rpc UpdateParameter(UpdateParameterRequest) returns (UpdateParameterResponse);
  rpc InitEmbeddingTable(InitEmbeddingTableRequest) returns (InitEmbeddingTableResponse);
}
```

---

## 2. RPC 请求 / 响应字段与服务端实现

服务端实现中，gRPC 为 `ParameterServiceImpl`（`grpc/grpc_ps_server.cpp`），bRPC 为 `BRPCParameterServiceImpl`（`brpc/brpc_ps_server.cpp`）。二者均持有 `CachePS* cache_ps_`

---

### 2.1 `GetParameter`

**Proto（`GetParameterRequest` / `GetParameterResponse`）**

| 字段 | 方向 | Proto 类型 | 服务端使用说明 |
|------|------|------------|----------------|
| `keys` | 请求 | `optional bytes` | 必填。解释为连续 `uint64_t`，长度须为 `8` 的倍数。 |
| `model_name` | 请求 | `optional bytes` | 当前服务端未读取，不影响处理。 |
| `perf` | 请求 | `optional bool` | 若为真则打 perf / 计时分支。 |
| `keys` | 响应 | `optional bytes` | **当前服务端未设置**。 |

**gRPC 处理函数**（`grpc/grpc_ps_server.cpp`）

```cpp
grpc::Status GetParameter(grpc::ServerContext* context,
                          const recstoreps::GetParameterRequest* request,
                          recstoreps::GetParameterResponse* reply) override;
```

- `base::ConstArray<uint64_t> keys_array(request->keys());` —— 即用 `ConstArray` 对 `std::string`（bytes）的构造函数，按 `sizeof(uint64_t)` 定长切 key。
- `cache_ps_->GetParameterRun2Completion(keys_array, packs, /*tid=*/0);`
- `ParameterCompressor::ToBlock` 后 `reply->mutable_parameter_value()->swap(blocks[0]);`

**bRPC 处理函数**（`brpc/brpc_ps_server.cpp`）

```cpp
void GetParameter(google::protobuf::RpcController* controller,
                  const recstoreps_brpc::GetParameterRequest* request,
                  recstoreps_brpc::GetParameterResponse* response,
                  google::protobuf::Closure* done);
```

- `ExtractPayloadBytes(cntl, request->keys(), ...)`：若 `request_attachment()` 非空则优先用附件，否则用 `request->keys()`；再 `keys_array.SetData(keys_data, keys_size)`，且校验 `keys_size % sizeof(uint64_t) == 0`。
- 同样 `cache_ps_->GetParameterRun2Completion(keys_array, packs, 0);`
- 结果：`compressor.AppendToIOBuf(&cntl->response_attachment());`（大块走 IOBuf，而非仅填 `response` 的 bytes 字段）。

**`CachePS` 调用**（`base/cache_ps_impl.h`）

```cpp
bool GetParameterRun2Completion(key_t key, ParameterPack& pack, int tid);        
bool GetParameterRun2Completion(base::ConstArray<uint64_t> keys,
                                std::vector<ParameterPack>& packs,
                                int tid);
```


---

### 2.2 `PutParameter`

**Proto**

| 字段 | 方向 | Proto 类型 | 说明 |
|------|------|------------|------|
| `parameter_value` | 请求 | `optional bytes` | 整块缓冲区视为 `ParameterCompressReader` 布局（与 `ParameterCompressItem` 序列一致）。 |
| `sucess` | 响应 | `optional bool` | 拼写与 proto 一致。**当前 gRPC / bRPC 服务端均未调用 `set_sucess`**，成功时仅返回 RPC 成功；不要依赖该字段。 |

**gRPC**

```cpp
grpc::Status PutParameter(grpc::ServerContext* context,
                          const recstoreps::PutParameterRequest* request,
                          recstoreps::PutParameterResponse* reply) override;
```

- `reader = reinterpret_cast<const ParameterCompressReader*>(request->parameter_value().data());`
- 对 `i in [0, reader->item_size())`：`cache_ps_->PutSingleParameter(reader->item(i), /*tid=*/0);`

**bRPC**

```cpp
void PutParameter(google::protobuf::RpcController* controller,
                  const recstoreps_brpc::PutParameterRequest* request,
                  recstoreps_brpc::PutParameterResponse* response,
                  google::protobuf::Closure* done);
```

- `ExtractPayloadBytes(cntl, request->parameter_value(), ...)` 得到 `payload_data` / `payload_size`。
- `reader = reinterpret_cast<const ParameterCompressReader*>(payload_data);`
- `reader->Valid(payload_size)` 不通过则打日志并返回。
- 同样循环 `cache_ps_->PutSingleParameter(reader->item(i), 0);`

**`CachePS`**

```cpp
void PutSingleParameter(const ParameterCompressItem* item, int tid);
```


---

### 2.3 `UpdateParameter`

**Proto**

| 字段 | 类型 | 说明 |
|------|------|------|
| `table_name` | `string` | 与 `InitEmbeddingTable` 注册的表名一致。 |
| `gradients` | `optional bytes` | 与 Put 相同布局的压缩梯度批次。 |
| `success` | `optional bool` | 服务端 `reply->set_success(...)`。 |

**gRPC**

```cpp
grpc::Status UpdateParameter(grpc::ServerContext* context,
                             const recstoreps::UpdateParameterRequest* request,
                             recstoreps::UpdateParameterResponse* reply) override;
```

- `table_name = request->table_name();`
- `reader = reinterpret_cast<const ParameterCompressReader*>(request->gradients().data());`（**无** `Valid`）
- `success = cache_ps_->UpdateParameter(table_name, reader, /*tid=*/0);`
- `reply->set_success(success);`，异常时 `set_success(false)`。

**bRPC**

```cpp
void UpdateParameter(google::protobuf::RpcController* controller,
                     const recstoreps_brpc::UpdateParameterRequest* request,
                     recstoreps_brpc::UpdateParameterResponse* reply,
                     google::protobuf::Closure* done);
```

- `gradients` 经 `ExtractPayloadBytes`；`reader->Valid(payload_size)` 不通过则抛错并 `set_success(false)`。
- 同样 `cache_ps_->UpdateParameter(table_name, reader, 0)` → `reply->set_success(success)`。

**`CachePS`**

```cpp
bool UpdateParameter(const std::string& table_name,
                     const ParameterCompressReader* reader,
                     unsigned tid);
```

需已 `InitTable`；内部 `optimizer_->Update(table_name, reader, tid)`（`optimizer/optimizer.cpp`）。

---

### 2.4 `InitEmbeddingTable`

**Proto**

| 字段 | 类型 | 说明 |
|------|------|------|
| `table_name` | `string` | 逻辑表名。 |
| `config_payload` | `optional bytes` | UTF-8 JSON，需含 `num_embeddings`、`embedding_dim`（与 `EmbeddingTableConfig::Serialize()` 一致）。 |
| `success` | `optional bool` | 解析失败或缺 `config_payload` 时为 `false`。 |

**gRPC**

```cpp
grpc::Status InitEmbeddingTable(
    grpc::ServerContext* context,
    const recstoreps::InitEmbeddingTableRequest* request,
    recstoreps::InitEmbeddingTableResponse* reply) override;
```

- `request->has_config_payload()` 为假 → `reply->set_success(false)`。
- `nlohmann::json::parse(request->config_payload());` 取 `uint64_t num_embeddings`、`embedding_dim`（`cfg.value("num_embeddings", 0)` 等）。
- `init_success = cache_ps_->InitTable(request->table_name(), num_embeddings, embedding_dim);`
- `reply->set_success(init_success);`

**bRPC**：同上逻辑，签名为 `InitEmbeddingTable(RpcController*, const InitEmbeddingTableRequest*, InitEmbeddingTableResponse*, Closure*)`。

**`CachePS`**

```cpp
bool InitTable(const std::string& table_name,
               uint64_t num_embeddings,
               uint64_t embedding_dim);
```

首次会创建默认 `SGD(0.01)` 并 `optimizer_->Init(..., base_kv_.get())`（见 `cache_ps_impl.h`）。

---

### 2.5 `Command`

| 字段 | 含义 |
|------|------|
| `command` | 枚举：`CLEAR_PS`、`RELOAD_PS`、`LOAD_FAKE_DATA`、`DUMP_FAKE_DATA` |
| `arg1` / `arg2` / `arg3` | `repeated bytes`，按具体命令解析 |
| `reply` | 文本回复 |

**实现要点**：见 **§5.2** 表中 `Command` 各行；`LOAD_FAKE_DATA` / `DUMP_FAKE_DATA` 主要为带宽压测，不读写 `BaseKV`。

---

## 3. 统一入口：`ps_server`

- 源码：`ps_server.cpp`。
- 读取 `--config_path`（文件不存在时可回退 `--brpc_config_path`）。
- 从 `cache_ps.ps_type` 取值（大小写不敏感）：`GRPC` → 工厂键 `GRPCParameterServer`；`BRPC` → `BRPCParameterServer`。
- 流程：`Factory<BaseParameterServer>::NewInstance(...)` → `Init(config)` → `Run()`。

---

## 4. 服务端实现：单进程多监听（单分片 / 多分片）

### 4.1 gRPC — `GRPCParameterServer`（`grpc/grpc_ps_server.cpp`）

| 模式 | 条件 | 监听与存储 |
|------|------|------------|
| 单分片 | `cache_ps.num_shards` 缺省或 `≤ 1` | 固定 `0.0.0.0:15000`；单个 `CachePS(config_["cache_ps"])` |
| 多分片 | `num_shards > 1` 且提供 `cache_ps.servers`（长度等于 `num_shards`） | 每个元素 `{host, port, shard}` 一线程：`ServerBuilder` 监听 `host:port`；从 `cache_ps` 拷贝出 `shard_config`，若存在 `base_kv_config.path` 则追加 `_{shard}` 避免路径冲突；每线程独立 `CachePS` + `ParameterServiceImpl` |

### 4.2 bRPC — `BRPCParameterServer`（`brpc/brpc_ps_server.cpp`）

| 模式 | 条件 | 监听与存储 |
|------|------|------------|
| 单分片 | 同上 | `0.0.0.0:` + `FLAGS_brpc_server_port`（默认 15000）；单个 `CachePS` |
| 多分片 | 同上 | 每 shard 一线程监听 `host:port`；当前实现每线程使用**同一份** `config_["cache_ps"]` 构造 `CachePS`（**未**像 gRPC 那样自动改写 `path` 后缀），部署时需自行避免多进程数据目录冲突 |

**工厂注册**：

- `FACTORY_REGISTER(BaseParameterServer, GRPCParameterServer, GRPCParameterServer)`
- `FACTORY_REGISTER(BaseParameterServer, BRPCParameterServer, BRPCParameterServer)`

---

## 5. 服务端核心：`CachePS` 与存储层 `BaseKV`

gRPC / bRPC 的 `ParameterServiceImpl` / `BRPCParameterServiceImpl` 不直接碰 KV 引擎，统一通过 **`CachePS`**（`base/cache_ps_impl.h`）访问 **`BaseKV`**
### 5.1 `CachePS` 如何创建存储引擎

构造时从 `cache_ps` JSON 读取：

- `num_threads` → `BaseKVConfig::num_threads_`
- `base_kv_config` → `BaseKVConfig::json_config_`

随后 `base::ResolveEngine(kv_config)` 根据 `index_type` / `value_type` 等解析具体引擎名，再通过 `base::Factory<BaseKV, const BaseKVConfig&>::NewInstance(engine, cfg)` 创建 **`std::unique_ptr<BaseKV> base_kv_`**。引擎注册与配置字段说明见 `base_kv.h` 顶部注释（`DATA_DIR`、`VALUE_SIZE_BYTES`、HYBRID 等）。



### 5.2 RPC → `CachePS` → 存储 / 优化器（调用链）

以下对 **gRPC 与 bRPC 服务端** 一致（实现代码对称）。

| RPC / 分支 | `CachePS` 入口 | 进一步调用 | 说明 |
|------------|----------------|------------|------|
| **GetParameter** | `GetParameterRun2Completion(keys, packs, tid)` | `base_kv_->BatchGet(keys, &values, tid)`，再组装 `ParameterPack` → `ParameterCompressor` 回包 | 只读 **原始 uint64 key**，不经 `SparseTensor` 打 tag。 |
| **PutParameter** | `PutSingleParameter(item, tid)`（按条循环） | `base_kv_->Put(key, string_view(embedding bytes), tid)` | 与 Get 相同，**直连 `BaseKV`**，key 无 optimizer tag。 |
| **InitEmbeddingTable** | `InitTable(table_name, num_embeddings, embedding_dim)` | 懒创建 `SGD(0.01)`（若尚无 `optimizer_`），再 `optimizer_->Init({table_name}, config, base_kv_.get())` | `SGD::Init` 为每张表创建 `SparseTensor`，把 **`BaseKV*`** 存进 tensor，供后续 Update 使用。 |
| **UpdateParameter** | `UpdateParameter(table_name, reader, tid)` | `optimizer_->Update(table_name, reader, tid)` | 经 `SparseTensor` 对 key 做 **tag 拼接** 后再 `BatchGet` / `Put`（见 `optimizer/optimizer.cpp`）。 |
| **Command `CLEAR_PS`** | `Clear()` | `base_kv_->clear()` | 清空 KV。 |
| **Command `RELOAD_PS`** | `Initialize(arg1, arg2)` | 当前实现内 `LoadCkpt` **为空实现**（直接 `return true`），**未**调用 `base_kv_` 加载检查点 | 若需真实重载，要在 `CachePS::LoadCkpt` 中接存储层 API。 |
| **Command `LOAD_FAKE_DATA`** | — | **不访问** `BaseKV` | 服务端按 `arg1` 中 `int64_t` 字节数生成随机 `reply` 载荷，测下行带宽。 |
| **Command `DUMP_FAKE_DATA`** | — | **不访问** `BaseKV` | 客户端上行 payload 由框架接收，服务端 `reply` 确认，测上行带宽。 |




---




## 6. 分布式客户端（跨 shard 路由）

### 6.1 配置 JSON

顶层对象需包含 `distributed_client`（也可将整份配置传入，构造器会抽取该段），例如：

```json
{
  "distributed_client": {
    "num_shards": 2,
    "hash_method": "city_hash",
    "max_keys_per_request": 500,
    "servers": [
      {"host": "127.0.0.1", "port": 15000, "shard": 0},
      {"host": "127.0.0.1", "port": 15001, "shard": 1}
    ]
  }
}
```

| 字段 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `servers` | 数组 | （必填） | 每项含 `host`、`port`、`shard`；`shard` 为逻辑分片号，用于与 `GetShardId` 结果对应。 |
| `num_shards` | 整数 | （必填） | 参与取模的分片数；与 `servers` 条数不一致时仅打 **WARNING**。 |
| `hash_method` | 字符串 | `"city_hash"` | `"city_hash"` → `GetHash(key) % num_shards_`（`base/hash.h`）；`"simple_mod"` → `key % num_shards_`。 |
| `max_keys_per_request` | 整数 | `500` | **每个 shard 上**累计 key 超过该值时会 **截断** 该 shard 的 key 列表并打 WARNING（可能丢请求内部分 key，部署时需调大或控制批量）。 |

路由约定：`shard_id = f(key) % num_shards`，其中 `f` 由 `hash_method` 决定。Get / Put / Update 均先 **`PartitionKeys`**，再对非空分片并发调用底层单机客户端，最后用 **`key_index_mapping_`** 把各分片结果填回**与输入 keys 相同顺序**的输出。

---

### 6.2 `DistributedGRPCParameterClient`（`grpc/dist_grpc_ps_client.*`）

- **工厂键**：`distributed_grpc` — `FACTORY_REGISTER(BasePSClient, distributed_grpc, DistributedGRPCParameterClient, json)`。
- **内部结构**：`std::vector<std::unique_ptr<GRPCParameterClient>> clients_`，`shard_to_client_index_[shard] → clients_` 下标；构造时对每个 `ServerConfig` 生成 `{"host","port","shard"}` 的 JSON 并 `make_unique<GRPCParameterClient>`。

### 6.3 `DistributedBRPCParameterClient`（`brpc/dist_brpc_ps_client.*`）

- **工厂键**：`distributed_brpc` — `FACTORY_REGISTER(BasePSClient, distributed_brpc, DistributedBRPCParameterClient, json)`。
- **内部结构**：与 gRPC 版对称，底层为 `BRPCParameterClient`。分片、分区、合并、`std::async` 并发模式与 **§6.4 / §6.5** 描述一致。

---

### 6.4 对外接口

两类分布式客户端均 **`public` 继承 `recstore::BasePSClient`**。

**实现 `BasePSClient` 的方法（签名见 `ps/base/base_client.h`）**

| 方法 | 分布式行为概要 |
|------|----------------|
| `GetParameter(keys, float*)` | 先 `GetParameter(keys, &vector<vector<float>>)`，再合并到连续 `float` 缓冲（假定各 shard 返回的 embedding 维一致）。 |
| `AsyncGetParameter` | 当前直接转调同步 `GetParameter(keys, float*)`。 |
| `PutParameter(keys, values)` | 按 shard 拆分 keys 与 `vector<vector<float>>`，`std::async` 并行 `client->PutParameter`；任一分片返回非 `1` 则整体失败。 |
| `UpdateParameter` / `UpdateParameterFlat` | 按 shard 拆分 key 与梯度行，并行 `UpdateParameter`；`Flat` 先展开为 `vector<vector<float>>` 再复用同一逻辑。 |
| `InitEmbeddingTable(table, config)` | **广播**：对每个 `clients_[i]` 异步调用 `InitEmbeddingTable`（**每个 shard 上都建同名表**），任一分片非 0 则返回 -1。 |
| `Command(PSCommand)` | **广播**：每个底层 client 异步 `Command(command)`，仅 `wait` 不汇总错误。 |


**扩展接口（头文件中声明，非 `BasePSClient` 虚函数）**

| 方法 | 说明 |
|------|------|
| `bool GetParameter(keys, vector<vector<float>>* values)` | 分布式读的主实现路径；内部 `PartitionKeys` + 并行 RPC + `MergeResults`。 |
| `int shard_count() const` | 返回构造时的 `num_shards_`。 |
| `bool ClearPS()` | 对所有 shard 异步 `ClearPS()`，**全部 true** 才返回 `true`。 |
| `bool LoadFakeData(int64_t n)` / `bool DumpFakeData(int64_t n)` | 各 shard 各调一遍（压测带宽；语义为「每节点都跑一遍」）。 |
| `bool LoadCkpt(model_paths, emb_paths)` | 各 shard 并行 `LoadCkpt`；全成功才 `true`。 |

**私有辅助（实现思路，不对外）**

- `GetShardId(uint64_t key) const`：哈希或取模得到 `0 .. num_shards_-1`。
- `PartitionKeys(keys, partitioned_keys)`：填充 `partitioned_key_buffer_` 与 `key_index_mapping_[shard]`（原始 key 在请求中的下标），并施加 **`max_keys_per_request_` 截断**。
- `MergeResults` / `MergeResultsToArray`：按 `key_index_mapping_` 把各 shard 的 `vector<vector<float>>` 写回全局顺序。

---

### 6.5 客户端侧实现要点

1. **单 key 归属唯一 shard**：任意 key 只发往一个 `GRPCParameterClient` / `BRPCParameterClient`，避免双写。
2. **并行度**：按 shard 维度 `std::async(std::launch::async, ...)`，空分片跳过；无全局锁，依赖底层 stub/channel 线程安全。
3. **顺序恢复**：分片返回的向量顺序与发往该分片的子 key 顺序一致；通过 `key_index_mapping_[shard][i] → original_index` 写回 `values[original_index]`。
4. **与单机 API 返回值约定一致**：例如 Put 期望子客户端返回 `1` 表示成功（与单机 `GRPCParameterClient::PutParameter` 约定一致），否则分布式 Put 报失败。


---


## 7. 测试方法

### 7.1 C++ 集成测试（CTest）

在已配置并编译的 `build/` 目录下，下列目标依赖**本机已有 PS 进程**在测试配置的地址/端口上监听（测试源码里多为 `127.0.0.1:15000` 或双端口 15000/15001 等，与 `grpc_ps_client_test.cpp`、`dist_grpc_ps_client_test.cpp`、`brpc_ps_client_test.cpp`、`dist_brpc_ps_client_test.cpp` 一致）：

```bash
cd build
ctest -R grpc_ps_client_test -VV
ctest -R dist_grpc_ps_client_test -VV
ctest -R brpc_ps_client_test -VV
ctest -R dist_brpc_ps_client_test -VV
```



## 8. 文档与代码索引

| 内容 | 位置 |
|------|------|
| PS 侧缓存与 KV 封装 | `ps/base/cache_ps_impl.h` |
| gRPC proto | `proto/ps.proto` |
| bRPC proto | `proto/ps_brpc.proto` |
| gRPC 服务端 | `grpc/grpc_ps_server.cpp` |
| bRPC 服务端 | `brpc/brpc_ps_server.cpp` |
| gRPC 单机客户端 | `grpc/grpc_ps_client.*` |
| gRPC 分布式客户端 | `grpc/dist_grpc_ps_client.*` |
| bRPC 单机客户端 | `brpc/brpc_ps_client.*` |
| bRPC 分布式客户端 | `brpc/dist_brpc_ps_client.*` |
| 进程入口 | `ps_server.cpp` |
