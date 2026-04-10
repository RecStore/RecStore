# PS Client 文档

## 概述

PS Client 提供两种实现，接口均继承自 `recstore::BasePSClient`：

| 类 | 文件 | 适用场景 |
|---|---|---|
| `GRPCParameterClient` | `grpc/grpc_ps_client.h` | 连接单个 PS 节点 |
| `DistributedGRPCParameterClient` | `grpc/dist_grpc_ps_client.h` | 连接多个分片 PS 节点 |

---

## GRPCParameterClient

### 构造

```cpp
// 推荐：通过 JSON 配置
explicit GRPCParameterClient(json config);
// config 示例：{"host": "localhost", "port": 15000, "shard": 0}

// 兼容写法
explicit GRPCParameterClient(const std::string& host, int port, int shard);
```

### 参数读取

```cpp
// 同步读取，values 为平铺 float 数组
bool GetParameter(const base::ConstArray<uint64_t>& keys, float* values);

// 同步读取，values 为每个 key 对应的 embedding vector
bool GetParameter(const base::ConstArray<uint64_t>& keys,
                  std::vector<std::vector<float>>* values);
```

大请求会自动按 `MAX_PARAMETER_BATCH`（2000）分批，通过 gRPC CompletionQueue 异步发送后聚合结果。

### 参数写入

```cpp
// 写入参数
bool PutParameter(const std::vector<uint64_t>& keys,
                  const std::vector<std::vector<float>>& values);

int PutParameter(const base::ConstArray<uint64_t>& keys,
                 const std::vector<std::vector<float>>& values);
```

### 梯度更新

```cpp
// values 为 vector of vector
bool UpdateParameter(const std::string& table_name,
                     const base::ConstArray<uint64_t>& keys,
                     const std::vector<std::vector<float>>& grads);

// grads 为平铺数组
bool UpdateParameterFlat(const std::string& table_name,
                         const base::ConstArray<uint64_t>& keys,
                         const float* grads,
                         int num_rows,
                         int embedding_dim);
```

### 异步预取（Prefetch）

```cpp
// 发起异步 Get，立即返回 prefetch_id
uint64_t PrefetchParameter(const base::ConstArray<uint64_t>& keys);

bool     IsPrefetchDone(uint64_t prefetch_id);   // 非阻塞查询
void     WaitForPrefetch(uint64_t prefetch_id);  // 阻塞等待

// 取回结果（vector of vector）
bool GetPrefetchResult(uint64_t prefetch_id,
                       std::vector<std::vector<float>>* values);

// 取回结果（平铺数组）
bool GetPrefetchResultFlat(uint64_t prefetch_id,
                           float* values,
                           int* num_rows,
                           int* embedding_dim);
```

### 异步写入（EmbWriteAsync）

```cpp
uint64_t EmbWriteAsync(const base::ConstArray<uint64_t>& keys,
                       const std::vector<std::vector<float>>& values);

bool IsWriteDone(uint64_t write_id);
void WaitForWrite(uint64_t write_id);
```

### Embedding Table 初始化

```cpp
bool InitEmbeddingTable(const std::string& table_name,
                        const nlohmann::json& config);
// config 示例：{"num_embeddings": 1000000, "embedding_dim": 128}
```

### 控制命令

```cpp
void Command(recstore::PSCommand command);  // 统一命令入口

bool ClearPS();                             // 清空 PS 数据
bool LoadCkpt(const std::vector<std::string>& model_config_paths,
              const std::vector<std::string>& emb_file_paths);
```

## benchmark

### LoadFakeData

从 server 读取 n 字节随机数据，用于测试**下行带宽**（server → client）。

```cpp
bool LoadFakeData(int64_t n);
```

server 直接生成随机数据返回，不访问真实存储。

### DumpFakeData

向 server 写入 n 字节随机数据，用于测试**上行带宽**（client → server）。

```cpp
bool DumpFakeData(int64_t n);
```

client 本地生成 n 字节数据通过 gRPC 发送，server 接收后直接丢弃，不访问真实存储。

### grpc_ps_bench

网络吞吐基准工具，源文件 `grpc/grpc_ps_bench.cpp`。

**编译**

```bash
cmake --build . --target grpc_ps_bench
```

**运行**

```bash
./grpc_ps_bench [--host 127.0.0.1] [--port 15000] [--warmup 10] [--iters 100]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | PS server 地址 |
| `--port` | `15000` | PS server 端口 |
| `--warmup` | `10` | 每个 size 预热轮数（不计时） |
| `--iters` | `100` | 每个 size 正式测量轮数 |

依次测试 4 KB / 64 KB / 512 KB / 4 MB 四个 payload，分别跑 `LoadFakeData` 和 `DumpFakeData`，输出每个组合的 avg / p50 / p99 延迟（us）和吞吐（MB/s）。

**输出示例**

```
PS Network Benchmark  server=127.0.0.1:15000  warmup=10  iters=100

LoadFakeData  (download: server -> client)
  [load]      4 KB  avg=    312.5 us  p50=    308.1 us  p99=    421.3 us  tput=    12.4 MB/s
  [load]     64 KB  avg=    540.2 us  p50=    535.0 us  p99=    620.8 us  tput=   114.6 MB/s
  [load]    512 KB  avg=   2103.7 us  p50=   2088.4 us  p99=   2340.1 us  tput=   235.8 MB/s
  [load]   4096 KB  avg=  15821.0 us  p50=  15700.3 us  p99=  17002.5 us  tput=   250.1 MB/s

DumpFakeData  (upload: client -> server)
  [dump]      4 KB  avg=    289.2 us  p50=    285.0 us  p99=    390.7 us  tput=    13.4 MB/s
  ...
```

---

## DistributedGRPCParameterClient

### 构造

```cpp
explicit DistributedGRPCParameterClient(json config);
```

配置示例：

```json
{
  "distributed_client": {
    "num_shards": 3,
    "hash_method": "city_hash",
    "max_keys_per_request": 500,
    "servers": [
      {"host": "10.0.0.1", "port": 15000, "shard": 0},
      {"host": "10.0.0.2", "port": 15000, "shard": 1},
      {"host": "10.0.0.3", "port": 15000, "shard": 2}
    ]
  }
}
```

`hash_method` 支持 `"city_hash"`（默认）和 `"simple_mod"`。

### 路由策略

```
shard_id = hash(key) % num_shards
```

Get/Put/Update 均自动将 keys 按 shard 分组，并行发送至各 shard，最终按原始 key 顺序聚合结果。

### 接口

与 `GRPCParameterClient` 相同，额外提供：

```cpp
int shard_count();  // 返回 shard 数量
```

广播命令（`ClearPS`、`LoadFakeData`、`DumpFakeData`、`LoadCkpt`、`InitEmbeddingTable`）会异步发送至所有 shard，全部成功才返回 `true`。

> 注意：`PrefetchParameter` 在分布式客户端中**未实现**，调用会返回错误。

---

## RPC 协议

Proto 定义见 `proto/ps.proto`。

```protobuf
service ParameterService {
  rpc GetParameter        (GetParameterRequest)        returns (GetParameterResponse);
  rpc Command             (CommandRequest)              returns (CommandResponse);
  rpc PutParameter        (PutParameterRequest)         returns (PutParameterResponse);
  rpc UpdateParameter     (UpdateParameterRequest)      returns (UpdateParameterResponse);
  rpc InitEmbeddingTable  (InitEmbeddingTableRequest)   returns (InitEmbeddingTableResponse);
}
```

`CommandRequest` 使用 `repeated bytes arg1/arg2/arg3` 传递可变参数。

---

## case test

```cpp
// 单节点
json config = {{"host", "localhost"}, {"port", 15000}, {"shard", 0}};
auto client = std::make_unique<GRPCParameterClient>(config);

// 同步 Get
base::ConstArray<uint64_t> keys(key_ptr, key_count);
std::vector<std::vector<float>> values;
client->GetParameter(keys, &values);

// 异步预取
uint64_t id = client->PrefetchParameter(keys);
client->WaitForPrefetch(id);
client->GetPrefetchResult(id, &values);

// 网络基准测试
client->LoadFakeData(4 * 1024 * 1024);  // 下行 4MB
client->DumpFakeData(4 * 1024 * 1024);  // 上行 4MB

// 多分片
json dist_config = { /* 见上方示例 */ };
auto dist_client = std::make_unique<DistributedGRPCParameterClient>(dist_config);
dist_client->GetParameter(keys, &values);  // 自动分片路由
```
