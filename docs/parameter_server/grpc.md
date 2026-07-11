# gRPC 模块

## 概述

gRPC 模块基于 Google 的 gRPC 框架实现参数服务器的网络通信。

## 服务器实现

### ParameterServiceImpl

gRPC 服务实现，处理客户端的参数请求。

实现的 RPC 方法：

| 方法 | 请求类型 | 响应类型 | 说明 |
|------|---------|---------|------|
| GetParameter | GetParameterRequest | GetParameterResponse | 获取参数值 |
| PutParameter | PutParameterRequest | PutParameterResponse | 设置参数初值 |
| UpdateParameter | UpdateParameterRequest | UpdateParameterResponse | 更新参数 |
| InitEmbeddingTable | InitEmbeddingTableRequest | InitEmbeddingTableResponse | 初始化嵌入表 |
| Command | CommandRequest | CommandResponse | 执行控制命令 |

## 客户端实现

### GRPCPSClient

单服务器客户端，继承自 `BasePSClient`，使用 gRPC Channel 与服务器通信。

关键成员：

| 成员 | 类型 | 说明 |
|------|------|------|
| channel_ | shared_ptr<Channel> | gRPC 通道 |
| stub_ | unique_ptr<Stub> | 服务 Stub |

### DistGRPCPSClient

分布式客户端，支持连接多个服务器进行参数分片。

| 成员 | 说明 |
|------|------|
| clients_ | 多个 GRPCPSClient 实例 |
| 分片策略 | 根据参数 key 哈希到不同服务器 |

### 预取机制

通过 `PrefetchBatch` 结构实现异步预取：

| 字段 | 类型 | 说明 |
|------|------|------|
| requests_ | vector<GetParameterRequest> | 请求列表 |
| responses_ | vector<GetParameterResponse> | 响应列表 |
| response_readers_ | vector<AsyncResponseReader> | 异步读取器 |
| cqs_ | CompletionQueue | gRPC 完成队列 |
| completed_count_ | int | 已完成计数 |

## 配置

```json
{
  "backend": "grpc",
  "server_addresses": ["192.168.1.1:50051"]
}
```

## 文件组织

| 文件 | 说明 |
|------|------|
| `grpc_ps_server.h` / `.cpp` | gRPC 服务器实现 |
| `grpc_ps_client.h` / `.cpp` | 单服务器客户端 |
| `dist_grpc_ps_client.h` / `.cpp` | 分布式客户端 |
| `*_test.cpp` | 单元测试 |
