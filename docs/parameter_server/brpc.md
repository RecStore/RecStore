# bRPC 模块

## 概述

bRPC 模块基于 apache 开源的 bRPC 框架实现。相比 gRPC，bRPC 在同数据中心环境下提供更低的延迟和更高的吞吐量。

## 与 gRPC 的差异

| 方面 | gRPC | bRPC |
|------|------|------|
| 协议 | HTTP/2 | 自定义二进制协议 |
| 序列化 | Protobuf | Protobuf |
| 连接管理 | HTTP/2 多路复用 | TCP 连接池 |
| 零拷贝 | 部分 | 完全 |

## 服务器实现

### BRPCParameterServiceImpl

bRPC 服务实现，接口与 gRPC 版本类似。

实现的 RPC 方法：

| 方法 | 说明 |
|------|------|
| GetParameter | 获取参数值 |
| PutParameter | 设置参数初值 |
| UpdateParameter | 更新参数 |
| InitEmbeddingTable | 初始化嵌入表 |

## 客户端实现

### BRPCParameterClient

单服务器客户端，使用 brpc::Channel 通信。

| 成员 | 类型 | 说明 |
|------|------|------|
| channel_ | brpc::Channel | bRPC 通道 |
| stub_ | ParameterService_Stub | 服务 Stub |

### DistributedBRPCParameterClient

分布式客户端，支持多服务器分片。

## 配置

```json
{
  "cache_ps": {
    "ps_type": "BRPC",
    "servers": [
      { "host": "192.168.1.1", "port": 50051, "shard": 0 }
    ]
  }
}
```

## 文件组织

| 文件 | 说明 |
|------|------|
| `brpc_ps_server.h` / `.cpp` | bRPC 服务器实现 |
| `brpc_ps_client.h` / `.cpp` | 单服务器客户端 |
| `dist_brpc_ps_client.h` / `.cpp` | 分布式客户端 |
| `*_test.cpp` | 单元测试 |
