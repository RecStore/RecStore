# RecStore 参数服务器模块

## 概述

RecStore 参数服务器（PS）模块负责分布式训练中模型参数和嵌入表的存储、管理和同步。采用分层设计，通过抽象接口支持 gRPC、bRPC 等多种通信协议。

## 架构设计

### 模块组织

| 模块 | 目录 | 说明 |
|------|------|------|
| base | `ps/base` | 抽象接口和基础数据结构 |
| grpc | `ps/grpc` | gRPC 协议实现 |
| brpc | `ps/brpc` | bRPC 协议实现 |
| proto | `ps/proto` | Protocol Buffer 定义 |
| rdma | `ps/rdma` | RDMA 实现（实验性）|

### 核心接口

`BaseParameterServer` 定义 `Init(config)` / `Run()`；`BasePSClient` 定义 `GetParameter`、`PutParameter`、`UpdateParameter`、`InitEmbeddingTable`、`AsyncGetParameter`、`PrefetchParameter` 等操作。完整签名、数据结构和控制命令见 [base.md](./base.md)。

## 工作流程

### 参数操作

1. **GetParameter** - 客户端发送 keys，服务器返回对应的参数值
2. **PutParameter** - 初始化阶段设置参数初值
3. **UpdateParameter** - 训练过程中更新参数（通常是应用梯度）

### 预取机制

```
1. PrefetchParameter(keys) -> prefetch_id
2. [等待网络传输和处理]
3. WaitForPrefetch(prefetch_id) / IsPrefetchDone(prefetch_id)
4. GetPrefetchResult(prefetch_id) -> values
```

## 模块说明

详细文档：

- [base.md](./base.md) - 基础模块
- [grpc.md](./grpc.md) - gRPC 实现
- [brpc.md](./brpc.md) - bRPC 实现  
- [proto.md](./proto.md) - 协议定义
- [rdma.md](./rdma.md) - RDMA 实现

## 配置

参数服务器通过 JSON 配置文件指定后端类型和参数。详见 [config.md](../config.md)。

基本配置结构：

```json
{
  "cache_ps": {
    "ps_type": "GRPC",
    "servers": [
      { "host": "192.168.1.1", "port": 50051, "shard": 0 }
    ]
  }
}
```

`ps_type` 取 `GRPC` 或 `BRPC`；多分片时在 `servers` 里按 `shard` 递增列出。
