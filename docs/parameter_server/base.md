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
