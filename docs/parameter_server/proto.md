# Proto 模块

## 概述

Proto 模块包含 Protocol Buffer 定义文件，定义了参数服务器的通信协议。

## ps.proto

### 消息定义

下表字段名与 `ps.proto` 逐字一致，包括历史拼写（`PutParameterResponse.sucess`）。

| 消息 | 字段 | 说明 |
|------|------|------|
| GetParameterRequest | keys (bytes) | 参数键列表 |
| | model_name (bytes) | 模型名称（可选）|
| | perf (bool) | 性能统计标志（可选）|
| GetParameterResponse | parameter_value (bytes) | 参数值 |
| | keys (bytes) | 参数键 |
| PutParameterRequest | parameter_value (bytes) | 参数值 |
| PutParameterResponse | sucess (bool) | 操作成功标志；proto 里就拼作 `sucess`，改字段名会影响所有已生成的客户端，暂不修 |
| UpdateParameterRequest | table_name (string) | 表名 |
| | gradients (bytes) | 梯度数据 |
| UpdateParameterResponse | success (bool) | 操作成功标志 |
| InitEmbeddingTableRequest | table_name (string) | 表名 |
| | config_payload (bytes) | 配置数据 |
| InitEmbeddingTableResponse | success (bool) | 操作成功标志 |
| | tag (int32) | 建表返回的非负 tag |
| SparseGradientCommitRequest | table_name (string) | 表名 |
| | payload (bytes) | 编码后的 sparse gradient |
| SparseGradientCommitResponse | committed_step (int64) | 提交后的 step |
| | error (string) | 失败原因，空表示成功 |
| CommandRequest | command (PSCommand) | 控制命令 |
| | arg1, arg2, arg3 (repeated bytes) | 参数 |
| CommandResponse | reply (string) | 响应消息 |

### PSCommand 枚举

| 值 | 说明 |
|-----|------|
| CLEAR_PS | 清空参数服务器 |
| RELOAD_PS | 重新加载参数 |
| LOAD_FAKE_DATA | 加载测试数据 |
| DUMP_FAKE_DATA | 导出测试数据 |
| SAVE_CHECKPOINT | 保存 checkpoint |
| LOAD_CHECKPOINT | 加载 checkpoint |

### 服务定义

```protobuf
service ParameterService {
  rpc SparseGradientCommit(SparseGradientCommitRequest) returns (SparseGradientCommitResponse);
  rpc GetParameter(GetParameterRequest) returns (GetParameterResponse);
  rpc Command(CommandRequest) returns (CommandResponse);
  rpc PutParameter(PutParameterRequest) returns (PutParameterResponse);
  rpc UpdateParameter(UpdateParameterRequest) returns (UpdateParameterResponse);
  rpc InitEmbeddingTable(InitEmbeddingTableRequest) returns (InitEmbeddingTableResponse);
}
```

## ps_brpc.proto

bRPC 协议定义文件，结构与 ps.proto 类似，针对 bRPC 框架进行优化。

## 代码生成

### C++

```bash
protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=grpc_cpp_plugin ps.proto
```

生成文件：
- `ps.pb.h` / `ps.pb.cc` - 消息类
- `ps.grpc.pb.h` / `ps.grpc.pb.cc` - 服务类

### Python

```bash
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ps.proto
```

生成文件：
- `ps_pb2.py` - 消息类
- `ps_pb2_grpc.py` - 服务类
