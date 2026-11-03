# 性能埋点

RecStore 的 Grafana 看板基于 `report()` 函数上报链路，完整链路为：

`RecStore (C++/Python 调用 report)` -> `HTTP report_API` -> `ClickHouse` -> `Grafana`

## 上报协议与入口

- 上报接口由 `recstore_config.json` 中 `report_API` 指定（默认：`http://127.0.0.1:8081/report`）。
- `report()` 会构造 JSON 并异步发送，核心字段如下：


| 字段名            | 含义           | 示例                                     |
| -------------- | ------------ | -------------------------------------- |
| `table_name`   | 目标报表/逻辑表名    | `op_latency`                           |
| `unique_id`    | 单次请求或链路标识    | `embread_debug` + 时间戳 |
| `metric_name`  | 指标名称         | `recstore_us`                          |
| `metric_value` | 指标数值（double） | `532.0`                                |


对应实现可参考：

- `src/base/report/report_client.cpp`
- `src/base/report/report_client.h`

## 常见 report 表与指标


| 表名                   | 常见指标名                                                                    | 说明                |
| -------------------- | ------------------------------------------------------------------------ | ----------------- |
| `op_latency`         | `recstore_us`、`recserver_us`                                             | 端到端或服务端处理耗时       |
| `embread_stages`     | `duration_us`、`request_size`、`cache_lookup_us`、`deserialize_duration_us` | EmbRead 细分阶段耗时与规模 |
| `ps_client_latency`  | `latency_us`                                                             | PS 客户端侧延迟         |
| `ps_server_latency`  | `latency_us`                                                             | PS 服务端侧延迟         |
| `emb_read_flame_map` | `level`、`value`、`self`、`start`                                           | 火焰图分层结构数据         |

??? note "待重写：旧版 txt 埋点"

    RecStore 使用 `xmh::Timer`（位于 `src/base/timer.h`）进行全链路的耗时打点。数据流向如下：

    ### 写路径自顶向下

    | 层级 (Layer) | Timer 名称 | 说明 | 代码位置 |
    | :--- | :--- | :--- | :--- |
    | **PyTorch OP** | `OP.EmbWrite.Total` | Python 端调用 C++ OP 的总耗时 | `src/framework/pytorch/op_torch.cc` |
    | | `OP.EmbWrite.Call` | 核心逻辑调用耗时 | |
    | **C++ Client** | `ClientOp.EmbWrite.Total` | 客户端操作总耗时 | `src/framework/op.cc` |
    | | `ClientOp.EmbWrite.BuildVector` | Tensor 转 C++ Vector 的开销 | |
    | **gRPC Client** | `Client.PutParameter.Total` | RPC 请求总耗时 | `src/ps/grpc/dist_grpc_ps_client.cpp` |
    | | `Client.PutParameter.Serialize` | 序列化 Data Request 耗时 | |
    | | `Client.PutParameter.RPC` | 网络传输 + 服务端处理总耗时 | |
    | **Server** | `PS.PutParameter.Handle` | 服务端处理总耗时 | `src/ps/grpc/grpc_ps_server.cpp` |
    | | `PS.PutParameter.KVPutAll` | 写入底层 KV 存储的耗时 | |

    ### 读路径自顶向下

    | 层级 (Layer) | Timer 名称 | 说明 | 代码位置 |
    | :--- | :--- | :--- | :--- |
    | **PyTorch OP** | `OP.EmbRead.Total` | Python 端调用 C++ OP 的总耗时 | `src/framework/pytorch/op_torch.cc` |
    | | `OP.EmbRead.ToCPUKeys` | Embedding Key 从 GPU 拷贝到 CPU 的耗时 | |
    | **gRPC Client** | `Client.GetParameter.Total` | RPC 请求总耗时 | `src/ps/grpc/dist_grpc_ps_client.cpp` |
    | | `Client.GetParameter.AsyncWait` | 异步等待网络回包的耗时（反映服务端处理+网络延迟） | |
    | **Server** | `KV BatchGet` | 底层 KV 引擎批量读取耗时 | `docs/storage/composite_kvengine.md` |

    ??? tip "什么是 Timer?"
        `xmh::Timer` 是一个高性能的 C++ 计时器工具类。它通过 `Timer::Start("Name")` 和 `Timer::Stop("Name")` 记录代码块耗时，并自动统计 P50、P99 等分位数值，定期输出到日志文件中。
