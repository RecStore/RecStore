# 测试用工具

## ps_server_helpers.py

`src/test/scripts/ps_server_helpers.py` 模块提供了一系列辅助函数，用于在测试环境中查找、检测和配置 `ps_server`。它通常与 `ps_server_runner.py` 配合使用，用于自动化测试流程。

| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PS_SERVER_PATH` | 指定 `ps_server` 的绝对路径 | 自动查找 |
| `RECSTORE_CONFIG` | 指定 `recstore_config.json` 路径 | None |
| `PS_LOG_DIR` | 指定日志输出目录 | `./logs` |
| `PS_TIMEOUT` | 启动超时时间（秒） | 60 |
| `PS_NUM_SHARDS` | 期望的分片数量 | 2 |
| `NO_PS_SERVER` | 强制跳过启动服务器 (Set to '1' or 'true') | False |

### `find_ps_server_binary()`

在 `build/bin/ps_server` 等常见构建目录中查找可执行文件。

| | 说明 |
| :--- | :--- |
| **返回值** | `ps_server` 的绝对路径。 |
| **查找顺序** | `OS ENV(PS_SERVER_PATH)` -> `./bin` -> `./build/bin` -> 上级目录的构建路径。 |

### `check_ps_server_running(ports=None)`

检查 Parameter Server 的默认端口（或指定端口）是否已在监听。

| | 说明 |
| :--- | :--- |
| **参数** | `ports` (list, optional): 要检查的端口列表。默认为 `[15000, 15001, 15002, 15003]`。 |
| **返回值** | `(is_running: bool, open_ports: list)` |

### `should_skip_server_start()`

判断当前测试是否应该跳过启动 `ps_server` 的步骤。判断逻辑:

- 检查环境变量 `CI` 或 `GITHUB_ACTIONS`。
- 检查环境变量 `NO_PS_SERVER`。
- 检查端口是否已被占用（意味着服务已经在运行）。

返回值: `(skip: bool, reason: str)`

### `get_server_config()`

获取标准化的服务器配置字典，优先读取环境变量。

返回值字典:

| 键名 | 说明 |
| :--- | :--- |
| `server_path` | 二进制文件路径 |
| `config_path` | 配置文件路径 (`RECSTORE_CONFIG`) |
| `log_dir` | 日志目录 (`PS_LOG_DIR`) |
| `timeout` | 超时时间 (`PS_TIMEOUT`) |
| `num_shards` | 分片数量 (`PS_NUM_SHARDS`) |

## ps_server_launcher (C++)

`src/test/server_mgr/ps_server_launcher.h` 与 `src/test/server_mgr/ps_server_launcher.cpp` 提供了 C++ 侧的 `ps_server` 启停能力，适合 C++ 单测与工具程序复用。

该模块与 Python 侧环境变量语义保持一致（便于混合测试场景复用同一套环境配置）。

| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `PS_SERVER_PATH` | 指定 `ps_server` 的绝对路径 | 自动查找 |
| `RECSTORE_CONFIG` | 指定 `recstore_config.json` 路径 | 自动查找 |
| `PS_LOG_DIR` | 指定日志输出目录 | `./logs` |
| `PS_TIMEOUT` | 启动超时时间（秒） | 60 |
| `PS_NUM_SHARDS` | 期望的分片数量 | 2 |
| `PS_SERVER_PS_TYPE` | 启动前临时覆盖 `cache_ps.ps_type`（如 `GRPC`/`BRPC`） | 不覆盖 |
| `PS_SERVER_PORTS` | 启动前临时覆盖服务端口，逗号分隔（如 `15123,15124`） | 不覆盖 |
| `NO_PS_SERVER` | 强制跳过启动服务器 | False |

### 主要能力

- 端口探活与启动决策（支持“部分端口打开”直接判错）。
- 基于日志的 shard ready 检测。
- 进程生命周期管理（优雅停止 + 超时强杀）。
- RAII 封装（`ScopedPSServer`）。

### 关键类型

| 类型 | 说明 |
| :--- | :--- |
| `LauncherOptions` | 启动参数（路径、超时、分片数等） |
| `LaunchDecision` | 启动决策结果（是否启动、失败原因、端口状态） |
| `LaunchResult` | 启动结果（PID、ready 分片、日志路径） |
| `PSServerLauncher` | 启停与状态查询主类 |
| `ScopedPSServer` | RAII 封装，作用域结束自动停止 |

### CMake 目标

- 库目标: `ps_server_mgr`
- 测试目标: `test_ps_server_launcher`

### 最小用法

```cpp
#include "ps_server_launcher.h"

using namespace recstore::test;

void RunTestCase() {
  LauncherOptions opts = PSServerLauncher::LoadOptionsFromEnvironment();
  ScopedPSServer server(opts, true);

  // test logic here
}
```

### 协议配置建议

- BRPC 客户端测试默认可使用仓库根目录的 `recstore_config.json`。
- gRPC 客户端测试建议在启动器里设置 `LauncherOptions.override_ps_type = "GRPC"`，必要时配合 `LauncherOptions.override_ports`（或 `PS_SERVER_PORTS`）使用独立端口，避免复用已占用端口导致协议不匹配。

### 目录说明

测试模块目录已统一为 `src/test/server_mgr`（原先较长命名 `server_management` 已替换）。

## analyze_embupdate_stages.py

`src/test/scripts/analyze_embupdate_stages.py` 用于分析 update 链路的分阶段性能数据。  
它会从本地 `REPORT_LOCAL_EVENT` 日志或 JSONL 事件文件中提取 `embupdate_stages` 表数据，并输出:

- 各阶段指标统计（mean/p50/p95/p99/max）
- 近似拆分（序列化、后端执行、网络/框架开销）
- 慢请求 TopN（按 trace 维度）

### 输入来源

支持两种输入格式:

1. glog 文本日志（包含 `REPORT_LOCAL_EVENT {...}` 行）
2. 纯 JSONL（每行一个事件 JSON）

### 常用命令

```bash
python3 src/test/scripts/analyze_embupdate_stages.py --input /path/to/recstore.log
```

```bash
python3 src/test/scripts/analyze_embupdate_stages.py \
  --input /path/to/server.log \
  --input /path/to/client.log \
  --top 20
```

```bash
python3 src/test/scripts/analyze_embupdate_stages.py \
  --input /path/to/report_events.jsonl \
  --trace-prefix grpc_client::EmbUpdate
```

```bash
python3 src/test/scripts/analyze_embupdate_stages.py \
  --input /path/to/report_events.jsonl \
  --group-by-prefix \
  --export-csv /tmp/embupdate_report.csv
```

### 关键指标解读

- `client_serialize_us`: 客户端将梯度打包成请求的耗时。
- `client_rpc_us`: 客户端从发起 RPC 到返回的耗时（包含网络与服务端处理）。
- `server_total_us`: 服务端 `UpdateParameter` 总耗时。
- `server_backend_update_us`: 服务端后端更新逻辑（cache/storage/update）执行耗时。
- `op_total_us`: op 层（`EmbUpdate`）总耗时。

近似网络/框架开销:

`client_rpc_us - server_total_us`

这有助于快速判断瓶颈更偏网络侧还是服务端执行侧。
