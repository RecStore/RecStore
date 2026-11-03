# 配置指南

按部署场景选择传输方式和存储后端：

| 场景 | 入口 |
| --- | --- |
| 单机 / 跨机训练 | [分布式训练配置](../dev/distributed.md) |
| 字段含义与默认值 | [配置字段参考](../config.md) |
| 存储引擎与 index/value 组合 | [KVEngineComposite](../storage/composite_kvengine.md) |
| RDMA 设备、GID 与 QP | [RDMA 参数](../parameter_server/rdma.md) |
| 命令行与环境变量 | [命令行与环境变量](../api/cli.md) |

配置文件是单个 JSON，顶层为 `cache_ps`、`distributed_client`、`client`；`report_API`、
`hugectr`、`hierkv` 按需添加。
