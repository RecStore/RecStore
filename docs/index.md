# RecStore 文档

RecStore 面向推荐模型，提供分布式参数存储与 Embedding 计算支持。

<div class="grid cards" markdown>

-   **使用**

    安装环境、运行模型、配置分布式训练。

    [快速运行](quickrun.md) · [安装与构建](quickstart.md)

-   **开发**

    系统架构、功能扩展与正确性测试。

    [开发指南](dev/index.md)

-   **性能**

    端到端、参数服务器、存储引擎基准与性能分析。

    [性能指南](performance/index.md)

-   **API**

    Python、PyTorch、C++ 接口与配置字段。

    [API 参考](api/index.md)

</div>

## 开始使用

1. [准备 Docker 开发镜像](dev/zero_config_run.md)，或[从源码构建](quickstart.md)。
2. 按[快速运行](quickrun.md)启动参数服务器和 DLRM。
3. 继续阅读[分布式训练](dev/distributed.md)或[配置指南](usage/configuration.md)。

运行问题见[常见问题与排障](faq.md)。
