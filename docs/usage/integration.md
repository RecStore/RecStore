# DLRM / TorchRec 集成

接入顺序建议：

1. 先按[快速运行](../quickrun.md)跑通 DLRM / rs_demo；
2. Embedding 模块的选择与构造见[Embedding 模块](../api/embedding.md)；
3. 多机与 TorchRec 对齐见[分布式训练与 TorchRec 对齐](../dev/distributed.md)；
4. 需要改优化的 sparse update 语义时看[CALC 接口](../calc/interfaces.md)。

当前模型接入优先使用 `RecStoreEmbeddingBagCollection`；单表实验可用 `DistEmbedding`。
