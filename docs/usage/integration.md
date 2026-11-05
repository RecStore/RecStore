# DLRM / TorchRec 集成

本页给出用编译好的 `recstore` Python 包接入模型的最小路径；完整训练参考 `model_zoo/rs_demo`。

## 1. 引入 Python 包

从源码安装（`pip install .` 走 scikit-build-core，默认在 `build/` 里编译 `recstore_torch_ops` 扩展）：

```bash
python3 -m pip install .
```

开发时也可以不安装，直接把包目录挂进 `PYTHONPATH`：

```bash
export PYTHONPATH="$PWD/src/python/pytorch:${PYTHONPATH:-}"
python3 -c "from recstore.KVClient import RecStoreClient; print(RecStoreClient)"
```

`import recstore` 会自动加载 `_recstore_ops.so`，查找顺序见 [RecStoreClient](../api/python.md)。

## 2. 最小可用示例

先按[快速运行](../quickrun.md)起一个参数服务器，然后：

```python
import torch
from recstore.KVClient import RecStoreClient

client = RecStoreClient(role="trainer")
client.set_ps_config("127.0.0.1", 15000)  # 与配置里 cache_ps.servers 的第一项一致

name, dim = "emb_table", 128
client.init_data(name, shape=(200_000, dim), dtype=torch.float32)

ids = torch.tensor([1, 7, 42], dtype=torch.int64)
values = client.pull(name, ids)                       # [3, dim]
client.push(name, ids, values + 0.5)                  # 写回
client.update(name, ids, torch.randn(3, dim) * 0.01)  # sparse gradient
```

多分片时不要逐个手填服务端，直接按[分布式训练](../dev/distributed.md)用同一份 `distributed_client` 配置启动。

## 3. 替换模型里的 Embedding

- EmbeddingBag：`RecStoreEmbeddingBagCollection`（当前推荐）。
- EmbeddingCollection：`RecStoreEmbeddingCollection`，需要注意 fused 键空间。
- 单表实验：`DistEmbedding`。

构造参数与融合开关见 [Embedding 模块](../api/embedding.md)；sparse update 的语义与 CALC 接口见 [CALC 接口](../calc/interfaces.md)。

## 4. 接入顺序

1. 按[快速运行](../quickrun.md)跑通 DLRM / rs_demo。
2. 把模型里的 Embedding 换成上面三个模块之一。
3. 多机场景按[分布式训练](../dev/distributed.md)对齐 `distributed_client`。
4. 需要改 sparse update 语义时再看 [CALC 接口](../calc/interfaces.md)。

## 5. 示例

仓库暂时没有独立的集成 demo，`model_zoo/rs_demo` 与 `model_zoo/torchrec_dlrm` 是当前可运行的参考实现。后续会补一个只依赖 `recstore` 包的最小训练循环，放在 `examples/` 下。
