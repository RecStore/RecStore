# BagPipe 缓存

原论文：Saurabh Agarwal、Chengpo Yan、Ziyi Zhang、Shivaram Venkataraman，[BagPipe: Accelerating Deep Recommendation Model Training](https://doi.org/10.1145/3600006.3613142)，SOSP 2023，348–363。

BagPipe 在训练端缓存 embedding，并提前从参数服务器（PS）预取后续批次需要的行。PS 保存完整参数并执行 SGD，GPU 缓存随每步提交结果更新。

## 启用

在现有训练命令中加入：

```sh
--ps-type RDMA --read-mode bagpipe --optimization-prescan \
  --optimization-cache-capacity 8192 --bagpipe-aggregation ps
```

默认逐批交换各 rank 的 ID，在线生成访问计划。`--optimization-prescan` 在训练前扫描同一批次序列，一次性交换 ID，并把计划保存在 CPU。`rs_demo` 的单机部署由启动器选择本地后端。

训练使用融合 embedding 表、FP32 稀疏梯度和固定学习率 SGD，客户端与 PS 使用相同学习率；每个样本的每个特征对应一个 ID。

| `--bagpipe-aggregation` | 梯度聚合方式 |
| --- | --- |
| `ps`（默认） | 各 rank 提交本地梯度，PS 汇总并除以 rank 数，再执行 SGD |
| `collective` | 训练端 all-reduce 后除以 rank 数，每行由指定 rank 提交给 PS |

## 每步流程

### 1. 准备：逐行决定是否预取

本批 ID 去重后，记录每行在所有 rank 中上次参与更新的步骤，依次判断：

| 行的类别 | 条件 | 处理 |
| --- | --- | --- |
| 等待更新的行 | 上次更新尚未确认提交 | 记录依赖，前向时等提交完成，再查缓存 |
| 可复用的行 | 依赖已提交，且已在缓存中 | 前向时查缓存取值 |
| 待预取的行 | 依赖已提交，且未在缓存中 | 有预取额度时发起 PS 读取，其余留到前向按需读取 |

### 2. 前向：取值与缓存替换

完成依赖等待后，每行依次从 GPU 缓存、预取结果、PS 按需查询中取值。命中的行移到 LRU 队尾；补齐的缺失行写入缓存时，优先使用空槽，缓存满时淘汰队首的最久未使用行并复用其槽位。其余缓存行保留。

### 3. 反向与提交

合并本地重复 ID 的梯度，按所选方式聚合。PS 收齐各 rank 的分块和步骤标记后执行更新，全部 shard 更新后，本步提交完成。

### 4. 刷新：逐行更新或失效

确认提交完成后，将仍在缓存中的行按本步访问情况分为四类：

| 行的类别 | 本步由谁访问 | `ps` 模式 | `collective` 模式 |
| --- | --- | --- | --- |
| 本地独占行 | 只有本 rank | 用本地梯度除以 rank 数后更新 | 用全局平均梯度更新 |
| 共享行 | 本 rank 和其他 rank | 失效 | 用全局平均梯度更新 |
| 仅远端访问的行 | 只有其他 rank | 失效 | 用全局平均梯度更新 |
| 未访问的行 | 所有 rank 都未访问 | 保留 | 保留 |

失效会移除缓存映射并释放槽位，下次访问时重新取值；LRU 淘汰则在前向补入新行时触发。已被淘汰的行由 PS 保存更新后的参数。

## 例子

以单 rank、缓存充足为例：step 0 访问 `[10, 20]`，提前准备的 step 1 访问 `[20, 30]`，其中 30 尚未缓存。

| 时刻 | 训练与数据读取 |
| --- | --- |
| step 0 计算 | GPU 用 10、20 做前向和反向，同时从 PS 预取 30；30 的值在 step 0 中保持不变 |
| step 0 提交 | PS 更新 10、20；客户端确认提交完成后，用梯度刷新缓存中的 20 |
| step 1 计算 | 20 取自更新后的缓存，30 取自预取结果 |

准备 step 1 时，20 属于等待更新的行，30 属于待预取的行。step 0 提交后，缓存中的 10、20 都作为本地独占行刷新。

## 缓存容量

`--optimization-cache-capacity` 每单位对应 64 行。FP32 embedding 的常驻缓存大小为 `cache_capacity × 64 × embedding_dim × 4` 字节；批次张量和预扫描计划另占空间。缓存替换由 CPU 上的 LRU 元数据管理，PS 的 DRAM/SSD 容量单独配置。

## 代码入口

| 文件（相对 `src/`） | 职责 |
| --- | --- |
| `python/pytorch/recstore/bagpipe_cache/plugin.py` | 接入训练循环 |
| `python/pytorch/recstore/bagpipe_cache/planned.py` | 访问计划、预取和缓存更新 |
| `python/pytorch/recstore/bagpipe_cache/optimizer.py` | 提交稀疏梯度 |
| `optimization/bagpipe/cache_metadata.h` | LRU、缓存准入和更新依赖 |
| `ps/base/sparse_gradient_commit.h` | 梯度聚合与按步提交 |

相关测试：`test_bagpipe_planned.py`、`test_bagpipe_metadata`、`test_framework_op_runtime`。
