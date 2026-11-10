# 混合存储（DRAM+SSD）端到端基准

本页给出 `TIERED_VALUE_STORE` 在双机 RDMA、`ps_pipeline` 读路径下的 SSD 比例扫描结果。
数值只对应当前机器与拓扑，用于同层对比；层级定义与口径纪律见[测试范围与指标](methodology.md)。
本页的比例扫描是 **cache-off 基线**：所有 lane 显式设置 `tiering.cache_policy=none`，数字不含冷热交换，只作对照。
读缓存（默认配置实际会开启）的单独对比见「读缓存」。

## 结论摘要

- **SSD 一碰就掉一半**：只有 16.3% 的访问落到 SSD，端到端吞吐从 37.3K 降到 18.7K samples/s（−50%）；
  54.1% 时 9.7K（−74%）；97% 以上 6.3K（−83%）。
- **读和写两条路都在退化**：`pull`（GET）从 3.3 ms 涨到 41.9 ms，`commit wait`（写回/应用）从 20.6 ms 涨到 116.5 ms；
  49% 以后 commit 的增量大于 pull。
- **“行占比”不等于“访问占比”**：每个特征每个样本各查一次，小表行少但访问量与 200K 大表相同；
  当前“先到先得”的放置让小表先占 DRAM，所以 32% 的行在 SSD 时只有 21.9% 的访问走 SSD。
- **本页是 cache-off 基线，不是默认配置**：`HybridValueStore` 的 `tiering.cache_policy` 库默认值是 `none`，
  但 `rs_demo` / benchmark server 生成的配置模板写的是 `lru`，e2e 这条路径实际会开启读缓存。
  本页 lane 全部显式设回 `none`，所以这里没有冷热交换，数字只作对照；开启后的对比见「读缓存」。
- **启动成本线性增长**：全表 zero-init 使 `embedding_init` 从 3.1 s 涨到 58.7 s（约 0.57 s / 1% 行在 SSD）。

## 测试环境

| 项 | 值 |
| --- | --- |
| 代码 | 上游 `main` @ `5f92ba2a2`（#340 合入后） |
| PS | node190，`build_release`，Release，`USE_PERF_REPORT=ON` |
| Client | node191，A100-PCIE-40GB，同一棵树的 Python + `lib_recstore_ops.so` |
| 传输 | RDMA，200G HDR IB（190 `ibp139s0` / 191 `mlx5_0`），`staging_copy` |
| 读路径 | `ps_pipeline`（无客户端读缓存，读排在全局 SGD commit 之后） |
| Workload | DLRM，batch 1024，embedding dim 128，26 张表，`num_embeddings=200000` |
| 数据 | `model_zoo/torchrec_dlrm/processed_day_0_data`，训练集 65,536 样本 |
| 轮次 | 64 steps（= 一个 epoch），warmup 5，重复 3 次 |

## 比例口径

`rs_demo` 的容量计算：

```text
capacity         = max(sum(num_embeddings_per_feature) * 2, 100_000) = 3,442,396
value_size       = embedding_dim * 4 = 512 B
total rows       = sum(num_embeddings_per_feature) = 1,721,198
DRAM tier bytes  = int(capacity * 512 * multiplier)
usable DRAM rows = floor(0.85 * DRAM tier bytes / 512)
```

`HybridValueStore` 使用高水位 `high_watermark_ratio`（默认 0.85）；`emb_write` 在训练前把每张表的全部行
按表顺序 zero-init 写一遍，因此 tier 划分在训练开始前就固定。

| 口径 | 定义 | 用途 |
| --- | --- | --- |
| **行占比** | `1 - usable_dram_rows / 1,721,198` | 容量规划；由 multiplier 直接决定 |
| **访问占比** | 按数据集访问频次加权（每张表每个样本一次，id 先对表容量取模） | 解释 step 时间；决定实际 SSD 压力 |

两者不同的原因：小表（3、4、10、36 行）只需要很少 DRAM 就能整表留在内存，但它们和大表一样贡献
65,536 次访问。把一张小表挤到 SSD 的吞吐代价与把一张 200K 大表挤下去接近。

## 结果（cache-off 基线）

3 次重复，排除 warmup；`samples/s` 与 `step` 的重复 CV 均 <2%。

下表不含冷热交换（`cache_policy=none`），只是基线参考；默认配置的读缓存效果见「读缓存」。

| 行在 SSD | 访问走 SSD | samples/s | 相对 DRAM | step ms | commit ms | pull ms | init s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0% | 0.0% | **37.3K** | 100% | 27.6 | 20.6 | 3.3 | 3.1 |
| 15.0% | 16.3% | 18.7K | 50% | 55.1 | 34.2 | 16.8 | 11.5 |
| 32.0% | 21.9% | 13.8K | 37% | 74.8 | 44.9 | 25.5 | 21.3 |
| 49.0% | 54.1% | 9.7K | 26% | 105.7 | 68.6 | 32.4 | 31.2 |
| 66.0% | 59.8% | 9.0K | 24% | 114.2 | 76.7 | 32.6 | 40.9 |
| 83.0% | 69.5% | 7.9K | 21% | 130.8 | 86.2 | 39.5 | 50.5 |
| 91.5% | 97.2% | 6.3K | 17% | 162.5 | 114.4 | 42.5 | 55.9 |
| 96.6% | 98.9% | 6.3K | 17% | 164.2 | 116.5 | 41.9 | 58.7 |

### 读缓存（#343 之后）

`tiering.cache_policy` 的库默认值是 `none`，但 `rs_demo` / benchmark server 生成的配置模板写的是 `lru`
（`src/python/pytorch/recstore/benchmark/server.py`），所以 e2e 实际默认开着缓存。本页扫描为了与旧数据可比，
显式设回 `none`；下面用同机 loopback、同样 15% 行在 SSD（16.3% 访问走 SSD）的两轮交错复测给出 cache-on 结果。

每轮 64 steps、排除 5 步 warmup、3 次重复；下表是每次重复的 `step` 均值和中位数指标
（`results/e2e_cache_recheck_11062048/`、`results/e2e_cache_recheck_11071449/`）：

| 轮次 | lane | step ms (r0 / r1 / r2) | step 中位数 | samples/s 中位数 | commit 中位数 | pull 中位数 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 11-06 20:48 | `none` | 59.49 / 56.96 / 61.04 | 59.49 | 18,323 | 32.88 ms | 18.43 ms |
| 11-06 20:48 | `lru`, budget 0.05, min 2 | 50.21 / 51.91 / 51.52 | 51.52 | 21,635 | 29.86 ms | 13.63 ms |
| 11-07 14:49 | `none` | 56.43 / 56.78 / 61.09 | 56.78 | 19,213 | 32.13 ms | 16.61 ms |
| 11-07 14:49 | `lru`, budget 0.05, min 2 | 50.87 / 50.46 / 49.71 | 50.46 | 22,050 | 29.28 ms | 13.42 ms |

| 轮次 | step | samples/s | commit | pull |
| --- | ---: | ---: | ---: | ---: |
| 11-06 20:48 | −13.4% | +18.1% | −3.02 ms | −4.80 ms |
| 11-07 14:49 | −11.1% | +14.8% | −2.85 ms | −3.19 ms |

两轮的 `none` 与 `lru` 区间都不重叠（11-06：56.96–61.04 vs 50.21–51.91；11-07：56.43–61.09 vs 49.71–50.87）。
收益主要落在 `pull`（−4.80 / −3.19 ms），`commit` 只降 2.9–3.0 ms：命中的是读，写回仍然是穿透的。
两轮绝对值受机器当天状态影响（同一 `none` lane 从 59.49 到 56.78 ms），只作轮内对照，不能和上面的双机主表混算。
双机 cache-on 的全比例扫描尚未重跑，主表仍是 cache-off 结果。

### 拆分

`step` 的增量几乎全部被 `ps_pipeline_read_wait` 吸收（占 step 从 76% 升到 95%），
dense / backward / sparse 基本不变。

| 行在 SSD | read wait 占 step | 其中 commit | 其中 pull | sparse | dense+backward+opt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0% | 76% | 20.6 ms | 3.3 ms | 10% | 14% |
| 15.0% | 88% | 34.2 ms | 16.8 ms | 5% | 7% |
| 32.0% | 91% | 44.9 ms | 25.5 ms | 4% | 5% |
| 49.0% | 93% | 68.6 ms | 32.4 ms | 3% | 4% |
| 66.0% | 93% | 76.7 ms | 32.6 ms | 3% | 3% |
| 83.0% | 94% | 86.2 ms | 39.5 ms | 3% | 3% |
| 91.5% | 95% | 114.4 ms | 42.5 ms | 3% | 2% |
| 96.6% | 95% | 116.5 ms | 41.9 ms | 3% | 2% |

`commit` 与 `pull` 是 `ps_pipeline` worker 内部的两个计时，都部分重叠在上一步尾部计算里，
两者相加会略大于 `read wait`，不要当成互斥分片。

相对 DRAM-only 的增量（ms）：

| 访问走 SSD | step | commit | pull |
| ---: | ---: | ---: | ---: |
| 16.3% | +27.5 | +13.5 | +13.5 |
| 21.9% | +47.2 | +24.2 | +22.2 |
| 54.1% | +78.1 | +47.9 | +29.1 |
| 59.8% | +86.6 | +56.0 | +29.3 |
| 69.5% | +103.2 | +65.5 | +36.3 |
| 97.2% | +134.9 | +93.8 | +39.2 |
| 98.9% | +136.6 | +95.8 | +38.6 |

## 判读

**吞吐由访问占比而不是行占比驱动。** 32% 行在 SSD 的 lane 只有 21.9% 访问走 SSD，因为 DRAM 预算
刚好覆盖所有小表和三张 200K 大表。multiplier 0.30 的边界落在第 4 张 200K 大表中间，把它后面的一批
小表整体挤到 SSD，访问占比从 21.9% 跳到 54.1%，step 从 74.8 ms 跳到 105.7 ms。能控制放置时，
应优先保住小表和热点表，而不是简单按行数比例切。

**本页数据是显式 cache-off，默认配置实际会开缓存。** #343 给 `HybridValueStore` 加了可选的 LRU 读缓存：
`tiering.cache_policy=lru` 时先从 `cache_budget_ratio`（默认 0.05）预留 DRAM，SSD 行在读完成且
连续 miss 达到 `promote_min_misses`（默认 2）后提升；`Write` 写穿、`Free` 失效，没有后台线程。
它不改变首次 `Alloc` 的放置，也不做后台冷热迁移。库默认值是 `none`，但 `rs_demo` / benchmark server
的配置模板写的是 `lru`，所以 e2e 这条路径默认是开的；本页 lane 全部显式设回 `none`，上表的放置结论
仍按“先到先得”读。

**写回路径比读路径更早成为瓶颈。** 49% 以后 commit 增量持续大于 pull 增量；SSD 行的更新走
`SsdValueStore` 的 `Write`/`BatchWrite`，在 server 线程里同步完成。批量、异步写回，
或把更新先写 DRAM 再后台刷盘，是值得试的方向。

**超过 90% SSD 后曲线变平。** 97% 与 91% 的 step 几乎一样（164.2 vs 162.5 ms），pull 甚至略降。
SSD 读取在 server 侧按 batch 走 io_uring，SSD 行越多，单次提交的并行度越高，每行的平均等待被摊薄；
这不代表 SSD 不慢，而是延迟被队列深度掩盖，吞吐由 SSD 带宽/队列上限决定。

**启动成本不可忽略。** `KVClient.init_data()` 会 `torch.zeros(shape)` 后 `emb_write` 全表，
172 万行全部写一遍；SSD 行占比每 +1%，init 约 +0.57 s（约 33 µs/行）。对短跑，init 甚至比训练本身还长；
生产上应使用 lazy insert 或 checkpoint 加载，避免全表 zero-init。

## 复现

PS 使用 `PetPSClusterRunner` 启动单 shard `petps_server`；关键配置：

```json
{
  "cache_ps": {
    "ps_type": "RDMA",
    "max_batch_keys_size": 65536,
    "num_threads": 32,
    "num_shards": 1,
    "servers": [{"host": "10.0.2.190", "port": 15000, "shard": 0}],
    "base_kv_config": {
      "capacity": 3442396,
      "index": {"type": "DRAM_PET_HASH"},
      "value": {
        "type": "TIERED_VALUE_STORE",
        "default_value_size_hint": 512,
        "dram_allocator": {
          "type": "PERSIST_LOOP_SLAB",
          "capacity_bytes": "<3,442,396 * 512 * multiplier>",
          "path": "<run>/dram"
        },
        "ssd_allocator": {
          "type": "SSD_SLAB",
          "capacity_bytes": 3580091840,
          "min_block_size": 128,
          "max_block_size": 4096,
          "path": "<run>/ssd.db",
          "io": {"type": "IOURING", "queue_depth": 512, "base_offset_bytes": 4096}
        },
        "tiering": {"cache_policy": "none"}
      }
    }
  },
  "rdma_deployment": {
    "num_clients": 1,
    "nodes": [
      {"node_id": 0, "role": "server", "device": "ibp139s0", "port": 1, "gid_index": 0, "mode": "ib"},
      {"node_id": 1, "role": "client", "device": "mlx5_0",   "port": 1, "gid_index": 0, "mode": "ib"}
    ]
  }
}
```

客户端命令（在 191 上，`RECSTORE_CONFIG` 指向同一份配置的副本）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore --ps-type RDMA \
  --recstore-index-type DRAM_PET_HASH \
  --ps-kv-backend recstore_tiered \
  --batch-size 1024 --embedding-dim 128 --num-embeddings 200000 \
  --steps 64 --warmup-steps 5 \
  --read-mode ps_pipeline --prefetch-depth 0 \
  --dense-arch-layer-sizes 512,256,128 \
  --data-dir model_zoo/torchrec_dlrm/processed_day_0_data \
  --output-root <client-output> --run-id <run_id> \
  --recstore-runtime-dir <client-runtime> \
  --no-start-server --server-host 10.0.2.190 --server-port0 15000 \
  --nnodes 1 --node-rank 0 --nproc-per-node 1 \
  --master-addr 10.0.2.191 --master-port 29501 --rdzv-id <run_id>
```

注意事项：

- 两端 IB 设备名不同，必须在 `rdma_deployment` 里按节点分别指定，不能用统一的 `mlx5_0`。
- 必须使用 `staging_copy`：RDMA 只注册 DRAM 的 value region，SSD 行没有可被远端直接 DMA 的本地地址。
- 每次重复都要重启 PS / 换 namespace：RDMA lane 的会合协议没有实例标识，复用旧 server 会
  `peer_unreachable`。
- PS 的 `dram` / `ssd.db` 每次用新目录，跑完删除，避免上一次的 tier 状态污染。
- 本页比例扫描显式写了 `"tiering": {"cache_policy": "none"}`；配置模板默认是 `lru`，直接复用模板会开启读缓存。

本次的 per-step CSV、生命周期 JSON、runner 日志和一次性 harness 保存在运行产物目录
`results/tiered_e2e_11052043/`（不随仓库提交）；需要原始数据时可通过 PR 附件或联系作者获取。

## 局限

- 单 shard、单 client rank、单卡；没有多 rank NCCL/DP 场景。
- 只跑一个 epoch（64 steps），没有覆盖训练后期的访问分布漂移。
- 数据是 `processed_day_0`，不是完整 Criteo 41M；绝对吞吐不能直接外推到生产。
- SSD 是 overlay/NVMe 上的 `ssd.db`，不是裸设备；IOURING + 512 B 随机写。
- 所有数字来自 `staging_copy`；`direct_sg` 不支持 SSD 行。
- n=3，CV<2%，但仍是 3 次重复。

## 相关

- [端到端训练基准](e2e.md)
- [测试范围与指标](methodology.md)
- [性能报告](reports.md)
- [KVEngineComposite / TIERED_VALUE_STORE 配置](../storage/composite_kvengine.md)
