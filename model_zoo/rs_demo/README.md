# rs_demo

用于在本地快速模拟较大数据量的 RecStore 训练读写更新压力，并导出结构化性能数据。
该 demo 默认复用 DLRM 同源数据入口和组织方式（`processed_day_0_data` + custom dataloader + KJT）。

## 1. 功能

- 使用 DLRM 相同数据来源：`model_zoo/torchrec_dlrm/processed_day_0_data`
- 使用 DLRM 相同稀疏组织：26 特征 -> KJT -> 拼接 ids
- 更新使用与 DLRM 融合模式一致的 fused id：`(table_idx << fuse_k) + id`
- 内部采用模块化结构：`config / data / runtime / runners / cli`
- 执行批量 `emb_read` + `emb_update_table` 循环（可调 steps/batch）
- 可选自动启动/停止 `ps_server`
- 强制开启本地结构化上报（JSONL）
- 自动调用 `analyze_embupdate_stages.py` 导出 CSV
- `read_before_update` 默认走 `emb_prefetch + emb_wait_result` 稳定读路径（避免同步读路径在部分环境下崩溃）

## 2. 快速运行

默认输出目录已迁移到共享挂载盘：

- `/nas/home/shq/docker/rs_demo/outputs/<run_id>`
- `/nas/home/shq/docker/rs_demo/logs/<run_id>`
- `/nas/home/shq/docker/rs_demo/runtime/<run_id>`

在仓库根目录执行：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --steps 60 \
  --batch-size 4096 \
  --num-embeddings 200000 \
  --embedding-dim 128 \
  --run-id rs-demo-recstore-local \
  --output-root /nas/home/shq/docker/rs_demo
```

单机 distributed TorchRec：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 4 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --rdzv-id rs-demo-local \
  --run-id rs-demo-local \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server
```

双机手工启动 distributed TorchRec：

机器 A（`node-rank 0`）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 2 \
  --node-rank 0 \
  --nproc-per-node 4 \
  --master-addr <machine-a-ip> \
  --master-port 29500 \
  --rdzv-id rs-demo-2node \
  --run-id rs-demo-2node \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server
```

机器 B（`node-rank 1`）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 2 \
  --node-rank 1 \
  --nproc-per-node 4 \
  --master-addr <machine-a-ip> \
  --master-port 29500 \
  --rdzv-id rs-demo-2node \
  --run-id rs-demo-2node \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server
```

如需 profiler trace（每次运行会在 trace dir 下生成多个 trace 文件，并聚合到 trace csv）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 4 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --rdzv-id rs-demo-profiler \
  --run-id rs-demo-profiler \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server \
  --torchrec-profiler
```

## 3. 常用参数

- `--num-embeddings`：表大小
- `--embedding-dim`：向量维度
- `--batch-size`：每步 keys 数
- `--steps`：总迭代数
- `--warmup-steps`：预热步数（不计入脚本内 read/update 统计）
- `--output-root`：输出根目录，默认 `/nas/home/shq/docker/rs_demo`
- `--run-id`：本次运行标识；未指定时自动生成
- `--data-dir`：DLRM processed day0 目录（默认 `model_zoo/torchrec_dlrm/processed_day_0_data`）
- `--fuse-k`：与 DLRM 相同的融合位移参数（默认 `30`）
- `--read-before-update/--no-read-before-update`：是否每步先读后更
  - 开启时：读路径采用 `prefetch/wait`，并统计 `emb_read` 耗时
- `--start-server/--no-start-server`：是否自动起停 `ps_server`
- `--server-port0/--server-port1`：server 端口（默认读取 `recstore_config.json`）
- `--allocator`：value 内存管理器（默认 `R2ShmMalloc`，更适合压测）
- `--nnodes`：TorchRec 分布式节点数
- `--node-rank`：当前节点编号
- `--nproc-per-node`：每个节点的进程数
- `--master-addr`：TorchRec rendezvous master 地址
- `--master-port`：TorchRec rendezvous master 端口
- `--rdzv-backend`：TorchRec rendezvous backend，默认 `c10d`
- `--rdzv-id`：TorchRec rendezvous 标识；多机手工启动时两端必须一致
- `--torchrec-main-csv`：TorchRec 主报表 CSV 路径
- `--torchrec-main-agg-csv`：TorchRec 主报表聚合 CSV 路径（mean/p50/p95/max）
- `--torchrec-profiler`：启用 Torch profiler 并导出 trace 聚合 CSV
- `--torchrec-trace-dir`：Torch profiler trace 输出目录
- `--torchrec-trace-csv`：Torch profiler trace 聚合 CSV 路径
- `--torchrec-compare-recstore-csv`：可选，指定 RecStore CSV 以导出对照差值表
- `--torchrec-compare-csv`：RecStore vs TorchRec 对照差值 CSV 路径

## 4. 结果文件

- RecStore JSONL：`<output_root>/outputs/<run_id>/recstore_events.jsonl`
- RecStore CSV：`<output_root>/outputs/<run_id>/recstore_embupdate.csv`
- Server 日志：`<output_root>/logs/<run_id>/ps_server.log`
- TorchRec rank CSV：`<output_root>/outputs/<run_id>/torchrec_ranks/rank*.csv`
- TorchRec 主报表 CSV：`<output_root>/outputs/<run_id>/torchrec_main.csv`
- TorchRec 主报表聚合 CSV：`<output_root>/outputs/<run_id>/torchrec_main_agg.csv`
- TorchRec profiler trace 目录：`<output_root>/outputs/<run_id>/torchrec_traces`
- TorchRec profiler trace CSV：`<output_root>/outputs/<run_id>/torchrec_trace.csv`
- RecStore vs TorchRec 对照 CSV：`<output_root>/outputs/<run_id>/recstore_torchrec_compare.csv`
- Runtime 配置与 KV 数据：`<output_root>/runtime/<run_id>/...`

TorchRec 主报表（`--torchrec-main-csv`）关键列：

- `collective_total_ms`：collective launch + wait 的总耗时
- `kv_local_only_ms`：本地 embedding lookup + pool 的耗时（不含 pack/unpack）
- `kv_extended_ms`：输入打包 + 本地 lookup/pool + 输出解包的总耗时
- `network_proxy_torchrec_extended_ms`：`collective_total + input_pack + output_unpack` 的扩展通信代理项

TorchRec 主报表聚合 CSV（`--torchrec-main-agg-csv`）会对每个 `*_ms` 列导出：

- `*_mean`
- `*_p50`
- `*_p95`
- `*_max`

对照差值 CSV（`--torchrec-compare-csv`）默认导出以下口径：

- `network_main`：`RecStore(network_transport)` vs `TorchRec(collective_total)`
- `network_extended`：`RecStore(network_transport)` vs `TorchRec(collective + pack + unpack)`
- `kv_strict`：`RecStore(storage_backend_update)` vs `TorchRec(kv_local_only)`
- `server_vs_extended`：`RecStore(server_total)` vs `TorchRec(kv_extended)`

`dist_mode=single_node` 表示当前为单机 distributed；`dist_mode=multi_node` 表示当前为多机 distributed。
