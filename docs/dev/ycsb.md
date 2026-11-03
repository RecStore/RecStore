# KVEngine 存储基准

`tools/benchmarks/run_kvengine_compare.py` 驱动 `benchmark_kv_engine`，在多个 KVEngine
组合上运行 YCSB 风格的 workload。它只测存储层，不经过 PS 和 RDMA，因此用来定位
“PS/network 是否已经被 KVEngine 本地能力卡住”。

## 支持的引擎

脚本用别名选择引擎组合（`--engines`）：

| 别名 | index | value store | 说明 |
|---|---|---|---|
| `petkv` | PET hash | 值内联 | 独立 KVEngine 实现 |
| `dram_pet_dram` | `DRAM_PET_HASH` | `DRAM_VALUE_STORE` | 默认组合 |
| `dram_pet_tiered` | `DRAM_PET_HASH` | `TIERED_VALUE_STORE` | 默认组合 |
| `dram_eh_dram` | `DRAM_EXTENDIBLE_HASH` | `DRAM_VALUE_STORE` | 需显式指定 |
| `dram_eh_ssd` | `DRAM_EXTENDIBLE_HASH` | `SSD_VALUE_STORE` | 需显式指定 |
| `dram_pet_ssd` | `DRAM_PET_HASH` | `SSD_VALUE_STORE` | 需显式指定 |
| `dram_eh_tiered` | `DRAM_EXTENDIBLE_HASH` | `TIERED_VALUE_STORE` | 需显式指定 |
| `fasterkv` / `fasterkv_ssd` | — | — | 外部引擎，需 `-DRECSTORE_ENABLE_THIRD_PARTY_ENGINES=ON` |
| `hps_rocksdb` | — | RocksDB | 外部引擎，同上 |

默认只跑 `petkv`、`dram_pet_dram`、`dram_pet_tiered`。

## 构建

```bash
cmake -S . -B build
cmake --build build --target benchmark_kv_engine -j
```

外部引擎（`fasterkv*`、`hps_rocksdb`）需要额外打开：

```bash
cmake -S . -B build -DRECSTORE_ENABLE_THIRD_PARTY_ENGINES=ON
cmake --build build --target benchmark_kv_engine -j
```

给 runner 加 `--build` 会自动完成上面两步，并根据 `--engines` 决定是否打开外部引擎。

## 运行

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --engines dram_pet_dram dram_pet_tiered \
  --workloads a b c \
  --distributions uniform zipfian \
  --record-count 1000000 \
  --runtime-seconds 5 \
  --threads 16 \
  --value-size 128 \
  --read-mode get \
  --output-dir results/kvengine_compare
```

- `--operation-count` 仅为兼容 `summary.csv` 保留，计时基准会忽略它；用
  `--runtime-seconds` 控制 run 阶段时长。
- `--load-threads 0` 表示 load 阶段复用 `--threads`。
- `--repeat` 重复轮次，`--keep-data` 保留上一轮数据，`--skip-load` / `--skip-run`
  只跑其中一个阶段。
- 数据目录默认落在 `/dev/shm/recstore`（DRAM）和 `/mnt/nvme1n1_recstore/recstore`（SSD），
  可用 `--dram-capacity-bytes` / `--ssd-capacity-bytes` 覆盖容量。

### 引擎专属参数

全局参数用 `--extra-arg` 透传给 `benchmark_kv_engine`：

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --engines dram_pet_dram \
  --workloads a \
  --extra-arg=--print_util=true
```

单个引擎的 gflag 用 `--engine-prop <engine>:<flag>=<value>`：

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --engines dram_pet_dram dram_eh_dram \
  --workloads a \
  --engine-prop dram_pet_dram:dram_capacity_bytes=1073741824
```

## KVEngine 随机 batch lookup 对齐

对齐 PS RDMA GET 的 `batch_keys=N` 时，使用 `benchmark_kv_engine` 的
`batch_get_flat` 模式。普通 `get` 每次读取单个 key；`batch_get_flat` 每次生成一组随机
key，直接调用 `BaseKV::BatchGetFlat`，吞吐单位为 keys/s。

通过对比 storage-only 与 PS/network，可以判断 PS RDMA 是否已经被 KVEngine 本地
lookup 上限卡住。2026-05-31 的 RDMA GET 报告使用了 `batch_keys=500`：

| index | storage-only `BatchGetFlat(500 random keys)` | PS RDMA 对齐结论 |
|---|---:|---|
| `DRAM_EXTENDIBLE_HASH` | `19.45M keys/s` | PS RDMA EH `19.37M keys/s` 基本碰到存储层上限 |
| `DRAM_PET_HASH` | `51.96M keys/s` | PET 需要搭配 staging-copy response path 才能释放存储层优势 |

PS RDMA 对齐结论和 response path 选择见 `docs/parameter_server/rdma.md`。

直接运行二进制：

```bash
./build/bin/benchmark_kv_engine \
  --engine_class=KVEngineComposite \
  --index_type=DRAM_PET_HASH \
  --value_store_type=DRAM_VALUE_STORE \
  --dram_path=/dev/shm/recstore_bench \
  --ssd_path=/tmp/recstore_bench_ssd \
  --record_count=300000 \
  --value_size=512 \
  --thread_num=16 \
  --load_thread_num=16 \
  --running_seconds=5 \
  --workload=c \
  --read_mode=batch_get_flat \
  --batch_keys=500
```

通过 runner 运行时，对应参数是：

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --engines dram_pet_dram \
  --workloads c \
  --record-count 300000 \
  --threads 16 \
  --read-mode batch_get_flat \
  --batch-keys 500 \
  --output-dir results/kv_batchget_compare
```

`summary.csv` 会记录 `read_mode` 和 `batch_keys`，用于和 PS/network 层报告对齐。

## 结果

每次运行在 `--output-dir` 下生成：

- `summary.csv`：每次引擎 / workload / repeat 一行；
- `kvengine_workload_summary.csv`：按 workload 聚合；
- `kvengine_ycsb_run_throughput.svg`：run 阶段吞吐图（matplotlib 可用时）。
- `log_path` / `raw_log_path`：单个引擎运行的完整日志。

`summary.csv` 的主要字段：

| 字段 | 内容 |
|---|---|
| `workload` | YCSB workload，`a` / `b` / `c` |
| `engine` | `--engines` 里的别名 |
| `index_type` / `value_store_type` | 该别名的 index 与 value store |
| `repeat` | 第几次重复运行，从 0 开始 |
| `record_count` / `threads` | load 记录数与线程数 |
| `distribution` / `zipfian_alpha` | key 分布 |
| `read_mode` / `batch_keys` | 读模式与 `batch_get_flat` 的批量大小 |
| `load_runtime_sec` / `load_throughput_ops_sec` | load 阶段耗时与吞吐 |
| `run_runtime_sec` / `run_throughput_ops_sec` | run 阶段耗时与吞吐 |
| `run_read_operations` / `run_update_operations` | run 阶段读写操作数 |
| `exit_code` / `error_tail` | 退出码与失败时的错误摘要 |

只重绘已有结果时加 `--draw --output-dir <dir>`。
