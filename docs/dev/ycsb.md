# KVEngine 存储基准

目前可以进行 RecStore 内部存储横向对比（`kvdb` 和 `cceh`）；以及 RecStore 与其他 KV 存储对比（`rocksdb`、`leveldb`、`lmdb`、`sqlite`、`wiredtiger`、...）。

## 支持的对比

外部存储要在 CMake 配置时手动启用，并且本地要有对应开发库：

- `kvdb`、`cceh`：RecStore 内部存储引擎，默认启用。
- `basic`：YCSB 自带内存 DB，只用于 smoke 测试，默认启用。
- `rocksdb`、`leveldb`、`lmdb`、`sqlite`、`wiredtiger`：需要 `-DBIND_XXX=ON`（使用大写，如 `-DBIND_ROCKSDB=ON`），并且本地要有对应开发库。

`ycsb` 二进制从 `third_party/ycsb/core/ycsbc.cc` 进入。DB 名称由各绑定文件里的 `DBFactory::RegisterDB(...)` 注册。

??? note "外部存储安装"

    | engine | 开发库包名 |
    |---|---|
    | `rocksdb` | `librocksdb-dev` |
    | `leveldb` | `libleveldb-dev` |
    | `lmdb` | `liblmdb-dev` |
    | `sqlite` | `libsqlite3-dev` |
    | `wiredtiger` | `libwiredtiger-dev` |

    Ubuntu / Debian 上可以直接用：

    ```bash title="通过 apt 安装全部依赖" linenums="1" hl_lines="3-7"
    sudo apt-get update
    sudo apt-get install -y \
      librocksdb-dev \
      libleveldb-dev \
      liblmdb-dev \
      libsqlite3-dev \
      libwiredtiger-dev
    ```

## 构建

只运行 RecStore 存储引擎：

```bash
cmake -S . -B build
cmake --build build --target ycsb -j
```

对比 RocksDB / LevelDB / LMDB：

```bash linenums="1" hl_lines="2-5"
cmake -S . -B build \
  -DBIND_ROCKSDB=ON \
  -DBIND_LEVELDB=ON \
  -DBIND_LMDB=ON \
  -DBIND_SQLITE=ON
cmake --build build --target ycsb -j
```

如果缺少某个库，CMake 会在 `find_library(... REQUIRED)` 处失败。先装库，或者先从 `--engines` 里去掉对应 engine。

## 横向对比

脚本：`tools/benchmarks/run_kvengine_compare.py`

默认只跑 `kvdb` 和 `cceh`，workload 用 `workloada`：

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --build \
  --engines kvdb cceh \
  --workloads workloada workloadb workloadc \
  --record-count 100000 \
  --operation-count 100000 \
  --threads 16 \
  --repeat 3 \
  --output-dir /tmp/recstore_ycsb_compare
```

每个 engine、workload、repeat 都有独立数据目录。默认会删除同名旧目录，避免复用上次 load 的数据。需要保留数据时加 `--keep-data`。

## KVEngine 随机 batch lookup 对齐

对齐 PS RDMA GET 的 `batch_keys=N` 时，使用 `benchmark_kv_engine` 的
`batch_get_flat` 模式。普通 YCSB read 每次读取单个 key；`batch_get_flat` 每次生成
一组随机 key，直接调用 `BaseKV::BatchGetFlat`，吞吐单位为 keys/s。

通过对比 storage-only 与 PS/network，可以判断 PS RDMA 是否已经被 KVEngine 本地
lookup 上限卡住。2026-05-31 的 RDMA GET 报告使用了 `batch_keys=500`：

| index | storage-only `BatchGetFlat(500 random keys)` | PS RDMA 对齐结论 |
|---|---:|---|
| `DRAM_EXTENDIBLE_HASH` | `19.45M keys/s` | PS RDMA EH `19.37M keys/s` 基本碰到存储层上限 |
| `DRAM_PET_HASH` | `51.96M keys/s` | PET 需要搭配 staging-copy response path 才能释放存储层优势 |

PS RDMA 对齐结论和 response path 选择见 `docs/parameter_server/rdma.md`。

直接运行二进制示例：

```bash
./build/bin/benchmark_kv_engine \
  --engine=dram \
  --index_type=DRAM_PET_HASH \
  --record_count=300000 \
  --operation_count=300000 \
  --value_size=512 \
  --threads=16 \
  --workload=c \
  --read_mode=batch_get_flat \
  --batch_keys=500
```

通过 compare runner 运行时，对应参数是：

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --engines kvdb \
  --workloads workloadc \
  --record-count 300000 \
  --operation-count 300000 \
  --threads 16 \
  --read-mode batch_get_flat \
  --batch-keys 500 \
  --output-dir results/kv_batchget_compare
```

`summary.csv` 会记录 `read_mode` 和 `batch_keys`，用于和 PS/network 层报告对齐。

## 外部存储

配好环境后执行：

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --engines kvdb cceh rocksdb leveldb lmdb \
  --workloads workloada workloadb workloadc \
  --record-count 100000 \
  --operation-count 100000 \
  --threads 16 \
  --repeat 3 \
  --output-dir /tmp/ycsb_storage_compare
```

脚本会自动给这些绑定传入独立路径：

- `kvdb`: `hybridkv.path=<run data dir>`
- `cceh`: `cceh.path=<run data dir>`
- `rocksdb`: `rocksdb.dbname=<run data dir>`
- `leveldb`: `leveldb.dbname=<run data dir>`
- `lmdb`: `lmdb.dbpath=<run data dir>`
- `sqlite`: `sqlite.dbpath=<run data dir>`
- `wiredtiger`: `wiredtiger.home=<run data dir>`

??? example "调参"

    全局 YCSB 参数用 `--extra-prop`：

    ```bash
    python3 tools/benchmarks/run_kvengine_compare.py \
      --engines kvdb cceh \
      --workloads workloada \
      --extra-prop fieldcount=10 \
      --extra-prop fieldlength=128 \
      --extra-prop requestdistribution=zipfian
    ```

    只给某个 engine 的参数用 `--engine-prop`：

    ```bash
    python3 tools/benchmarks/run_kvengine_compare.py \
      --engines kvdb cceh \
      --workloads workloada \
      --engine-prop kvdb:hybridkv.synthetic_bytes=1024 \
      --engine-prop kvdb:hybridkv.read_return=none \
      --engine-prop cceh:cceh.value_size=1024
    ```

    `kvdb` 默认用 `hybridkv.mode=perf` 和 `hybridkv.read_return=none`，适合先看存储路径吞吐。如果要检查字段反序列化成本，可以改成：

    ```bash
    --engine-prop kvdb:hybridkv.mode=compat --engine-prop kvdb:hybridkv.read_return=parse
    ```

??? example "单 engine 运行"

    脚本最终也是调用 `./build/bin/ycsb`。需要手工调试时可以直接跑：

    ```bash
    ./build/bin/ycsb -load -run -db kvdb -threads 16 \
      -P third_party/ycsb/workloads/workloada \
      -P third_party/ycsb/db/kv_db.properties \
      -p hybridkv.path=/tmp/ycsb-kvdb \
      -p recordcount=10000 \
      -p operationcount=10000 \
      -p measurementtype=basic
    ```

    `cceh`：

    ```bash
    ./build/bin/ycsb -load -run -db cceh -threads 16 \
      -P third_party/ycsb/workloads/workloada \
      -P third_party/ycsb/db/cceh.properties \
      -p cceh.path=/tmp/ycsb-cceh \
      -p recordcount=10000 \
      -p operationcount=10000 \
      -p measurementtype=basic
    ```

## 结果

脚本会输出一个汇总 `summary.csv`：

```bash
OUT=/tmp/recstore_ycsb_compare
column -s, -t "$OUT/summary.csv" | less -S
```

??? example "输出示例"

    ```csv
    workload,engine,db,repeat,record_count,operation_count,threads,phase,exit_code,load_runtime_sec,load_operations,load_throughput_ops_sec,run_runtime_sec,run_operations,run_throughput_ops_sec,data_path,log_path,error_tail
    workloada,rocksdb,rocksdb,0,100000,100000,16,load-run,0,1.41976,100000,70434.5,0.311262,100000,321273.0,/tmp/recstore_ycsb_third_party_compare_100k_t16/data/workloada_rocksdb_r0,/tmp/recstore_ycsb_third_party_compare_100k_t16/logs/workloada_rocksdb_r0.log,
    workloada,leveldb,leveldb,0,100000,100000,16,load-run,0,0.752207,100000,132942.0,0.567562,100000,176192.0,/tmp/recstore_ycsb_third_party_compare_100k_t16/data/workloada_leveldb_r0,/tmp/recstore_ycsb_third_party_compare_100k_t16/logs/workloada_leveldb_r0.log,
    workloada,lmdb,lmdb,0,100000,100000,16,load-run,0,2.40212,100000,41629.9,1.44824,100000,69049.2,/tmp/recstore_ycsb_third_party_compare_100k_t16/data/workloada_lmdb_r0,/tmp/recstore_ycsb_third_party_compare_100k_t16/logs/workloada_lmdb_r0.log,
    workloada,sqlite,sqlite,0,100000,100000,16,load-run,0,7.40299,100000,13508.1,2.63494,100000,37951.6,/tmp/recstore_ycsb_third_party_compare_100k_t16/data/workloada_sqlite_r0,/tmp/recstore_ycsb_third_party_compare_100k_t16/logs/workloada_sqlite_r0.log,
    workloada,wiredtiger,wiredtiger,0,100000,100000,16,load-run,0,1.72329,100000,58028.4,0.580689,100000,172209.0,/tmp/recstore_ycsb_third_party_compare_100k_t16/data/workloada_wiredtiger_r0,/tmp/recstore_ycsb_third_party_compare_100k_t16/logs/workloada_wiredtiger_r0.log,
    ```

`summary.csv` 里的主要字段：

| 字段 | 内容 |
|---|---|
| `workload` | 使用的 workload 文件名，例如 `workloada`。 |
| `engine` | 脚本里的 engine 名称，例如 `kvdb`、`cceh`、`rocksdb`。 |
| `repeat` | 第几次重复运行，从 0 开始。 |
| `record_count` | load 阶段写入的记录数。 |
| `operation_count` | run 阶段执行的操作数。 |
| `threads` | YCSB 客户端线程数。 |
| `load_runtime_sec` | load 阶段耗时，单位秒。 |
| `load_throughput_ops_sec` | load 阶段吞吐，单位 ops/s。 |
| `run_runtime_sec` | run 阶段耗时，单位秒。 |
| `run_throughput_ops_sec` | run 阶段吞吐，单位 ops/s。 |
| `exit_code` | YCSB 进程退出码，0 表示命令正常结束。 |
| `data_path` | 本次运行使用的数据目录。 |
| `log_path` | 本次运行的完整 stdout/stderr 日志。 |

如果某个 engine 失败，`exit_code` 非 0，完整 stdout/stderr 在对应 `log_path`。
