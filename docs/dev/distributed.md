# 分布式训练

以两台机器为例，每台机器 1 个 trainer，1 个分片的存储后端。

## 环境

约定：

- 机器 A: `10.0.0.1`，shard 0，trainer rank 0
- 机器 B: `10.0.0.2`，shard 1，trainer rank 1
- `master_addr=10.0.0.1`
- `master_port=29670`

## 配置文件

两台机器必须共享同一份 `distributed_client` 配置，尤其是 `num_shards`、`hash_method`、`servers`，下称 `./recstore_config.distributed.json`。

??? example "配置文件示例"

    ```json title="recstore_config.distributed.json" linenums="1" hl_lines="6 7-18 37 38 40-51"
    {
      "cache_ps": {
        "ps_type": "BRPC",
        "num_threads": 32,
        "num_shards": 2,
        "servers": [
          {
            "host": "10.0.0.1",
            "port": 15123,
            "shard": 0
          },
          {
            "host": "10.0.0.2",
            "port": 15123,
            "shard": 1
          }
        ],
        "base_kv_config": {
          "engine_type": "KVEngineComposite",
          "capacity": 1000000,
          "index": {
            "type": "DRAM_PET_HASH"
          },
          "value": {
            "type": "DRAM_VALUE_STORE",
            "path": "/tmp/recstore_dist_data",
            "default_value_size_hint": 512,
            "dram_allocator": {
              "type": "CONCURRENT_SLAB_MEMORY_POOL",
              "capacity_bytes": 512000000
            }
          }
        }
      },
      "distributed_client": {
        "num_shards": 2,
        "hash_method": "city_hash",
        "max_keys_per_request": 500,
        "servers": [
          {
            "host": "10.0.0.1",
            "port": 15123,
            "shard": 0
          },
          {
            "host": "10.0.0.2",
            "port": 15123,
            "shard": 1
          }
        ]
      },
      "client": {
        "host": "10.0.0.1",
        "port": 15123,
        "shard": 0
      }
    }
    ```

高亮的字段是两台机器必须一致的部分：

- `cache_ps.num_shards` 与 `distributed_client.num_shards` 必须相等，且等于 `servers` 的条目数；
- 两处 `servers` 的 `host` / `port` / `shard` 要能一一对应，`shard` 覆盖 `0..num_shards-1`（顺序无所谓）；
- `hash_method` 决定 key 落到哪个分片，两端不一致会出现"读得到但写到别的分片"；
- `client` 只给单机入口用，分布式训练不走它。

## 参数服务器

=== "机器 A"

    ```bash
    ./build/bin/ps_server \
      --config_path ./recstore_config.distributed.json \
      -local_shard_id=0
    ```

    期望日志包含：

    ```text
    bRPC Server shard 0 listening on 10.0.0.1:15123
    ```

=== "机器 B"

    ```bash
    ./build/bin/ps_server \
      --config_path ./recstore_config.distributed.json \
      -local_shard_id=1
    ```

    期望日志包含：

    ```text
    bRPC Server shard 1 listening on 10.0.0.2:15123
    ```

??? example "连通性检查"

    本机端口：

    === "A"

        ```bash
        ss -ltnp | grep 15123
        ```

    === "B"

        ```bash
        ss -ltnp | grep 15123
        ```

    ------

    跨机连通性检查：

    === "A -> B"

        ```bash
        nc -vz 10.0.0.2 15123
        ```

    === "B -> A"

        ```bash
        nc -vz 10.0.0.1 15123
        ```

## 训练

可以指定训练输出存放位置：

```bash
mkdir -p ./tmp/rs_demo ./tmp/recstore-dist-shared
```

=== "机器 A"

    ```bash linenums="1" hl_lines="9 10"
    python3 model_zoo/rs_demo/run_mock_stress.py \
      --backend recstore \
      --nnodes 2 \
      --node-rank 0 \
      --nproc-per-node 1 \
      --master-addr 10.0.0.1 \
      --master-port 29670 \
      --run-id recstore-dist-2node-2proc \
      --output-root ./tmp/rs_demo \
      --recstore-runtime-dir ./tmp/recstore-dist-shared \
      --no-start-server
    ```

=== "机器 B"

    ```bash linenums="1" hl_lines="9 10"
    python3 model_zoo/rs_demo/run_mock_stress.py \
      --backend recstore \
      --nnodes 2 \
      --node-rank 1 \
      --nproc-per-node 1 \
      --master-addr 10.0.0.1 \
      --master-port 29670 \
      --run-id recstore-dist-2node-2proc \
      --output-root ./tmp/rs_demo \
      --recstore-runtime-dir ./tmp/recstore-dist-shared \
      --no-start-server
    ```

## 结果

```bash title="输出目录"
ls -l ./tmp/rs_demo/outputs/recstore-dist-2node-2proc
```

```bash title="rank 子目录"
ls -l ./tmp/rs_demo/outputs/recstore-dist-2node-2proc/recstore_ranks
```

```bash title="聚合结果"
cat ./tmp/rs_demo/outputs/recstore-dist-2node-2proc/recstore_main_agg.csv
```

```bash title="明细结果前 5 行"
head -n 5 ./tmp/rs_demo/outputs/recstore-dist-2node-2proc/recstore_main.csv
```

## TorchRec 对齐

=== "机器 A"

    ```bash linenums="1" hl_lines="9"
    python3 model_zoo/rs_demo/run_mock_stress.py \
      --backend torchrec \
      --nnodes 2 \
      --node-rank 0 \
      --nproc-per-node 1 \
      --master-addr 10.0.0.1 \
      --master-port 29671 \
      --run-id torchrec-fair-2node-2proc \
      --output-root ./tmp/rs_demo \
      --torchrec-dist-mode fair_remote \
      --no-start-server
    ```

=== "机器 B"

    ```bash linenums="1" hl_lines="9"
    python3 model_zoo/rs_demo/run_mock_stress.py \
      --backend torchrec \
      --nnodes 2 \
      --node-rank 1 \
      --nproc-per-node 1 \
      --master-addr 10.0.0.1 \
      --master-port 29671 \
      --run-id torchrec-fair-2node-2proc \
      --output-root ./tmp/rs_demo \
      --torchrec-dist-mode fair_remote \
      --no-start-server
    ```

## 工具命令

```bash title="server 日志"
tail -f /path/to/ps_server.log
```

```bash title="rank 日志"
ls -l ./tmp/rs_demo/outputs/recstore-dist-2node-2proc/recstore_ranks
```

```bash title="worker 指纹"
cat ./tmp/rs_demo/outputs/recstore-dist-2node-2proc/recstore_worker_fingerprints.json
```
