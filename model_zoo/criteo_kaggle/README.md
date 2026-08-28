# Criteo Kaggle（两机 RDMA）

本目录有**两套互不替代**的入口：

| 入口 | 目的 | 训练程序 |
|---|---|---|
| `run_torchrec.sh` / `run_recstore.sh` | 完整训练，出 **val AUC** | `dlrm_main_*single*.py`，默认 2 epoch |
| `bench_compare.py` | RecStore-RDMA vs TorchRec-HBM 的短步 **吞吐/延迟** | `model_zoo/rs_demo/run_mock_stress.py`，默认 80 step |

两机仓库走 NFS，脚本不 rsync。必须在本机（默认 `10.0.2.192`）启动：RDMA control-plane 要 bind 本机地址，PS shard0 落在本机。

## 文件

| 文件 | 作用 |
|---|---|
| `preprocess.py` | `train.txt` → `processed/day_0_{dense,sparse,labels}.npy` |
| `cluster.sh` | **只给全量训练用**。source，不要直接执行。拓扑、SSH、NCCL、拉起两 rank |
| `run_torchrec.sh` | 两机 TorchRec-HBM 全量训练 |
| `run_recstore.sh` | 两机 RecStore-RDMA 全量训练（先起 PS，再起 trainer） |
| `start_rdma_ps.py` | 被 `run_recstore.sh` 调用：按 `cluster.sh` 的环境变量起两 shard `petps_server`，写出 RDMA 客户端环境 |
| `recstore_config.json` | 全量 RecStore 训练的 PS / 客户端配置（IP、端口、`/dev/shm` KV） |
| `bench_compare.py` | 短步 e2e 对比。**不读** `cluster.sh` / `recstore_config.json`，拓扑用自己的命令行参数，内部调 `tools.benchmarks.e2e.custom.cli` |

## 依赖

- 预处理产物 `processed/day_0_*.npy`
- RecStore 路径还需要 `build/bin/petps_server`
- 两机 `mlx5_0` 为 ACTIVE InfiniBand
- `sshpass`；对端默认 `root@10.0.2.191` 端口 `22222`

## 预处理

```bash
python3 model_zoo/criteo_kaggle/preprocess.py
# 或: python3 model_zoo/criteo_kaggle/preprocess.py /path/to/train.txt /path/to/out
```

默认读本目录 `train.txt`，写出 `processed/train_{dense,sparse,labels}.npy`，dense 做 `log(x+3)`，把 `-inf` 换成 `log(3)`，再 symlink 成 `day_0_*.npy`（trainer / e2e 只认这个名字）。

## 全量训练（val AUC）

固定两机拓扑，改 `cluster.sh` 里的环境变量（或启动前 export 覆盖）：

- 对端 `10.0.2.191` GPU **0**：client rank0（rendezvous master）+ PS shard1
- 本机 `10.0.2.192` GPU **3**：client rank1 + PS shard0
- SSH：`SSHPASS`（默认 `1234`）+ `sshpass -e`，密码不进命令行
- NCCL：`NCCL_IB_HCA=mlx5_0`，dense DDP / TorchRec embedding all-reduce 走 IB

在仓库根目录、本机上跑：

```bash
./model_zoo/criteo_kaggle/run_torchrec.sh
./model_zoo/criteo_kaggle/run_recstore.sh
```

默认：`batch_size=2048`，Adagrad `lr=0.05`，2 epoch，`embedding_dim=128`，`--allow_tf32`。可用 `CRITEO_BATCH_SIZE`、`CRITEO_LR`、`CRITEO_EPOCHS`、`CRITEO_LOG_DIR` 覆盖。额外参数原样传给 trainer。

`run_recstore.sh` 还会读 `recstore_config.json`（KV 在每机 `/dev/shm/recstore_kv/...`，不写 NFS），经 `start_rdma_ps.py` 拉起 `petps_server`。

日志：`results/criteo_kaggle_train/logs/`（`torchrec_n{0,1}.log`、`recstore_n{0,1}.log`、RecStore 还有 `petps.log`）。两 rank 各自在自己的 data shard 上算 AUROC，**以 rank0 日志里的 Validation AUROC 为准**。结束时检查两份 trainer 日志都出现 `NET/IB` 和 `mlx5_0:1/IB`。

## 短步 e2e（吞吐 / 延迟）

`bench_compare.py` 是另一条路径：起自己的 PS、跑 `run_mock_stress.py`，默认 80 step / 5 warmup / repeat 3，`num_embeddings=800000`。**不出 AUC**，也不能替代上面的 `run_*.sh`。

```bash
python3 model_zoo/criteo_kaggle/bench_compare.py
python3 model_zoo/criteo_kaggle/bench_compare.py --steps 40
python3 model_zoo/criteo_kaggle/bench_compare.py \
    --clients 10.0.2.191:1,10.0.2.192:3 \
    --ps 10.0.2.191,10.0.2.192
```

默认拓扑与全量训练**不是同一张卡**：对端用 GPU **1**（全量训练是 GPU 0）。`--clients` 列表顺序即 `node_rank`，第一个是 rendezvous master。`--ps` 里必须有一台是本机；若本机不在最前，脚本会把它挪到 shard0。

其余默认：`--ssh-user root`、`--ssh-port 22222`、`--ssh-pass 1234`、`--ps-port 15000`、`--master-port 29500`、`--data-dir <此目录>/processed`、`--output-dir results/criteo_kaggle_e2e_<MMDDHHMM>`、`--batch-size 2048`、`--index-type DRAM_PET_HASH`。未识别参数原样转给 e2e CLI（如 `--no-torchrec`、`--dry-run`）。默认 `--skip-build`，请先编好 `petps_server`。

报告在 `--output-dir/summary.md`。若跑了 TorchRec，会校验其 NCCL 日志里出现 `NET/IB` 与 `mlx5_0:1/IB`。

## 参考 AUC（不是本拓扑）

单机 1 卡、RecStore-**BRPC** 上一次全量训练：

| 后端 | Epoch 1 val AUC | Epoch 2 val AUC |
|---|---|---|
| TorchRec-HBM | **0.7945** | 0.7679 |
| RecStore-BRPC | 0.7803 | **0.7867** |

本目录入口是两机 RecStore-RDMA，不要把上表当成 RDMA 结果。
