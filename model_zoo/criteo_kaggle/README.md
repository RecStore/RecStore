# Criteo Kaggle（两机 RDMA）

完整训练出 val AUC，以及 RecStore-RDMA vs TorchRec-HBM 的短步 e2e 吞吐/延迟对比。两机仓库通过 NFS 共享，脚本不 rsync。

## 拓扑

两机各 1 个训练进程（每进程 1 张 GPU）+ 各 1 个 RDMA PS：

- 对端：client rank0（rendezvous master）+ PS shard1
- 本机：client rank1 + PS shard0（RDMA control-plane 必须在本机）

主机、SSH、GPU 由 `cluster.sh` 的环境变量配置（`CRITEO_LOCAL_*` / `CRITEO_REMOTE_*`），不要把地址写进命令。

- RecStore embedding：RDMA（`petps_server`，不用 BRPC/`ps_server`）
- TorchRec embedding all-reduce 与两边 dense DDP：NCCL IB，`NCCL_IB_HCA=mlx5_0`
- KV 数据在每机 `/dev/shm/...`，不写 NFS

对端 SSH 密码走 `SSHPASS` + `sshpass -e`，不出现在命令行。

## 依赖

- `build/bin/petps_server`（RecStore 训练 / e2e）
- 两机 `mlx5_0` 为 ACTIVE InfiniBand
- `sshpass`、NFS 上的本仓库
- 预处理产物 `processed/day_0_*.npy`

## 预处理

```bash
python3 model_zoo/criteo_kaggle/preprocess.py
# 或: python3 model_zoo/criteo_kaggle/preprocess.py /path/to/train.txt /path/to/out
```

从 `train.txt` 流式写出 `processed/train_{dense,sparse,labels}.npy`，dense 做 `log(x+3)`，把 `-inf` 换成 `log(3)`，并链到 `day_0_*.npy`（`CustomCriteoDataset` / e2e 只认这个名字）。原始 `readme.txt` 是数据集说明。

## 完整训练（val AUC）

在仓库根目录、本机上跑：

```bash
./model_zoo/criteo_kaggle/run_torchrec.sh
./model_zoo/criteo_kaggle/run_recstore.sh
```

默认：`batch_size=2048`，Adagrad `lr=0.05`，2 epoch，`--allow_tf32`。可用环境变量覆盖：`CRITEO_BATCH_SIZE`、`CRITEO_LR`、`CRITEO_EPOCHS`、`CRITEO_LOG_DIR`。额外参数原样传给 trainer。

日志：`results/criteo_kaggle_train/logs/`（`torchrec_n{0,1}.log`、`recstore_n{0,1}.log`）。两 rank 各自在自己的 data shard 上算 AUROC，**以 rank0 日志里的 Validation AUROC 为准**。脚本结束会检查两份日志都出现 `NET/IB` 和 `mlx5_0:1/IB`。

## e2e 吞吐 / 延迟

```bash
./model_zoo/criteo_kaggle/bench_compare.sh
# OUTPUT_DIR=results/foo ./model_zoo/criteo_kaggle/bench_compare.sh --steps 40
```

调用已有 `python3 -m tools.benchmarks.e2e.custom.cli --transports rdma`，默认 80 step / warmup 5 / repeat 3 / `num-embeddings=800000`。报告在 `$OUTPUT_DIR/summary.md`。`--` 后参数传给 CLI。默认 `--skip-build --skip-tests`，请先编好 `petps_server`。

## 参考 AUC（不是本拓扑）

单机 1 卡、RecStore-**BRPC** 上一次全量训练：

| 后端 | Epoch 1 val AUC | Epoch 2 val AUC |
|---|---|---|
| TorchRec-HBM | **0.7945** | 0.7679 |
| RecStore-BRPC | 0.7803 | **0.7867** |

本目录入口是两机 RecStore-RDMA，不要把上表当成 RDMA 结果。
