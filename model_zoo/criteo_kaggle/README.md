# Criteo Kaggle（两机 RDMA）

完整训练出 val AUC，以及 RecStore-RDMA vs TorchRec-HBM 的短步 e2e 吞吐/延迟对比。两机仓库通过 NFS 共享，脚本不 rsync。

## 拓扑

两机各 1 个训练进程（每进程 1 张 GPU）+ 各 1 个 RDMA PS：

- 对端：client rank0（rendezvous master）+ PS shard1
- 本机：client rank1 + PS shard0（RDMA control-plane 必须在本机）

全量训练脚本（`run_*.sh`）的主机、SSH、GPU 由 `cluster.sh` 的环境变量配置（`CRITEO_LOCAL_*` / `CRITEO_REMOTE_*`）。e2e 对比脚本 `bench_compare.py` 不读这些，拓扑由命令行两个配置给出（见下）。

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

单文件 Python 脚本 `bench_compare.py`（原 `bench_compare.sh` + `cluster.sh` 的逻辑都在里面）：

```bash
python3 model_zoo/criteo_kaggle/bench_compare.py \
    --clients "[(10.0.2.191, 1), (10.0.2.192, 3)]" \
    --ps "[10.0.2.191, 10.0.2.192]"

# 不带这两个参数时会交互式询问；也可用 CRITEO_CLIENTS / CRITEO_PS 环境变量
python3 model_zoo/criteo_kaggle/bench_compare.py --clients 10.0.2.191:1,10.0.2.192:3 --ps 10.0.2.191,10.0.2.192 --steps 40
```

两个配置：

- `--clients`：计算节点 `(IP, GPU)` pair 列表。列表顺序即 `node_rank`，**第一个是 rendezvous master**；每节点 1 进程 1 卡。
- `--ps`：参数服务器 IP 列表，顺序即 shard 顺序。**其中必须有一台是本机**（RDMA control-plane 要 bind 本机地址）；如果本机那台不在最前面，脚本会自动把它挪到 shard0 并打印一行提示。同一 IP 出现多次时 PS 端口依次 +1。

两种写法都吃：`[(ip, gpu), ...]` / `ip:gpu,ip:gpu`（PS 同理）。IP 是否为本机由 `ip -4 addr` / `ifconfig` + 主机名解析判断，本机走本地进程，其余走 `ssh {--ssh-user}@IP`。

其他参数（均有默认值）：`--ssh-user root`、`--ssh-port 22222`、`--ps-port 15000`、`--master-port 29500`、`--data-dir <此目录>/processed`、`--output-dir results/criteo_kaggle_e2e_<MMDDHHMM>`、`--batch-size 2048`、`--num-embeddings 800000`、`--steps 80`、`--warmup-steps 5`、`--repeat 3`、`--index-type DRAM_PET_HASH`。未识别的参数原样转给 e2e CLI（如 `--no-torchrec`、`--dry-run`、`--read-mode`）。

脚本自己完成：sshpass shim（密码走 `SSHPASS`，默认 `1234`，不出现在命令行；未装 sshpass 又清空 `SSHPASS` 则退回密钥登录）、预处理产物与 `build/bin/petps_server` 检查、每台机 `mlx5_0` ACTIVE/InfiniBand preflight、跑前跑后清理残留进程、调用 `tools.benchmarks.e2e.custom.cli --transports rdma --skip-build --skip-tests`、最后校验 TorchRec NCCL 日志里出现 `NET/IB` 与 `mlx5_0:1/IB`。报告在 `$OUTPUT_DIR/summary.md`。因为默认 `--skip-build`，请先编好 `petps_server`。

## 参考 AUC（不是本拓扑）

单机 1 卡、RecStore-**BRPC** 上一次全量训练：

| 后端 | Epoch 1 val AUC | Epoch 2 val AUC |
|---|---|---|
| TorchRec-HBM | **0.7945** | 0.7679 |
| RecStore-BRPC | 0.7803 | **0.7867** |

本目录入口是两机 RecStore-RDMA，不要把上表当成 RDMA 结果。
