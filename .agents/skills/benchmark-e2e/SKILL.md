---
name: benchmark-e2e
description: Use when setting up RecStore end-to-end DLRM/TorchRec benchmarks with explicit client placement, PS placement/type, sharding, runnable commands, and Chinese performance summaries.
---

# Benchmark E2E

## Workflow

Use this skill from a RecStore checkout. Do not run helper scripts from this
skill directory; call project scripts directly.

1. Confirm the current directory is the RecStore repo root, or pass `--repo`.
2. Prompt only for P0 inputs unless the user already provided them:
   - client list, each entry as
     `(ssh_host, ssh_port, repo_root, ip, gpu_id, node_rank, nproc_per_node)`;
     default = one local client, `ssh_port=22`, GPU 0, `node_rank=0`, `nproc_per_node=1`
   - PS server list, each entry as
     `(ssh_host, ssh_port, repo_root, ip, ps_port, shard_id)`;
     default = one local PS, `ssh_port=22`, `127.0.0.1:15000`, shard 0
   - result output directory (default = `results/e2e_$(date +%m%d%H%M)`)
   - `ssh_host` is the SSH login target (`local` means run commands locally);
     `ip` is the runtime data-plane address for PS/client traffic and distributed
     rendezvous. They may differ when SSH goes through a bastion or hostname
     while training uses a cluster IP. `ssh_port` is the SSH login port;
     `ps_port` is the PS service port and must not be confused with `ssh_port`.
3. Apply P1 defaults without prompting, and record them in `summary.md`:
   - model = `dlrm`
   - client deployment = inferred from client list
   - PS deployment = inferred from PS server list
   - shard policy = one shard per PS; `distributed_client.num_shards` equals
     the server count
   - dataset path = `model_zoo/torchrec_dlrm/processed_day_0_data`
   - runtime directory = `<output_dir>/runtime`
   - batch size = `1024`
   - embedding dimension = `128`
   - num embeddings = `200000`
   - steps = `80`, warmup steps = `5`, repeat = `3`
   - read mode = `prefetch`, prefetch depth = `0`
   - RecStore index type = `DRAM_PET_HASH`
   - comparison lanes = RecStore-`<PS_TYPE>` + TorchRec-HBM
   - for large `num_embeddings` such as `800000`, use a RecStore PS capacity
     large enough for the fused/multi-table key space; `capacity=8000000` is a
     safe baseline for the day0 DLRM runs. Do not use the small template
     capacity for large E2E comparisons.
4. Override P1 defaults only when the user mentions them. Ask P2 questions only
   when required by the requested experiment:
   - additional TorchRec baseline lanes such as `uvm_caching`, or disabling
     TorchRec with `--no-torchrec`
   - multiple batch sizes, embedding dimensions, or cardinality sweeps
   - non-default index type, read mode, prefetch depth, or run length
   - custom dataset, runtime directory, or per-feature cardinalities
   - RecStore optimization ablations. These are experiment-specific add-on
     lanes, not generic comparison objects. Do not add them to the default
     RecStore vs TorchRec matrix unless the user explicitly asks for prefetch,
     no-prefetch, fusion, or optimization-ablation comparisons. Use
     `RecStore-<PS_TYPE>` as the main RecStore backend label, for example
     `RecStore-BRPC`, `RecStore-GRPC`, `RecStore-SHM`, or `RecStore-RDMA`.
     When an ablation is requested, use these suffixes consistently:
     - `RecStore-<PS_TYPE>`: main lane, `--read-mode prefetch
       --prefetch-depth 0`, fused path enabled. This is same-batch issue+wait,
       not lookahead overlap.
     - `RecStore-<PS_TYPE>-无预取`: add only for no-prefetch ablation;
       `--read-mode direct`, fused path enabled.
     - `RecStore-<PS_TYPE>-无预取无融合`: add only for combined no-prefetch and
       no-fusion ablation; `--read-mode direct --disable-recstore-fusion`.
       Before running, verify the local and remote runner support this CLI.
     - `RecStore-<PS_TYPE>-预取N`: add only for lookahead prefetch experiments;
       `--read-mode prefetch --prefetch-depth N` with `N > 0`, fused path
       enabled.
5. Run:
   - `cmake -S . -B build`
   - `cmake --build build --target ps_server -j`
   - run the correctness tests relevant to the requested PS backend. For BRPC,
     use `ctest -R 'brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure`;
     for GRPC, LOCAL_SHM/SHM, RDMA, or other PS types, use the corresponding
     targeted tests before E2E.
   - start one `ps_server` per PS entry
   - run `model_zoo/rs_demo/run_mock_stress.py` for each RecStore E2E client
   - run matched TorchRec-HBM client commands with the same workload
   - for multi-host runs, verify the runner files and CLI options are present
     on every host before launching. If local code changed, sync the relevant
     files or compare fingerprints; otherwise remote workers may fail with
     `unrecognized arguments` or mismatched behavior.
6. Save deployment, commands, logs, runtime config, CSV artifacts, and
   `summary.md` under the chosen result directory.
7. Write `summary.md` as exactly three report sections, with benchmark
   hyperparameters recorded as Chinese prose under `Workload 说明` before the
   first table:
   - Workload description
   - E2E throughput
   - E2E latency breakdown

## Command Template

Ask the user for P0 inputs only: `client_list`, `ps_server_list`, and
`output_dir`. Use P1 defaults for everything else unless the user explicitly
overrides them.

```bash
cmake -S . -B build
cmake --build build --target ps_server -j
# Example for BRPC. Replace with the targeted tests for the requested PS type.
ctest -R 'brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure
```

Create `<runtime_dir>/recstore_config.json` with the requested PS type and
shard layout. This example uses BRPC; replace `ps_type`, startup requirements,
and environment variables as needed for GRPC, LOCAL_SHM/SHM, RDMA, or other
available PS backends:

```json
{
  "cache_ps": {
    "ps_type": "BRPC",
    "num_shards": 1,
    "servers": [
      {"host": "127.0.0.1", "port": 15000, "shard": 0}
    ],
    "base_kv_config": {
      "capacity": 8000000,
      "index": {"type": "DRAM_PET_HASH"},
      "value": {"type": "DRAM_VALUE_STORE"}
    }
  },
  "distributed_client": {
    "num_shards": 1,
    "hash_method": "city_hash",
    "max_keys_per_request": 65536,
    "servers": [
      {"host": "127.0.0.1", "port": 15000, "shard": 0}
    ]
  }
}
```

Start each PS server from its configured host:

```bash
cd <server_repo>
build/bin/ps_server --config_path <runtime_dir>/recstore_config.json
```

Run one training command per client entry:

```bash
cd <client_repo>
CUDA_VISIBLE_DEVICES=<gpu_id> \
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --ps-type <PS_TYPE> \
  --recstore-index-type <index_type> \
  --ps-kv-backend recstore_dram \
  --batch-size <batch_size> \
  --embedding-dim <embedding_dim> \
  --num-embeddings <num_embeddings> \
  --steps <steps> \
  --warmup-steps <warmup_steps> \
  --read-mode <read_mode> \
  --prefetch-depth <prefetch_depth> \
  --data-dir <dataset_path> \
  --output-root <output_dir> \
  --run-id <run_id> \
  --recstore-runtime-dir <runtime_dir> \
  --no-start-server \
  --server-host <first_ps_ip> \
  --server-port0 <first_ps_port>
```

For distributed clients, add:

```bash
  --nnodes <client_node_count> \
  --node-rank <node_rank> \
  --nproc-per-node <nproc_per_node> \
  --master-addr <master_ip> \
  --master-port <master_port>
```

For single-node distributed runs, `--master-addr` must be an address on that
same node. For example, a 191 single-node 2-GPU run should use
`--master-addr 10.0.2.191`, not a 190 address. For cross-node runs, use the
rank-0 host such as `10.0.2.190`.

The current `run_mock_stress.py` parent runner assumes homogeneous
`nproc_per_node` because it computes `world_size = nnodes * nproc_per_node`.
Do not use it as-is for heterogeneous trainer placement such as
`190:1 trainer + 191:2 trainers`. Manual RecStore workers may be possible, but
TorchRec `DistributedModelParallel` planning still uses `local_world_size` from
`nproc_per_node` and can create invalid placements such as `cuda:2` on a node
with only two visible GPUs. Report such cases as unsupported unless the runner
has been changed to support heterogeneous local world sizes.

Run the matched TorchRec-HBM command by default with the same dataset, batch
size, embedding dimension, steps, warmup steps, client deployment, and GPU
placement. If the user requests `uvm_caching`, add a second TorchRec baseline;
if the user requests RecStore-only, pass `--no-torchrec`.

For TorchRec distributed lanes, prefer NCCL over IB when comparing against a
networked RecStore PS. Do not hard-code HCA or interface names blindly: inspect
the hosts first and set `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, and related
variables to match the actual common devices. The following is only an example
for a cluster where all nodes should use `mlx5_0` and `eno8303`:

```bash
NCCL_IB_DISABLE=0
NCCL_IB_HCA=mlx5_0
NCCL_IB_MERGE_NICS=0
NCCL_SOCKET_IFNAME=eno8303
GLOO_SOCKET_IFNAME=eno8303
NCCL_SOCKET_FAMILY=AF_INET
NCCL_DEBUG=INFO
```

After the run, check the NCCL logs for `NET/IB` and the expected HCA/interface
before describing the TorchRec lane as NCCL-IB. If the HCA name is different,
check for the actual device name selected by NCCL instead.

## Deployment Record

Write `<output_dir>/deployment.md` before running:

```text
模型: dlrm
传输: <PS_TYPE>
client:
  - ssh_host=<ssh_host>, ssh_port=<ssh_port>, repo=<repo_root>, ip=<client_ip>, gpu=<gpu_id>, node_rank=<rank>, nproc_per_node=<nproc>
ps:
  - ssh_host=<ssh_host>, ssh_port=<ssh_port>, repo=<repo_root>, ip=<ps_ip>, ps_port=<ps_port>, shard=<shard_id>
client 部署: single-node | distributed
PS 部署: single-ps | sharded-ps
分片: <num_shards>, hash_method=city_hash, max_keys_per_request=<value>
dataset: <dataset_path>
runtime: <runtime_dir>
output: <output_dir>
```

## Summary Format

Generate `<output_dir>/summary.md` from `recstore_main.csv`,
`recstore_main_agg.csv`, or the matrix runner's `summary_e2e.csv` after the E2E
benchmark finishes. Keep only these three sections:

1. `Workload 说明`
2. `E2E 吞吐（samples/s，...）`
3. `E2E 延迟分解（ms，...）`

Under `Workload 说明`, before the workload table, record the benchmark
hyperparameters in Chinese prose. Include at least: model, transport, client
list, PS server list, client deployment, PS deployment, shard count, shard
mapping, dataset path, runtime directory, output directory, batch size,
embedding dimension, num embeddings, steps, warmup steps, repeat, read mode,
prefetch depth, RecStore index type, and GPU placement.

Use `M` for values >= 1,000,000 and `K` for values >= 1,000. Include repeat
mean and CV columns only when repeat >= 3.

For any RecStore vs TorchRec comparison, include an E2E latency breakdown, not
only throughput. At minimum report `batch_prepare_ms`, `input_pack_ms`,
`embed_lookup_local_ms`, `dense_fwd_ms`, `backward_ms`, `optimizer_ms`,
`sparse_update_ms`, and `step_total_ms` where available. This breakdown is
required even when the throughput ordering looks reasonable, because RecStore
and TorchRec runner paths can differ in sparse update, ID deduplication, and
batch preparation work.

For multi-rank CSVs, compute aggregate throughput by grouping rows by `step`,
summing per-rank `samples_per_sec` within each step, then averaging over steps
after excluding warmup rows. Do not average all rank rows directly, because that
produces a per-rank throughput instead of job throughput.

If a distributed parent process exits nonzero but all expected `rank*.csv`
files exist, verify the rank CSVs before treating the run as failed. Check the
expected rank count, row count, warmup markers, and final step. If complete,
manually merge the rank CSVs, record the parent-process failure reason in the
report, and do not silently hide the issue.

## Reporting Rules

- Do not claim tests pass unless the commands completed successfully.
- If a PS correctness test for the requested backend fails, stop before E2E and
  report the log path.
- If an E2E command exits nonzero, report the command and log path in the final
  response.
- Keep generated project-facing report text in Chinese.
- If a TorchRec distributed run is reported, state whether NCCL-IB was actually
  observed in the log. If not verified, label it as unverified.

## Current Bring-up Notes

- `cache_ps.servers` and `distributed_client.servers` must describe the same PS
  shard layout.
- Treat `hash_method`, `num_shards`, and `servers` as separate routing fields.
- Do not assume `shard_id == server list index`; route by the explicit shard id.
- For single-PS bring-up, prefer `DRAM_PET_HASH` first, then add non-default
  index types, extra TorchRec baselines, or RecStore optimization ablations only
  after the requested PS backend is stable.
- The default RecStore main lane uses `--prefetch-depth 0` on the `prefetch`
  path, which is same-batch issue+wait. Treat `--read-mode direct` and
  `--prefetch-depth > 0` as requested ablations, not mandatory comparison lanes.
