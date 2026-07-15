# Performance Comparison Background

This document is legacy background for interpreting performance comparisons and
labeling benchmark layers. It is not the source of truth for current benchmark
commands, defaults, lanes, or report formats.

Use the matching benchmark skill first:

- E2E RecStore/TorchRec benchmarks:
  `.agents/skills/benchmark-e2e/SKILL.md`
- PS, transport, and RDMA benchmarks:
  `.agents/skills/benchmark-ps/SKILL.md`
- KVEngine and storage-only benchmarks:
  `.agents/skills/benchmark-kvengine/SKILL.md`

When this document disagrees with a matching skill, follow the skill.

## 1. Scope

Use this guide when comparing:

- storage backends, index types, value stores, or memory placements
- BRPC, GRPC, LOCAL_SHM, RDMA, or other RecStore transports
- storage-only vs PS/network performance
- PS/network vs PyTorch/model performance
- RecStore vs third-party engines such as HPS, RocksDB, or HierarchicalKV
- a new optimization vs an existing baseline

Do not turn one benchmark lane into an architecture-level claim.

## 2. Layer Boundaries

Always label every result with one of these layers.

### Storage-only

Direct backend benchmark. This measures storage engine behavior without PS
transport overhead.

Use this layer for RecStore native backends and third-party engines unless those
engines are explicitly integrated into PS and PyTorch paths.

### PS/network

Client/server benchmark through RecStore PS. This measures client routing,
transport, server dispatch, serialization, and copy overhead.

Do not use this layer to claim storage-engine superiority by itself.

### PyTorch/model

Model integration benchmark, usually through `model_zoo/rs_demo`.

This layer includes ID preparation, lookup wait, EmbeddingBag pooling, sparse
update scheduling, backward, and Python runner overhead.

## 3. Workload Matrix

Do not use this document to choose current default lanes or parameters. The
benchmark skills define the current workload matrix for E2E, PS, RDMA, and
KVEngine work.

For any comparison, record the actual backend, PS type, index type, value store,
allocator, transport, value size, key or row scale, batch size, distribution,
runtime, warmup, repeat count, and topology used in that run.

## 4. Workflow

1. Read the matching benchmark skill and use it for commands and report format.
2. Check branch, dirty files, and stale benchmark binaries.
3. Run storage-only benchmarks first if measuring degradation.
4. Run PS/network benchmarks with matching value size, key count, operation, and
   backend where possible.
5. Run PyTorch/model benchmarks after lower layers are understood.
6. Aggregate raw results into CSV files.
7. Report ratios only when numerator and denominator use compatible operation
   and layer semantics.

## 5. Required Records

Every comparison must record:

- goal
- branch and relevant commits
- compared objects and their layers
- exact commands, runner scripts, or raw output paths
- value size, capacity/record count, batch size, thread/process count
- distribution and operation mode
- repeat policy
- aggregate CSV paths
- warnings, startup failures, timeouts, OOMs, or allocator substitutions

If a path fails before steady-state measurement, report it as startup,
reachability, configuration, timeout, or OOM failure. Do not convert it into a
throughput result.

## 6. Aggregation

Use consistent units:

- storage-only: `M keys/s`
- PS/network: `M keys/s`
- PyTorch/model: `M rows/s` plus `ms`

Storage and PS:

```text
M keys/s = throughput_keys_sec / 1e6
```

PyTorch EmbeddingBag:

```text
rows_per_step = batch_size * sparse_features_per_sample
read M rows/s = rows_per_step / (embed_lookup_local_ms / 1000) / 1e6
write M rows/s = rows_per_step / (sparse_update_ms / 1000) / 1e6
```

Layer retention:

```text
PS retention = PS M keys/s / storage-only M keys/s
PyTorch retention = PyTorch M rows/s / PS M keys/s
```

Call ratios "retention" or "remaining throughput", not speedup. If retention is
above `1.0`, explain timing-boundary mismatch instead of claiming the higher
layer is faster.

## 7. Fairness Checks Before Concluding

Before concluding, confirm:

- same value size, operation mode, and distribution, or differences explained
- same key/row scale, or capacity policy stated
- PyTorch warmup rows excluded
- third-party engines compared only at integrated layers
- GPU/HBM assumptions stated for TorchRec or GPU paths
- shared-memory and RPC results not collapsed into one generic conclusion
- CPU affinity, readiness, timeout, and allocator warnings recorded
- skill-specific preflight checks completed or skipped with reasons

## 8. Interpretation Rules

Good conclusions:

- "At storage-only layer, `DRAM_PET_HASH` has higher read throughput under this
  uniform workload."
- "After PS/network, read throughput converges, suggesting transport and
  client/server overhead dominate this setting."

Bad conclusions:

- "`DRAM_PET_HASH` is always better."
- "LOCAL_SHM must be fastest because it uses shared memory."
- "PyTorch is faster than PS because converted rows/s is larger."

Common caveats:

- LOCAL_SHM can be slower than RPC if queue, slot, synchronization, or copy cost
  dominates.
- `DRAM_UNORDERED_MAP` may lose less after PS because PS overhead hides faster
  storage engines' advantage.
- Third-party storage-only numbers do not belong in PS/network or PyTorch tables
  until those integrations exist.
- Allocator substitutions, startup workarounds, failed lanes, skips, and
  timeouts must be stated in the report instead of silently folded into a
  throughput table.
