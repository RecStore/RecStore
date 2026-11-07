# BagPipe with PS-owned parameters

Default (online) and planned (prescan) execution share a bounded GPU cache
controller. The PS owns embedding values and applies SGD each step; its
DRAM/SSD tiers are configured independently of GPU cache capacity.

## Execution modes

`--read-mode bagpipe` enables the runtime. `--optimization-prescan` builds
CPU access plans before training; without it, ranks exchange batch IDs online.
Both support two gradient aggregation modes:

| Option | Aggregation | PS update |
| --- | --- | --- |
| `--bagpipe-aggregation collective` | All-reduce active-row gradients and divide by world size | One designated rank submits each row; PS commits after all rank markers |
| `--bagpipe-aggregation ps` (default) | Send local unique-row gradients to their PS shard | PS sums in rank/chunk order, divides by world size, and applies SGD once |

Both modes use dense DDP collectives and ID metadata exchange. Prescan
exchanges IDs once and retains plans in CPU memory; online mode exchanges
IDs each batch.

Example additions to an existing RDMA training command:

```sh
--ps-type RDMA --read-mode bagpipe --optimization-prescan \
  --optimization-cache-capacity 8192 --bagpipe-aggregation ps
```

The `rs_demo` launcher selects a local backend for single-node deployments,
including commands with `--ps-type RDMA`.

## Memory and cache coherence

`cache_capacity` is measured in units of 64 rows. Resident GPU value storage
uses `cache_capacity * 64 * embedding_dim * 4` bytes. CPU LRU metadata maps
keys to slots. The current batch and bounded prepared-batch queue use
additional temporary tensors. Batches larger than the cache gather into their
own tensors and admit only bounded rows. Prescan plans consume CPU memory
proportional to the access trace.

Admission returns the final source row for each destination slot, giving GPU
scatters unique destinations even for oversized batches. Metadata operations
are serialized with commit completion.

After a PS commit, each rank updates cached rows accessed only by that rank
using its local gradient divided by world size. Rows modified by another rank
are invalidated and fetched on demand. Collective aggregation makes gradients
available for all affected resident rows, which can then be updated locally.
Eviction discards clean cache rows.

Plans track each local row's previous global update step. Prefetch is issued
only for rows whose preceding updates have committed; other misses wait until
consumption.
A forward waits for a pending commit when its rows depend on that step, allowing
independent forward work to overlap PS submission. Each sparse update waits
for its predecessor.

## RDMA submission

Gradients are staged asynchronously into pinned CPU buffers. A native worker
waits for producer-stream CUDA events before reading them and retains the
source and staging buffers through submission. IDs in CPU plans avoid a GPU
round trip.

The adapter packs gradients into bounded packets and submits chunks
round-robin across shards. Multiple RC writes can precede ACK waits, bounded
by QP/slot capacity. Every shard receives rank markers, including shards with
zero rows in the step.

GET windows and gradient submissions share RC slots. Before reusing slots,
the adapter materializes outstanding prefetch results in CPU buffers while
preserving their handles. Slot ownership is serialized within the adapter;
GPU computation and PS aggregation remain asynchronous.

## Module layout

Paths below are relative to `src/`. There are two distinct implementations:
the planned controller used by `--read-mode bagpipe`, and the older C++ GPU
cache controller kept for the `enable_bagpipe_cache` / local-ShM path.

PyTorch adapters supply buffer operations to the common execution layer.
`emb_read` and `local_lookup_flat` are used by the planned path;
`gpu_cache_lookup_flat` belongs to the legacy GPU cache. Adapters retain
source/staging buffers through commit completion and borrowed local-read
storage through H2D completion.

### Production planned controller (`--read-mode bagpipe`)

`--read-mode bagpipe` (or `--optimization-plugin bagpipe`) resolves through the
optimization plugin registry and runs Python-side planning with PS-owned SGD:

| Module | Responsibility |
| --- | --- |
| `python/pytorch/recstore/optim/plugin.py`, `.../optim/registry.py` | Plugin lifecycle hooks (`on_prepare`, `on_consume`, `on_step_end`, `shutdown`) and optimizer creation |
| `python/pytorch/recstore/bagpipe_cache/plugin.py` | `BagPipePlugin`: binds the planned controller and `BagPipeSparseSGD` to the training loop |
| `python/pytorch/recstore/bagpipe_cache/planned.py` | `BagPipeCacheController` used in production: CPU access plans, admission, prefetch scheduling, and sparse-gradient commit coordination for online and prescan modes |
| `python/pytorch/recstore/bagpipe_cache/optimizer.py` | `BagPipeSparseSGD`: routes gradients through `controller.update_grads()` and the PS commit protocol |
| `optimization/bagpipe/cache_metadata.h` | `CacheMetadata`: batched LRU queries, admission, invalidation, and previous-update tracking using the C++ standard library |
| `framework/op.{h,cc}` | Sparse commit submission/wait/close through `RecTensor`; cache clearing after successful table initialization or checkpoint restore |
| `framework/common/sparse_gradient_commit_runtime.h` | Input readiness, asynchronous handles, committed-step polling, timeout, and peer abort |
| `framework/pytorch/cache/` | Tensor allocation, dtype/shape/device conversion, result wrapping, and metadata bindings |
| `framework/gpu/host_memory.cu` | Host-memory registration and stream-completion retention |

`CacheMetadata` borrows input arrays for each call and returns vectors. Its
PyTorch binding, `BagPipeCacheMetadata`, exchanges compact position and slot
tensors with `planned.py`, which handles planning, gathers, autograd, and
communication scheduling.

### Legacy GPU-cache controller (`controller.py`)

`python/pytorch/recstore/bagpipe_cache/controller.py` is the older
`enable_bagpipe_cache` path for the local-ShM fast path with the C++ GPU cache
enabled. It combines the `comm`, `prefetch`, `eviction`, and `grads` mixins
and implements BagPipe-style TTL eviction and writeback; it was a drop-in
replacement for `LookaheadPrefetcher`.

This class is **not** the `read_mode=bagpipe` implementation. The package
`__init__.py` exports the planned controller, and no production entry point
imports `controller.py`; only
`python/pytorch/recstore/unittest/test_bagpipe_cache.py` does so directly. The
`prefetch.py`, `comm.py`, `grads.py`, and `eviction.py` modules belong to this
legacy path.

| Module | Responsibility |
| --- | --- |
| `python/pytorch/recstore/bagpipe_cache/controller.py` + mixins | Orchestration: TTL eviction/writeback, prefetch, gradient flush, and comm scheduling for the legacy path |
| `framework/cache/cache_aware_op.h` | Query/read-misses/fill/scatter sequence, low-hit bypass, fallback, and invalidation around cached updates/writes (`EmbUpdateCached`, `EmbWriteCached`) |
| `framework/cache/gpu_embedding_cache.{h,cu}` | Device-buffer GPU cache: lifecycle, cross-stream ordering, generation, residency, and cache SGD; API independent of Torch |

GPU cache callers supply device pointers, sizes, a device ordinal, and a CUDA
stream, retaining buffers until completion. Query waits for the miss count;
hit-only reads and ordinary mutations enqueue work. Clear/disable wait before
freeing cache storage. Explicit enable/disable/reset reset the calling thread's
bypass state; internal clearing preserves it.

Strict cache SGD sums duplicate gradients with CUDA/Thrust before updating
each row once. A missing key causes invalidation of the requested keys and a
`false` result. Best-effort SGD updates resident rows asynchronously and skips
missing keys.

Tests: `test_bagpipe_metadata` covers `CacheMetadata`;
`test_framework_op_runtime` covers the commit/op runtime;
`test_framework_gpu_cache` covers the C++ GPU cache used by the legacy path.
The planned Python controller is covered by
`python/pytorch/recstore/unittest/test_bagpipe_planned.py` and the legacy
controller by `test_bagpipe_cache.py`, but neither Python module is registered
with ctest yet.

## Sparse gradient commits

`SparseGradientCommit` aggregates sparse gradients and commits ordered updates.
Its header, client, and coordinator live in `src/ps/base/sparse_gradient_commit.h`.
Python exposes `sparse_gradient_commit_async`, `sparse_gradient_commit_wait`,
and `sparse_gradient_commit_close`. The asynchronous handle completes when
all shards report the requested committed step.

A session identifies one job on one fused table. Packets carry session, step,
rank, world size, chunk index/count, dimension, and SGD learning rate. The PS
acknowledges chunks immediately and applies the update once all ranks' chunks
arrive. Clients observe the minimum committed step across shards. Empty-rank
and all-empty-step markers advance the same protocol. Conflicting pending
retries and out-of-order steps fail; committed-step retries do not repeat SGD.

Backend errors can leave partial updates and fail the session. Recovery
restores the job and PS from a consistent checkpoint. Rank loss produces an
error or timeout. Successful shutdown drains pending work and closes every
rank's session; failure cleanup uses no new distributed collective.

## Requirements and metrics

The runtime requires fused tables, FP32 gradients, constant-rate SGD with
matching PS/client learning rates, fixed feature order, one ID per feature per
sample, and one fused sparse update per step per rank. Empty ranks participate.
Prescan requires the same ID sequence during training. Uneven rank termination,
external writers, adaptive optimizers, and elastic recovery are unsupported.

Metrics cover cache hits, demand/prefetch IDs, eviction/invalidation counts,
resident bytes, skipped dependency waits, commit-wait time, and gradient
payload bytes. Prescan startup and shutdown costs are recorded separately.
Compare modes at the same cache capacity and include final commit time.
