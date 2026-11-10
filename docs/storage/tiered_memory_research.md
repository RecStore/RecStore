# DRAM and SSD tiering optimization

The target is 40% of live value bytes on SSD, with at least 30% of matched
all-DRAM storage throughput and 80% of matched all-DRAM model throughput.
On 2026-11-10 (Asia/Shanghai), the user explicitly selected the real DLRM
workload for final acceptance and retained synthetic YCSB results as boundary
evidence. Under that accepted scope, the implemented asynchronous exchange
meets both thresholds; the final evidence is in
`results/tiered_acceptance_11101709/`.
The measurements below were collected before integration with the newer
main-line APIs and optional LRU read cache. They used no SSD read cache.
The integration checks cover correctness and do not provide new throughput
measurements.

SSD residency and the fraction of requests served from SSD are different
quantities. Index memory, allocator overhead, staging buffers, and any DRAM
cache must be reported separately. Fixed-size rows permit a row-count residency
ratio; variable-sized rows require a byte-weighted ratio.

## Research and applicability

| Work | Mechanism | Application to RecStore |
| --- | --- | --- |
| [Memtis, SOSP 2023](https://multics69.github.io/pages/pubs/memtis-lee-sosp23.pdf) | Classify pages from the access distribution and choose page sizes according to subpage skew. | Rank embedding rows against a fixed DRAM budget; avoid admitting cold neighbors merely because they share an SSD page. |
| [S3-FIFO, SOSP 2023](https://jasony.me/publication/sosp23-s3fifo.pdf) | Filter one-hit objects with a small FIFO before main-cache admission. | Evaluate a probation queue and bounded frequency counters before maintaining an LRU list on every hit. |
| [Nomad, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/xiang) | Non-exclusive tiering, transactional migration, and shadow copies move migration off the execution path. | Keep a valid SSD backing copy of clean promoted rows so clean eviction needs no write. Training updates must invalidate or version the shadow. |
| [Colloid, SOSP 2024](https://www.cs.cornell.edu/~ragarwal/pubs/colloid.pdf) | Place data according to measured loaded access latency rather than unloaded tier latency. | Use observed SSD completion latency and queue occupancy to regulate migration traffic. Do not saturate foreground reads with demotion writes. |
| [Tiered Memory Management Beyond Hotness, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/liu) | SOAR and ALTO account for latency and memory-level parallelism when deciding performance importance. | Prioritize rows that cause exposed training stalls; request count alone misses batching, deduplication, and overlap. |
| [Scalable Far Memory, SOSP 2025](https://panjason.github.io/assets/papers/mage.pdf) | Separate fault-in and eviction, pipeline eviction batches, and reduce shared accounting contention. | Keep dirty demotion out of foreground reads and bound evictor concurrency; measure the cost of synchronization rather than optimizing policy precision alone. |
| [Tierce, SOSP 2026](https://huaicheng.github.io/p/sosp26-tierce.pdf) | Attribute exposed stalls to individual colocated workloads using workload-local overlap and latency signals. | Distinguish model-visible lookup waits from PS worker load and unrelated host traffic. Its evaluation uses a NUMA latency proxy for CXL, not SSD. |
| [Jenga, 2025 preprint](https://arxiv.org/abs/2510.22869) | Group related allocations and smooth hotness decay to reduce misplaced hot objects and migration thrashing. | Evaluate bounded recent-history counters and separate admission/eviction thresholds; embedding-row semantics are more useful here than allocation stack traces. |

These adaptations are engineering hypotheses, not results demonstrated by the
papers for RecStore. The memory-tiering papers primarily address byte-addressable
memory. SSD requires explicit I/O, block amplification, and dirty writeback.
Their reported speedups are not predictions for DRAM/SSD tiering.

The [OSDI 2026 program](https://www.usenix.org/conference/osdi26/technical-sessions)
also lists NEMO memory telemetry and OBASE object-aware placement. They motivate
further investigation of measurement overhead and hot/cold object fragmentation;
no numerical result from those papers is used as a RecStore acceptance result.

## Current implementation

`HybridValueStore` encodes DRAM or SSD placement in each value handle and
reserves DRAM slot bytes up to the placement watermark. The independent LRU
read cache, now available on main, remains disabled by default. Enabling it
reserves part of the configured DRAM capacity for cached SSD copies and reduces
the primary placement budget; see [the configuration guide](composite_kvengine.md).
Online exchange uses its own opt-in policy and changes primary placement rather
than adding cached copies. Comparisons against the measurements below must keep
the read cache disabled and verify the final residency.

The implemented increments:

- Groups tiered batch allocations by medium and calls the SSD batch allocation
  path, preserving page-level write coalescing.
- Reads each distinct SSD slab page once per batch, shares its image across
  duplicate and same-page rows, and supports rows crossing page boundaries.
- Atomically reserves the actual DRAM slot capacity to enforce the watermark
  under concurrent allocation.
- Reports live placement and SSD read-row counts alongside physical page reads.
- Supports opt-in immutable key profiles through `tiering.dram_keys_path`.
  Only listed keys may use DRAM, still subject to the slot-byte watermark.
- Waits for each bounded io_uring completion window instead of busy polling.
  The window fits in the completion queue; a 97-page regression exercises
  multiple windows and a tail.

The storage benchmark now uses equally sized read and write batches in
`batch_get_flat` mode. Previously, a read action counted 500 keys but a write
action counted one, so nominal YCSB A/B did not represent their per-key ratios.
An explicit `load_batch_keys` option permits matched PS-style batched loading;
its default remains single-key loading. All comparison lanes must use the same
load mode. The factory header is included explicitly so static Release builds
retain engine registrations.

## Feasibility checks

Let `D` be matched all-DRAM key throughput, `q` the fraction of read keys that
need SSD, `a` the physical SSD pages read per such key, and `I` the sustained
foreground page-read rate. A necessary read-only condition for storage retention
`r` is:

```text
I >= r * D * q * a
```

For example, if `D=50M keys/s`, uniform placement has `q=0.4`, and `a=1`,
30% retention requires 6M page reads/s, approximately 24.6 GB/s of 4 KiB reads.
This is an illustrative bound, not a measured device limit. Batch latency and
CPU overhead can impose tighter bounds. Uniform traffic has no predictable hot
set to exploit; scrambled Zipf and real traces must be measured separately.

On the tested host, `/tmp` is on RAID0 `md0`, backed by `nvme1`, `nvme2`, and
`nvme5`. Each reports a current 8 GT/s, four-lane PCIe link. Using the
[PCIe Gen3 payload-bandwidth formula](https://cdrdv2-public.intel.com/332995/332995-skl-io-platform-datasheet-vol1_rev004.pdf),
their aggregate theoretical link rate is 11.815 GB/s before protocol overhead.
The measured uniform C all-DRAM baseline is 80.290M keys/s. Under the explicit
one-page-per-cold-row assumptions above, 30% retention requires 39.464 GB/s;
even the idealized link bound corresponds to only 8.98% retention. This bound
does not apply to a trace with fewer SSD requests, compression, or an effective
layout that materially reduces pages per requested row.

The follow-up small-transfer experiment in `results/tiered_sector_11100608/`
uses the same fully written 512MiB file, 16 threads, direct I/O, and three
repeats per setting. At 200 reads per thread/batch, 512B averages 627.1K IOPS
and 4KiB averages 643.7K. Registered 512B buffers at batch 17 do not improve
the request rate. These are measured I/O-stack references, not mathematical
hardware limits: physical ordering, request coalescing, layout, and host load
can change them. No sector-read API was added after this negative result.

The drives identify as Samsung `MZ1LB960HAJQ-00007`. The manufacturer's
[PM983 M.2 data sheet, Table 3](https://www.stesz.com/Upload/SAMSUNG/PM983/PM983-M.2-Datasheet.pdf)
lists up to 400K random 4KiB read IOPS for that 960GB model. Vendor reference
numbers must not be treated as a strict bound for a different request size.

A fresh 36-row matrix with the current verified code retains A/B/C throughput
of 27.28%/9.47%/1.60% on uniform and 25.39%/9.90%/1.95% on scrambled Zipf.
Every run exits successfully; the report exposes the final SSD ratio and actual
operation mix. These lanes use allocation placement with online admission off.
A separate diagnostic removes placement uncertainty: it gives the synthetic
Zipf generator's most popular 60% of ranks to DRAM, without learning or
migration. Across three matched repeats, this rank oracle retains only 11.96%,
with 6.2167% of reads still reaching SSD and exactly 40% of rows resident there.
It is not a learned policy or an acceptance result. Under this tested read
path, eliminating placement mistakes alone does not meet the full matrix gate.

The original synthetic benchmark fills each worker's values with a repeated
byte. A compression result on that low-entropy content cannot establish the
same result for embedding values. Any future compression experiment must
include an independent incompressible control and account for resident bytes;
labeling cached compressed values as metadata would not satisfy the capacity
constraint. The successful real-trace result and the failed synthetic matrix
remain separate evidence.

For a serial execution model with baseline storage-critical fraction `f`,
storage retention `r`, and no additional overlap, model retention is
`1 / (1 + f * (1/r - 1))`. If `r=0.3`, 80% model retention requires `f<=0.1071`.
The two throughput targets therefore need separate acceptance tests. Storage
retention alone does not establish model retention.

## Reproducible profile construction

`tools/benchmarks/build_tiered_dram_profile.py` reads only a historical prefix
of a signed-integer `[samples, features]` NumPy sparse array. It applies the
same absolute-value/modulo normalization as the tested workload. Use the
actual per-feature cardinalities, not the per-table cap repeated 26 times.
Both strategies rank observed keys by historical occurrence count. `density`
also assigns unseen rows the table's `singleton_count / unseen_row_count`
estimate; `frequency` leaves their score at zero. Ties use ascending global ID.
This table estimate is our heuristic, not a metric implemented by Memtis or ALTO.

For example, with a small two-table dataset:

```bash
python3 tools/benchmarks/build_tiered_dram_profile.py \
  --sparse-path /path/to/day_0_sparse.npy \
  --cardinalities '[200000, 39060]' \
  --profile-end 160000 --dram-rows 143436 \
  --strategy density --key-encoding fused --fusion-k 30 --tensor-tag 0 \
  --output-dir "results/tiered_profile_$(date +%m%d%H%M)"
```

The output is `dram_keys.bin` plus provenance in `profile.json`. The directory
must be new. The binary is a sorted, unique, native-endian `uint64_t` array,
without a header. Its metadata records host byte order and checksum; do not
move it to a host of another byte order without conversion.

PS fused keys are `(tensor_tag << 56) | (feature_id << fusion_k) | local_id`.
For fused training, use the master table's parameter tag. Storage benchmarks
using contiguous global IDs require `--key-encoding contiguous`. Verifying
both total live rows and the final SSD ratio caught an otherwise silent
namespace error in the first PS profile experiment.

All profiles in the held-out experiment use `[0, 160000)` samples; evaluation
uses the disjoint later `[160000, 192768)` samples. With 1,721,198 fixed 512B
rows, 1,032,718 profile keys permit 60% of the logical payload in DRAM. Their
profile vector itself costs 8,261,744 bytes. Index memory, allocator metadata,
temporary I/O buffers, communication regions, and COW staging are additional
costs, reported separately. This is not a whole-process 60% RSS limit.

The three E2E tiered lanes use the same 622,061,206B DRAM allocator. Profile
lanes allow a `.90` physical watermark instead of `.85` so an update batch can
allocate its new copies before retiring old ones. This uses existing allocator
headroom, not a larger allocator or an extra payload cache. The immutable
eligible set still caps steady logical DRAM rows at 1,032,718; final quiescent
statistics verify 40.0000% SSD. A changing workload or a full physical-memory
budget needs separate validation.

## Measured outcome and remaining limit

The model numbers below are historical. Subsequent inspection found that the
old SGD copy fallback modified SSD read buffers without writing them back.
Five-step loss agreement did not establish persistence for every cold row.
They must not be used to accept the training target. The next iteration adds
explicit mixed-tier SGD writeback, known-value cold-row tests, and longer
model loss comparisons; its evidence is kept separately in
`results/tiered_online_11100405/`.

On the held-out window, the density profile has a 96.685% deduplicated hit
estimate, versus 92.042% for frequency alone. The storage-only trace uses
16 independently seeded circular readers and 500-key calls; its order and
batch boundaries differ from the model's 1024-sample batches. Comparisons
within each layer use matched inputs and settings.

Historical E2E runs in `results/tiered_profile_11100244/` reported 92.53%
retention by the arithmetic mean of step throughput. Those model results are
ineligible for acceptance because of the missing cold SGD writeback described
above, even though a five-step loss comparison agreed. They are retained for
audit, not as evidence that the model target has been achieved.

The corresponding storage-only density trace measures about 17.2M keys/s
against 106.7M all DRAM, approximately 16.1% retention. **The 30% storage target
is not met.** A separate read-only microbenchmark of the same RAID file
measures roughly 0.61M random 4KiB reads/s at 16 threads and 17 reads/batch.
Registered buffers, 1KiB requests, and fewer submission threads with larger
batches did not materially lift that rate. They were not added to production.
CPU samples show substantial time in the kernel RAID and HPET clock paths;
no system clock or device configuration was changed.

At the measured DRAM rate and page amplification, 30% storage retention would
need roughly twice this random-read rate, or a deduplicated DRAM hit rate above
about 98.3%. Static historical placement alone has not achieved that hit rate.
These numbers motivate online exchange; they do not establish that any online
policy will meet the target.

Raw commands, losses, residency snapshots, and Chinese reports are preserved
locally under `results/tiered_profile_11100244/`; the matched E2E report is
`e2e_verified/summary.md`, and the final storage trace report is
`storage_window_verified/summary.md`. The original 10M-row uniform/scrambled
Zipf A/B/C matrix is kept under `results/tiered_research_11100143/`. Generated
results are excluded from Git.

## Next policy increment

The current opt-in online implementation adds second-miss admission, bounded
ghost history, FIFO replacement, and a writer-priority read-copy/publication
guard. Its budget scales by request size because LOCAL_SHM combines many
500-key windows into one call. It also fixes explicit SSD SGD writeback;
known-value tests cover duplicate gradients and missing rows. Eighty-step
model comparisons agree with aligned-initialization TorchRec within
5.96e-8 loss difference.

Three current storage repeats at 40.0000% SSD measure 70.662M keys/s against
101.067M all DRAM, retaining 69.92% on the finite held-out trace. Three model
repeats retain 86.13% by arithmetic mean of per-step throughput, but only
73.02% by total samples divided by measured step time. The first online run
has a roughly 420ms step with 395ms inside the backend update stage. The
slower run remains included. The 80% model target is therefore still open.
Details are in `results/tiered_online_11100405/summary.md`.

The opt-in asynchronous experiment (`tiering.async_admission=true`) now
separates private destination preparation from publication. A single worker
and one queued batch bound staging; each batch contains at most 256 row pairs.
The pending batch merges candidates from small concurrent requests.
Hot snapshots reuse foreground read/update buffers. Victim copies are captured
under the guard, private destinations are written without it, and a global
mutation generation plus source-handle checks reject stale publication. This
conservative global data-mutation generation may reject useful batches on unrelated writes;
per-row versions are deferred until measurements justify them. The allocator
watermark accounts for staging, and quiescent statistics drain the worker.
Worker errors propagate to drain and later guarded operations; destruction
joins the thread before destroying storage.

A deterministic blocked-write regression verifies that foreground DRAM SGD
completes during background SSD preparation and that the resumed worker drops
its now-stale victim copy. Additional checks cover explicit cold SGD writeback,
duplicate gradients, and worker failure propagation. Timing, numerical checks,
and new matched performance runs are kept under
`results/tiered_stall_11100511/`. The first asynchronous attempt retained only 6.70% storage throughput and
64.59% pooled model throughput. It incorrectly invalidated unrelated snapshots
on value-preserving migration and could not merge small queued requests.
Keeping those failed runs, we restricted generation changes to foreground data
mutations, filtered already-moved source handles, and merged pending requests
within the same 256-row cap. Three new storage repeats retain 77.45%, compared
to 72.10% for synchronous exchange in that matched set. All asynchronous repeats
end at 40.0000% SSD with no capacity rejection. The revised worker's independent 80-step loss comparison agrees within
5.96e-8. Its three normal-file model runs retain 84.42% by pooled samples divided
by measured step time plus final drain (77.357K all DRAM versus 65.306K
asynchronous). The matched synchronous lane retains 74.40%. These results are
in `coalesced_normal/`; the storage result is in `storage_coalesced/`.

A single diagnostic with fully written SSD files retains 86.92% asynchronously.
It records file initialization time separately and is not merged with the
three-repeat normal-file result. It does not establish that file allocation
or kernel locking caused the earlier outlier. The final ownership fix replaces retained thread-local slab scratch with one
scoped aggregate buffer per batch. Both Release builds pass storage/PS checks,
and three asynchronous lifecycle cases pass five repeats, including balanced
scratch allocations on injected failure. The final 80-step direct loss test
agrees with aligned-initialization TorchRec within 5.96e-8.

Three final matched repeats measure 78.224M storage keys/s
against 102.105M all DRAM, retaining
76.61%. Pooled model throughput is
66.896K samples/s against
72.941K, retaining
91.71%. All tiered runs finish at 40.0000% SSD.
These final results are in `final_storage/`, `final_e2e/`, and
`loss_alignment_final.md` under `results/tiered_stall_11100511/`.

This satisfies both thresholds for the finite held-out workload, which repeats
32 query windows and ultimately fits its accessed hot set in DRAM. The user
accepted that real DLRM workload as the final gate on 2026-11-10. The 10M-row,
128B uniform/scrambled-Zipf A/B/C matrix remains boundary evidence and does not
pass across all cases. Acceptance does not extrapolate the trace result to a
growing working set or a one-pass stream.
A timing pilot or a faster rerun does not erase the earlier slow result.

Measure frequency after client deduplication, foreground SSD stall time, physical
pages per request, dirty writeback traffic, and promotion usefulness. Profile
on an earlier time window and evaluate on a disjoint later window. An in-sample
oracle is only a bound. Charge metadata and cache payload against the DRAM
budget rather than adding an unaccounted cache to a 60% resident store.

A candidate admission score is predicted reuse times exposed stall saved,
minus promotion, victim writeback, and CPU costs. This is a RecStore-specific
proxy inspired by the papers, not their hardware AOL metric. Admit only above
a positive benefit margin; use hysteresis and a bounded migration budget.
Compare probationary admission with static profile-guided placement first.

A policy-only replay starts with the historical density profile, admits on a
second SSD miss, and uses bounded CLOCK replacement at the same logical
60/40 split. The first two sequential passes over the held-out query window
miss 3.275% and 2.866% of keys; the third pass misses none. It makes 6,759
promotions and the same number of demotions across those passes. Without SSD
shadows, each demotion still needs writeback. This small finite-window result
shows why online admission is worth testing; it excludes synchronization,
allocation, and I/O costs and is not an engine throughput result. A phase-shift
or one-pass stream must also be evaluated before relying on the warm result.

Online mode disables borrowed direct-row and RDMA views and keeps read copies
under the shared guard. Prepared destinations remain private until both sides
are complete; the generation check prevents a snapshot from overwriting newer
gradients. This protocol does not make arbitrary external raw pointers safe
across asynchronous eviction.

Lookahead is another independent experiment: it must respect sparse update
visibility across steps and pass loss-alignment checks. A prefetch depth above
zero cannot be credited with performance gains if it reads stale parameters.

## Validation

Regression tests cover same-page and duplicate reads, page-boundary values,
tiered batch round trips, static layouts, and existing engine combinations.
Performance runs record Release build settings, exact commands, allocator,
actual residency, operation mix, distributions, and raw failures under
`results/`. Reports keep storage-only, PS/network, and model measurements
separate. The finite-trace thresholds now pass with matched semantics and residency.
The accepted real-DLRM goal is verified. Synthetic matrix failures and
workloads whose active set does not fit the DRAM tier remain documented
limitations, rather than acceptance gates.

## Exposed storage paths

A follow-up inventory and 12 read-only I/O diagnostics are recorded in
`results/tiered_constraints_11100626/`. The two P5920 drives form md1, but
the container exposes only bound files from that volume, with no raw device
node or standalone data directory. The workspace uses a separate overlay
volume whose physical backing is unproven. That path measures about 25K IOPS
versus about 655K on the authorised /tmp path, so it does not provide a faster
alternative. No disk was mounted or reformatted and no clocksource was changed.

Using the newest matched read-only baselines, the one-request-per-cold-row
model requires 9.417M IOPS for uniform and 1.450M for the synthetic-rank Zipf
oracle at 30% throughput retention. The corresponding 4KiB traffic is 38.57
and 5.94 GB/s. These are conditional requirements, not new measured successes.
New hardware needs its own matched all-DRAM baseline and sustained I/O test;
vendor aggregate peaks alone cannot establish the target. The original
residency and throughput gates remain unchanged.
