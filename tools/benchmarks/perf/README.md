# Performance case registry

`cases.json` is the single source of truth for what "the performance tests" are.
`tools/benchmarks/perf_suite.py` executes it, aggregates repeats and compares
against per-machine baselines.

## Execution model

| Label | Runs where | Purpose |
|---|---|---|
| `perf_smoke` | normal `ctest` lanes, including GitHub CI | Tiny canary: binary starts, metric parses. No number comparison. |
| `perf` | owned hardware only (node190/191) | Real measurement with repeats and baseline checks. Excluded from CI via `-LE "^perf$"`. |

GitHub-hosted runners have no RDMA, shared CPUs and no stable machine identity,
so they must never produce numbers. They only prove the harness still works.

## Case fields

```json
{
  "id": "kvengine_c_read",
  "layer": "storage | ps_transport | e2e",
  "runner": "kvengine | ps_local_shm",
  "labels": ["perf_smoke", "perf"],
  "repeats": 3,
  "noise_floor_pct": 5.0,
  "params": {},
  "metrics": [
    { "key": "throughput_ops_sec", "unit": "ops/s", "direction": "higher" }
  ]
}
```

- `repeats` — default sample count for `run` (override with `--repeats`).
- `noise_floor_pct` — a change smaller than this is not a result. Values follow
  `docs/agent/perf.md` / `tmp/docs/memory/performance-measurement-discipline.md`
  (storage 5%, transport 4-10%, e2e 15%).
- `direction` — `higher` is better for throughput, `lower` for latency.
- Parameters stay in the registry on purpose: a run that is configured ad hoc
  cannot be reproduced or compared.
- `kvengine` params become `--key=value` flags. The optional SSD/zipfian flags
  (`zipfian_alpha`, `zipfian_scramble`, `ssd_path`, `ssd_capacity_bytes`,
  `ssd_io_backend`, `ssd_queue_depth`) are appended only when the param is
  present. `ssd_path` may contain `{work_dir}`, which expands to the per-repeat
  scratch directory (`work/<case>/rNN`); that SSD tree is deleted and recreated
  before every repeat, because a leftover file silently changes the residency
  pattern. A `ssd_path` outside the per-repeat work directory is rejected rather
  than cleaned.

## Commands

```bash
# CI canary (~5 s, CPU only)
python3 tools/benchmarks/perf_suite.py run --label perf_smoke

# Owned hardware: repeat, write artifacts, compare with the baseline
python3 tools/benchmarks/perf_suite.py run --label perf --repeats 3 --check

# Record a baseline for this machine (explicit, reviewed)
python3 tools/benchmarks/perf_suite.py run --label perf --repeats 3 --write-baseline

# Hot-path PR: interleaved A/B of two builds on the same machine
python3 tools/benchmarks/perf_suite.py ab \
    --case kvengine_c_read_full --a-build /tmp/base-build --b-build ./build --check

# Same binary, two configs: A/B the parameters instead of the build
python3 tools/benchmarks/perf_suite.py ab --case kvengine_tiered_scramble \
    --a-param zipfian_scramble=false --b-param zipfian_scramble=true --repeats 3

# Collect every metrics.json under results/ into one CSV
python3 tools/benchmarks/perf_suite.py aggregate --csv results/perf_history.csv
```

`--a-param`/`--b-param` take `key=value` pairs that override the case params for
one side (values are parsed as JSON when possible, so `true` is a bool). Both
`--a-build`/`--b-build` default to `./build`, so a config-only A/B needs neither.

Exit codes: `0` ok, `1` execution error, `2` regression beyond the noise floor.

## Artifacts

Each run writes one directory (default `results/perf_<MMDDHHMM>/`):

- `metrics.json` — every sample, machine readable
- `summary.md` — medians, spread, baseline ratio, regressions
- `env.json` — commit, build options, CPU, NIC, loadavg, binary hashes
- `logs/` — raw benchmark output per case and repeat
- `work/` — per-case scratch data (DRAM paths, local_shm config)

`env.json` is what makes a number defensible: without the build options,
pinned cores, load average and binary hash it should not be quoted.

## Baselines

Baselines live in `baselines/<env-id>/<case-id>.json`, one bucket per machine
(`env-id` defaults to `<hostname>-<cpu-model>`; the committed reference bucket is
`node190-intel-r-xeon-r-gold-5420`). Pass `--env-id` only to use a different
alias deliberately. Rules:

- A new machine means a new bucket; never edit an old machine's baseline.
- Baselines are only updated by an explicit `--write-baseline` run whose diff is
  reviewed. Nightly runs never overwrite them, otherwise a slow drift would be
  silently folded into the reference.
- `--check` reports `no-baseline` for machines without a bucket instead of
  failing, so the suite is usable before a baseline exists.
- `--check` also records the CPU clock ceiling (`cpu_mhz_max`); if the current
  run differs from the baseline by more than 15% the rows are reported as
  `env-mismatch` instead of a regression.
- If a run's own repeats disagree by more than the case's noise floor, the row
  is reported as `unstable` and is **not** gated. A noisy median is not a
  measurement; re-run on an idle machine or use `ab`.

`ab` decides differently, because its repeats are interleaved inside one
session: a change counts when the two sample ranges are disjoint. That keeps a
3x effect reportable at 15% spread, while a 6% wobble inside the floor with
overlapping ranges is reported as `unstable` instead of a false regression.

## Known variance (node190)

Do not baseline a case until its repeats are stable *across* invocations, not
just within one. The original 1M-record / 8-thread YCSB A case on node190 read
either ~7.2M or ~11.1M ops/s (1.5x) depending on the process run, with <2%
spread inside each run and identical commands, binaries, clock (~2.69 GHz) and
CPU affinity. The state survived minutes and flipped between otherwise
identical suite invocations, and it could not be attributed to clock, NUMA
placement or capacity within the investigation, so the case was replaced by a
cache-resident single-thread variant (50k records) whose repeats stay within
0.3%. Memory-bandwidth scaling belongs in e2e, not in a regression baseline.

`ab` is meant for the `perf` cases (multi-second runs with repeats). The
`perf_smoke` cases are one second long and can legitimately differ by 30% even
between two identical builds; they are canaries, not comparisons.

`kvengine_a_mixed_full` is recorded as a snapshot too: on 2026-11-06 the same
binary and params produced 5.85M / 7.21M / 11.15M ops/s across invocations, with
<1% spread inside each run. A later check matched the baseline within 0.3%, so
the floor is only trustworthy when several invocations agree; otherwise use
interleaved `ab`.

### `kvengine_tiered_scramble` (issue #266)

This case is not stable yet and its recorded baseline is a snapshot, not a
reference. Four baseline/check invocations on 2026-11-06 (same binary, same
params, idle machine) produced medians of 505K, 394K, 646K ops/s with 16-40%
spread inside each 3-repeat run. Controls on the same machine at the same time:

- `kvengine_c_read_full` (DRAM only): 195.0M ops/s, 0.1% spread.
- same tiered case with `zipfian_scramble=false`: 1.225M ops/s, 2.8% spread.
- same tiered case with `distribution=uniform`: 1.117M ops/s, 1.4% spread.

The access sequence is deterministic (fixed per-thread seeds and a fixed
scramble stride), so the remaining per-process variable is which keys the
16-thread load put into the 10% DRAM slice. With `zipfian_alpha=0.9` the
hottest few ranks carry a large share of the accesses; whether they happen to
land in DRAM moves the SSD op share by roughly ±10 percentage points, i.e.
the same order as the observed swing. Treat the 5% noise floor as unusable for
this case until #266 changes placement, and validate with interleaved `ab` runs
plus cache hit/miss counters rather than `--check` across invocations.

Interleaved `ab` on the same binary and machine is the measurement that does
hold up, because the two configs alternate inside one session:

| side | params | median ops/s | B/A |
|---|---|---:|---:|
| A | `zipfian_scramble=false` | 1.901M / 1.938M | — |
| B | `zipfian_scramble=true` | 597K / 554K | 0.31 / 0.29 |

i.e. the load-order effect alone is a **3.2-3.5x** difference, reproduced twice
with disjoint sample ranges. That is the gap #266 has to close.

#### Read cache (`tiering_cache_policy`)

The tiered store can keep a copy of read-hot SSD values in DRAM
(`value.tiering.cache_policy=lru`). It is **off by default**, so the registered
case above still measures placement alone; the cache is measured with `ab` on
the same binary:

```bash
python3 tools/benchmarks/perf_suite.py ab --case kvengine_tiered_scramble \
    --a-param tiering_cache_policy=none --b-param tiering_cache_policy=lru \
    --repeats 3 --a-build ./build_release --b-build ./build_release
```

The `YCSB_RESULT` line carries `tiered_cache_*` counters next to the
throughput, so the ratio can be explained rather than just observed. They are
parsed as *optional* metrics: with `cache_policy=none` the fields are absent, so
they must never be added to a case's `metrics` list (that list is checked for
presence on every repeat). `ab` prints them per repeat next to the declared
metrics.

Knobs (all under `value.tiering`, all exposed as `--tiering_cache_*` flags in
`benchmark_kv_engine`):

| key | default | meaning |
|---|---|---|
| `cache_policy` | `none` | `none` or `lru` |
| `cache_budget_ratio` | `0.05` | share of the DRAM layer the cache may use |
| `cache_shards` | `16` | LRU shards (one mutex each) |
| `promote_min_misses` | `2` | misses before a value is admitted; `2` filters one-shot misses |

Reported counters: `tiered_cache_budget_bytes` (configured),
`tiered_cache_effective_budget_bytes` (budget actually in force after the shared
DRAM limit), `tiered_cache_dram_limit_bytes`, `tiered_cache_bytes`,
`tiered_cache_entries`, `tiered_cache_hits`, `tiered_cache_misses`,
`tiered_cache_hit_rate`, `tiered_cache_promotions`, `tiered_cache_evictions`,
`tiered_cache_alloc_failures`, `tiered_cache_admission_rejects`.

The budget is **reserved**, not shared dynamically: primary placement stops at
`high_watermark * (capacity - budget)`, and the cache is additionally capped so
that primary + cache stays under `high_watermark * capacity`. Without that
reservation the DRAM allocator runs out and `ConcurrentSlabMemoryPool` aborts
the process (`SlotCapacity` counts user bytes, the allocator charges 8 more per
slot, so the 0.85 watermark is already ~0.90 of real memory).

**The default budget is 0.05, not 0.5, because e2e says so.** Micro-benchmarks
with `batch_get_flat(64)` amortise SSD latency across the batch, so they keep
rewarding a bigger cache budget: `budget=0.5` measured 3.43x (min_misses=1) and
3.97x (min_misses=2), while `budget=0.05` measured 2.87x. The e2e lane
(`ps_pipeline`, 15% SSD rows, 64 steps) inverts that ordering, because every
row is a separate read and the primary-placement rows the cache budget takes
away become visible SSD access:

| `cache_budget_ratio` | e2e step ms (median) | vs `none` |
|---|---|---|
| `none` | 52.26 | 1.00 |
| 0.03 | 45.93 | 0.86 |
| **0.05** | **44.81** | **0.86** |
| 0.07 | 45.05 | 0.86 |
| 0.10 | 46.77 | 0.89 |
| 0.25 | 58.89 | 1.13 |
| 0.50 | 73.64 | 1.41 |

So: 0.05 is the plateau (three repeats, emb 37.37-37.75 ms vs 44.78-45.27 ms for
`none`, disjoint), and 0.5 loses 41% end to end. Do not quote the micro-benchmark
ratio as a production number; quote the e2e step-time delta (~14%, i.e. ~17% more
samples/s).

That run promoted on every miss, so the cache was churned by one-shot misses:
`promotions` matched `misses` (4.39M each) and the hit rate sat ~18 points
below what the workload's skew allows. `promote_min_misses=2` fixes the admission
side (a key is copied in only on its second miss); compare it the same way:

```bash
--a-param tiering_promote_min_misses=1 --b-param tiering_promote_min_misses=2
```

`tiered_cache_admission_rejects` counts the misses that filter rejected, so the
promotion count can be checked against `misses - rejects`.

## Adding a case

1. Pick the runner (`kvengine`, `ps_local_shm`) or add one in `perf_suite.py`.
2. Add the case to `cases.json` with fixed parameters and a noise floor.
3. If it is safe in CI, label it `perf_smoke` and register it in
   `src/test/CMakeLists.txt`; otherwise label it `perf` only.
4. Record a baseline on the target machine and describe it in the commit.

`test_perf_suite.py` covers parsing, aggregation, evaluation and the registry
itself, so registry or threshold regressions fail in the normal `python` lane.
