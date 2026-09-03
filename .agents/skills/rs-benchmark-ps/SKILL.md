---
name: rs-benchmark-ps
description: Configure, run, validate, and report RecStore Parameter Server benchmarks with tools/benchmarks/run_benchmark_ps.py, covering local and cross-host topologies from explicit IP lists, RDMA/GRPC/BRPC comparison runs, RDMA module correctness checks, SSH/container execution, optimized DRAM_PET_HASH RDMA GET auto/staging-copy mode, and Chinese summary.md output.
---

# Benchmark PS

Use this as the single skill for RecStore PS benchmark work. It replaces the
old `benchmark-ps`, `rdma-module`, and `cross-host-rdma-ps` skills.

Run from the RecStore repo root. Do not run helper scripts from this skill
directory; call project scripts directly.

## Workflow

1. Confirm the current directory is the RecStore repo root, or move there.
2. Prompt the user for the benchmark configuration when values are missing:
   - test mode: `local`, `cross-host`, or both
   - transports: default `rdma`; use `grpc,brpc` or `rdma,grpc,brpc` only when
     the user asks for comparison
   - endpoint IP list: default `127.0.0.1`; for two cross-host IPs, default
     server = first IP and client = second IP
   - server/client role mapping if the IP list has more than two hosts
   - SSH targets for cross-host runs, default `xieminhui@<endpoint_ip>` in the
     current lab
   - remote container, default `recstore`; remote repo, default `/app/RecStore`
   - output directory, default `results/benchmark_ps_$(date +%m%d%H%M)`
   - workload size profile: `smoke`, `rpc-small`, `rdma-capacity`, or custom
   - client processes per IP, client threads per process, server RDMA threads,
     runtime seconds, and repeat count
3. Always compile in Release mode for PS benchmarks that feed `summary.md`:
   `cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release`.
   Optional Debug smoke-only builds may use `cmake -S . -B build`, but do not
   report Debug/O0 numbers as throughput results.
4. Validate the runner and relevant tests before non-trivial benchmark work.
5. Generate one command per compatible transport group. Split RDMA and RPC when
   their safe concurrency settings differ.
6. Immediately before `run_benchmark_ps.py`, clean leftover RecStore/PS
   processes on **every unique host** that will participate (local
   client/server, and for cross-host every SSH target / container). Use
   `tools/benchmarks/kill_bench_procs.sh` only — do not hand-roll `pkill -f`.
   - local: `./tools/benchmarks/kill_bench_procs.sh`
   - cross-host container: stream the local script into the remote container:
     `ssh <ssh_target> "docker exec -i <container> bash -s" < tools/benchmarks/kill_bench_procs.sh`
   Fail the run if cleanup exits nonzero.
7. Run `tools/benchmarks/run_benchmark_ps.py`.
8. Save generated configs, logs, `summary.csv`, and Chinese `summary.md` under
   the chosen output directory.
9. Report success, skipped rows, failures, and exact commands in Chinese.

## Configuration Defaults

Use these defaults only when the user accepts them or the request clearly asks
for a quick bring-up.

| Field | Local default | Cross-host default |
| --- | --- | --- |
| endpoint IPs | `127.0.0.1` | first IP = server, second IP = client |
| runner backend | `--execution-backend local` | `--execution-backend ssh` |
| server topology | `--server-shard-ips 127.0.0.1` | RDMA uses `--server-plan 0:<server_ssh>:25000:0`; single-shard GRPC/BRPC uses `--server-plan 0:<server_ssh>:15000:0` |
| client topology | `--client-ips 127.0.0.1` | repeated `--client-plan` entries |
| build dir | `build_release` (Release) for benchmarks; optional `build` for smoke-only | local and remote `build_release` |
| remote sync | none | `--remote-sync check` |
| remote repo | none | `/app/RecStore` |
| remote container | none | `recstore` |

Workload profiles:

| Profile | record-count | value-size | batch-keys | runtime | repeat | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke` | 10000 | 128 | 64 | 1 | 1 | Fast local/RDMA sanity |
| `rpc-small` | 5000 | 128 | 64 | 1 | 1 | Fast GRPC/BRPC cross-host smoke |
| `rpc-stress-small` | 50000 | 128 | 64 | 3 | 1 | Small GRPC/BRPC cross-host stress after smoke passes |
| `fair-small` | 10000 | 128 | 64 | 1 | 1 | RDMA/GRPC/BRPC with aligned client threads |
| `rdma-capacity` | 1000000 | 512 | 500 | 5 | 1 | Current PET_HASH RDMA GET capacity lane |

Concurrency defaults:

- RPC small comparison: `client-processes-per-ip=1`,
  `client-threads-per-process=2`, `client-load-threads-per-process=2`,
  `server-worker-threads=2`; after smoke passes, use
  `rpc-stress-small` with 4 client processes and 4 threads per process
- fair transport comparison: use `client-threads-per-process=1` and
  `client-load-threads-per-process=1` for all transports
- RDMA smoke: `client-processes-per-ip=1`,
  `client-threads-per-process=1`, `server-rdma-threads=1`
- RDMA capacity: `client-processes-per-ip=4`,
  `client-threads-per-process=3`, `server-rdma-threads=16`,
  `prefetch-depth=16`, `rdma-rc-qps-per-client-per-shard=16`,
  `rdma-rc-profile-interval-ms=0`

## Topology Construction

For local tests, prefer simple host lists:

```text
--client-ips 127.0.0.1
--server-shard-ips 127.0.0.1
--client-processes-per-ip <n>
```

For local multi-shard stress, repeat the server IP once per shard:

```text
--server-shard-ips 127.0.0.1,127.0.0.1
```

Do not describe loopback multi-shard results as cross-machine scaling.

For cross-host tests, prefer explicit plans. Use the transport's real
single-shard server port:

```text
--server-plan 0:<server_ssh>:15000:0   # single-shard GRPC/BRPC
--server-plan 0:<server_ssh>:25000:0   # RDMA
--client-plan 0:<client_ssh>,1:<client_ssh>,2:<client_ssh>,3:<client_ssh>
```

Do not reuse the RDMA `25000` server-plan port for single-shard GRPC/BRPC
smoke runs. The single-shard gRPC server listens on `15000`, and the runner
passes `--brpc_server_port` for bRPC to match the selected plan port.

Keep SSH targets and RecStore endpoint IPs conceptually separate. If the runner
or generated configs expose endpoint fields separately, endpoint hosts should be
pure IPs while SSH targets may be `user@host`.

When the user provides an IP list:

- one IP with `local`: use it for both client and server
- one IP with `cross-host`: ask for a separate client/server role mapping
- two IPs with `cross-host`: default first = server, second = client
- more than two IPs: ask for server/client role mapping and shard count

For every cross-host RDMA run, choose a unique `--rdma-control-plane-port`.

## Preflight Commands

Release build (required for throughput / `summary.md` runs) and runner
validation:

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target ps_transport_benchmark ps_server petps_server -j
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py
ctest --test-dir build_release -R 'grpc_ps_client_test|dist_grpc_ps_client_test|brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_transport_benchmark|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure
./tools/benchmarks/kill_bench_procs.sh
```

RDMA verbs check:

```bash
ls -l /dev/infiniband
ls /dev/infiniband/uverbs*
```

If verbs devices are missing, skip RDMA benchmark rows and report that only
build/unit-level validation was possible.

Cross-host remote checks:

```bash
ssh <server_ssh> "docker exec <container> bash -lc 'cd <remote_repo> && pwd'"
ssh <client_ssh> "docker exec <container> bash -lc 'cd <remote_repo> && pwd'"
ssh <server_ssh> "docker exec <container> bash -lc 'ls -l /dev/infiniband'"
ssh <client_ssh> "docker exec <container> bash -lc 'ls -l /dev/infiniband'"
ssh <server_ssh> "docker exec -i <container> bash -s" < tools/benchmarks/kill_bench_procs.sh
ssh <client_ssh> "docker exec -i <container> bash -s" < tools/benchmarks/kill_bench_procs.sh
```

Do not hand-roll broad `pkill -f`. Always use `tools/benchmarks/kill_bench_procs.sh`
on every involved host/container before the PS experiment.

## RDMA Correctness

Use this when changing or validating RDMA code paths:

```bash
cmake -S . -B build
cmake --build build --target \
  test_rdma_rc_protocol \
  test_raw_verbs_allocator \
  test_rdmaps_client_adapter \
  test_allshards_ps_client \
  petps_server \
  petps_integration_test \
  -j
ctest --test-dir build -R 'test_rdma_rc_protocol|test_raw_verbs_allocator|test_rdmaps_client_adapter' -VV
```

For client-side wait/coroutine changes, include:

```bash
ctest --test-dir build -R 'test_allshards_ps_client|test_rdmaps_client_adapter' -VV
```

Run real PetPS RDMA integration only when verbs devices exist:

```bash
cmake -S . -B build -DENABLE_RDMA_INTEGRATION_TESTS=ON
cmake --build build --target petps_server petps_integration_test -j
ctest --test-dir build -L rdma_integration -VV
```

If `ctest` returns skip code `77`, report the skip reason instead of treating
it as a pass.

## Command Templates

Local RDMA smoke:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count 10000 \
  --value-size 128 \
  --batch-keys 64 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 1 \
  --repeat 1 \
  --rdma-wait-timeout-ms 8000 \
  --rdma-rc-qps-per-client-per-shard 4 \
  --rdma-rc-slots-per-qp 1 \
  --execution-backend local \
  --server-rdma-threads 1 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --output-dir <output_dir>/rdma_smoke
```

Local GRPC/BRPC small comparison:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports grpc,brpc \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count 10000 \
  --value-size 128 \
  --batch-keys 64 \
  --client-threads-per-process 4 \
  --client-load-threads-per-process 4 \
  --runtime-seconds 1 \
  --repeat 1 \
  --execution-backend local \
  --server-worker-threads 4 \
  --output-dir <output_dir>/rpc_small
```

Canonical two-host GRPC/BRPC smoke:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports grpc,brpc \
  --server-plan 0:<server_ssh>:15000:0 \
  --client-plan 0:<client_ssh> \
  --record-count 5000 \
  --value-size 128 \
  --batch-keys 64 \
  --client-threads-per-process 2 \
  --client-load-threads-per-process 2 \
  --runtime-seconds 1 \
  --repeat 1 \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo <remote_repo> \
  --build-dir build_release \
  --remote-build-dir build_release \
  --remote-container <container> \
  --server-worker-threads 2 \
  --cluster-timeout 15 \
  --output-dir <output_dir>/cross_rpc_smoke \
  --show-runner-logs
```

Canonical two-host GRPC/BRPC small stress, only after smoke passes:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports grpc,brpc \
  --server-plan 0:<server_ssh>:15000:0 \
  --client-plan 0:<client_ssh>,1:<client_ssh>,2:<client_ssh>,3:<client_ssh> \
  --record-count 50000 \
  --value-size 128 \
  --batch-keys 64 \
  --client-threads-per-process 4 \
  --client-load-threads-per-process 4 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo <remote_repo> \
  --build-dir build_release \
  --remote-build-dir build_release \
  --remote-container <container> \
  --server-worker-threads 4 \
  --output-dir <output_dir>/cross_rpc_stress_small \
  --show-runner-logs
```

Fair local small matrix across RDMA/GRPC/BRPC:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma,grpc,brpc \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count 10000 \
  --value-size 128 \
  --batch-keys 64 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 1 \
  --repeat 1 \
  --execution-backend local \
  --server-worker-threads 4 \
  --server-rdma-threads 1 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --output-dir <output_dir>/fair_small
```

Local optimized RDMA GET capacity:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 4 \
  --record-count 1000000 \
  --value-size 512 \
  --batch-keys 500 \
  --index-type DRAM_PET_HASH \
  --client-threads-per-process 3 \
  --client-load-threads-per-process 3 \
  --runtime-seconds 5 \
  --repeat 1 \
  --execution-backend local \
  --build-dir build_release \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --rdma-rc-profile-interval-ms 0 \
  --rdma-rc-server-numa-id 0 \
  --rdma-rc-client-numa-id 0 \
  --rdma-server-bind-core-offset 0 \
  --rdma-client-bind-core-offset 16 \
  --rdma-client-bind-core-stride 2 \
  --output-dir <output_dir>/rdma_capacity
```

Canonical two-host RDMA PET_HASH run:

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --server-plan 0:<server_ssh>:25000:0 \
  --client-plan 0:<client_ssh>,1:<client_ssh>,2:<client_ssh>,3:<client_ssh> \
  --record-count 1000000 \
  --value-size 512 \
  --batch-keys 500 \
  --index-type DRAM_PET_HASH \
  --client-threads-per-process 3 \
  --client-load-threads-per-process 3 \
  --runtime-seconds 5 \
  --repeat 1 \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo <remote_repo> \
  --build-dir build_release \
  --remote-build-dir build_release \
  --remote-container <container> \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --rdma-rc-profile-interval-ms 0 \
  --rdma-rc-server-numa-id 0 \
  --rdma-rc-client-numa-id 0 \
  --rdma-server-bind-core-offset 0 \
  --rdma-client-bind-core-offset 16 \
  --rdma-client-bind-core-stride 2 \
  --rdma-control-plane-port <unused_port> \
  --output-dir <output_dir>/cross_rdmaps \
  --show-runner-logs
```

## Transport Comparison Rules

- Split RDMA and RPC into separate commands when their concurrency settings
  differ.
- Fair comparison means the same `client-threads-per-process` and
  `client-load-threads-per-process` across all transports.
- Mixed-concurrency capacity checks are allowed, but label them as capacity or
  stress observations instead of fair transport comparisons.
- Use `rpc-small` for GRPC/BRPC by default. Do not run large RPC comparisons
  unless the user explicitly asks.
- Do not run `rdma,grpc,brpc` in one command unless the chosen thread settings,
  workload size, and server settings are valid for every selected transport.

## RDMA Parameter Rules

- `--batch-keys` is keys per request, not total keys.
- `--rdma-rc-qps-per-client-per-shard` is RC QP pool size, not target QPS.
- For async paths, require
  `qps-per-client-per-shard * slots-per-qp >= async-depth`.
- `--prefetch-depth` overrides the default RDMA fetch pipeline depth. Use it
  with transaction fetch runs.
- RDMA GET response mode is layout-dependent. Use
  `--rdma-get-response-mode auto` by default: it maps `DRAM_PET_HASH` to
  `staging_copy` and other index types to `direct_sg`.
- For `DRAM_PET_HASH + auto/staging_copy`, keep
  `--rdma-rc-server-get-workers 0`.
- Keep `--rdma-rc-profile-interval-ms=0` for default throughput and capacity
  runs. Profile logging can materially reduce measured throughput; enable it
  only for diagnostic rows and label those rows as diagnostic/profile-enabled.
- Keep `--rdma-rc-server-coroutines-per-thread=1` unless explicitly testing
  scanner scheduling.
- Do not reintroduce retired direct-SG enable/disable flags or
  `rdma_rc_get_inner_parallelism` as default tuning dimensions.
- `--rdma-rc-fake-get-mode` and `--rdma-rc-skip-client-copy` are diagnostic
  knobs. Keep those rows separate from normal benchmark results.

## RDMA Diagnosis

Use one-knob variants and label them as diagnostics:

- server device placement: vary `--rdma-rc-server-numa-id`
- request/status path ceiling: `--rdma-rc-fake-get-mode status_only`
- server payload without PET_HASH lookup: `--rdma-rc-fake-get-mode payload_memset`
- client response-copy check: `--rdma-rc-skip-client-copy`
- outstanding/QP check:
  `--prefetch-depth 32 --rdma-rc-qps-per-client-per-shard 32`
- profile:
  `--rdma-rc-profile-interval-ms 1000` or another explicit nonzero interval.
  Treat the row as diagnostic/profile-enabled, not as the default capacity
  result.

Extract total run throughput:

```bash
awk -F, 'NR>1 && $3=="run" {sum+=$18; n++} END {printf "run_sum_mkeys=%.3f clients=%d\n", sum/1000000, n}' <output_dir>/summary.csv
```

Inspect profile tails:

```bash
rg 'component=rdma_rc_server_profile' <output_dir>/logs/rdma/repeat_0/server/server_0.log | tail -n 5
rg 'component=rdma_rc_transport_profile role=server' <output_dir>/logs/rdma/repeat_0/server/server_0.log | tail -n 5
rg 'component=rdma_rc_client_profile' <output_dir>/logs/rdma/repeat_0/client_0.stdout.log | tail -n 5
```

Compare:

- throughput: `key_ops_per_sec / 1e6`
- server: `scan_hit_pct`, `handled_get`, `handle_get_avg_ns`,
  `complete_response_avg_ns`, `poll_loop_avg_ns`
- server transport: `complete_count`, `response_payload_bytes`,
  `complete_avg_ns`
- client: `submit_avg_ns`, `wait_status_avg_ns`, `pending_rpc_peak`

Low server `scan_hit_pct` means pollers rarely observe ready slots during their
scan. It does not by itself prove clients are slow.

## Summary Format

Write `<output_dir>/summary.md` in Chinese after the benchmark finishes. The
first two lines of `summary.md` must be the current git commit hash and
hostname from the RecStore checkout / host used for the run (`git rev-parse
HEAD` on line 1, `hostname` on line 2), then a blank line, then the report
body. Include:

1. `配置说明`: mode, transports, endpoint IPs, role mapping, backend, build
   directory/build type, remote repo/container when used, output directory
2. `Workload 说明`: record count, value size, batch keys, runtime seconds,
   repeat, distribution, mode/read ratio, index type, allocator, prefetch depth
3. `并发与 RDMA 参数`: client processes, client threads, load threads, server
   worker threads, server RDMA threads, QP pool, slots per QP, response mode,
   profile interval, NUMA/core binding when set
4. `结果表`: rows from `summary.csv`, with throughput shown in M keys/s when
   applicable
5. `失败与跳过`: failed rows, skipped RDMA rows, missing verbs, timeout, or
   nonzero exit reasons

If any row exits nonzero, still write `summary.md` from available artifacts and
state failures clearly in the final response.

## Reporting Rules

- Reply to the user in Chinese.
- Do not claim tests pass unless the command completed successfully.
- Do not claim RDMA correctness or benchmark success unless the relevant command
  completed successfully.
- If RDMA verbs are missing, state that only build/unit-level validation was
  possible.
- Always include the exact command or command file path.
- Separate fair comparisons, capacity checks, and diagnostic fake-mode rows.
- Keep generated project-facing report text in Chinese.

## Current Bring-up Notes

- On the 2026-06-03 p4/t3/q16/d16 Release/O3 lane, local profile-disabled
  PET_HASH RDMA GET throughput was about `45.720 M keys/s`.
- On the same lane, cross-host profile-disabled throughput was about
  `45.523 M keys/s`.
- Do not compare Release/O3 runs against older Debug/O0 baselines without
  labeling the build type.
