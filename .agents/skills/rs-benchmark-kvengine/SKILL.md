---
name: rs-benchmark-kvengine
description: Run RecStore KVEngine correctness, YCSB, and storage-only batch lookup benchmark workflows, including read_mode=batch_get_flat for aligning KVEngine limits with PS RDMA batch GET. Use when Codex needs to validate src/test/test_kvengine.cpp, run tools/benchmarks/run_kvengine_compare.py, prompt for thread count, SSD benchmark path, results directory, read mode, and batch keys, then generate a Chinese summary.md.
---

# Benchmark KVEngine

## Workflow

Use this skill from a RecStore checkout. Do not run helper scripts from this skill directory; call the project script directly.

1. Confirm the current directory is the RecStore repo root, or pass `--repo`.
2. Prompt the user for:
   - thread count (default = 16)
   - SSD root path for benchmark data (default = /mnt/nvme1n1_recstore/recstore)
   - result output directory (default = results/benchmark_kvengine_$(date +%m%d%H%M))
   - workloads to run (default = a b c)
   - repeat count for YCSB (default = 3, ask if user wants 1)
   - distributions to use (default = uniform and zipfian)
   - record-count (default=10M)
   - read mode (default = `get`; use `batch_get_flat` when aligning with PS RDMA GET)
   - batch keys (default = `500` for `batch_get_flat`)
3. Run build and correctness checks:
   - Always compile in Release mode for KVEngine benchmarks. The compare
     script expects `build/bin/benchmark_kv_engine`, so configure Release into
     `build`:
     `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
   - `cmake --build build --target test_kvengine -j`
   - `ctest --test-dir build -R '^test_kvengine$' -VV`
   - `cmake --build build --target benchmark_kv_engine -j`
4. Immediately before `run_kvengine_compare.py`, clean leftover
   RecStore/TorchRec/KVEngine processes on every host that will run the
   benchmark (local-only by default). Use
   `./tools/benchmarks/kill_bench_procs.sh` only — do not hand-roll `pkill -f`.
   Fail the run if cleanup exits nonzero.
5. Run `tools/benchmarks/run_kvengine_compare.py`.
6. Save logs and CSV/SVG artifacts under the chosen result directory.
7. Write `summary.md` as exactly three report tables, with the benchmark hyperparameters recorded as Chinese prose under `Workload 说明` before the first table:
   - Workload description
   - Run throughput
   - Load throughput

## Command Template

Ask the user for `threads`, `ssd_root`, and `output_dir`. Use defaults only when the user accepts them.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_kvengine -j
ctest --test-dir build -R '^test_kvengine$' -VV
cmake --build build --target benchmark_kv_engine -j
./tools/benchmarks/kill_bench_procs.sh
```

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --output-dir <output_dir> \
  --workloads a b c \
  --distributions uniform \
  --record-count 10000000 \
  --runtime-seconds 3 \
  --threads <threads> \
  --load-threads <threads> \
  --repeat 1 \
  --value-size 128 \
  --read-mode get
```

`tools/benchmarks/run_kvengine_compare.py` currently uses `/mnt/nvme1n1_recstore/recstore` internally for SSD data. If the user provides a different SSD path, create a temporary symlink or patch the command wrapper only after making that choice explicit.

If the user asks for "3 次平均", pass `--repeat 3`; otherwise preserve the requested repeat count in the `summary.md` heading.

For storage-only PS RDMA alignment, use random batch lookup rather than single-key reads. This is the path used to establish that `DRAM_EXTENDIBLE_HASH` is around `19.45M keys/s` for `BatchGetFlat(500 random keys)`, while `DRAM_PET_HASH` is around `51.96M keys/s`.

```bash
python3 tools/benchmarks/run_kvengine_compare.py \
  --output-dir <output_dir> \
  --engines dram_eh_dram dram_pet_dram \
  --workloads workloadc \
  --distributions uniform \
  --record-count 300000 \
  --runtime-seconds 3 \
  --threads 16 \
  --load-threads 16 \
  --repeat 1 \
  --value-size 512 \
  --read-mode batch_get_flat \
  --batch-keys 500
```

## Summary Format

Generate `<output_dir>/summary.md` from `kvengine_workload_summary.csv` after YCSB finishes. The first two lines of `summary.md` must be the current git commit hash and hostname from the RecStore checkout / host used for the run (`git rev-parse HEAD` on line 1, `hostname` on line 2), then a blank line, then the report body. Keep only these three sections after that header:

1. `Workload 说明`
2. `Run 吞吐（ops/s，...）`
3. `Load 吞吐（ops/s，...）`

Under `Workload 说明`, before the workload table, record the benchmark hyperparameters in Chinese prose. Include at least: `threads`, `load_threads`, `record_count`, `runtime_seconds`, `repeat`, `value_size`, `read_mode`, `batch_keys` when applicable, `distributions`, `workloads`, SSD root path, output directory, `ssd_io_backend`, `ssd_queue_depth`, and allocator choices that affect the benchmark.

Use `M` for values >= 1,000,000 and `K` for values >= 1,000. Include the `三 workload 平均` column only in the Run table.

## Reporting Rules

- Do not claim tests pass unless the script completed successfully.
- If `test_kvengine` fails, stop before YCSB and report the log path.
- If any YCSB row exits nonzero, still write `summary.md`, but state failures in the final response and point to `summary.csv`.
- Keep generated project-facing report text in Chinese.

## Current Bring-up Notes

- `run_kvengine_compare.py` renders `kvengine_ycsb_run_throughput.svg` unconditionally at the end of a normal run. If `matplotlib` is missing, the command can exit nonzero after `summary.csv` and `kvengine_workload_summary.csv` are already written.
- In that case, still generate `summary.md` from `kvengine_workload_summary.csv` and report the missing chart dependency separately.
- For `DRAM_VALUE_STORE` lanes, watch for allocator failures such as `ConcurrentSlabMemoryPool OOM`. If that happens, record the failing engine row and consider rerunning with explicit `--dram-capacity-bytes` or a different allocator.
- Judge `petkv` success by exit code plus `YCSB_LOAD_RESULT` / `YCSB_RESULT`, not by incidental `PetHash invalid. capacity_ == 0` log lines alone.
- Use `read_mode=batch_get_flat --batch-keys 500` when the goal is to compare storage-only KVEngine limits with PS RDMA GET `batch_keys=500`. Do not compare that number directly with ordinary single-key YCSB `get` throughput without labeling the operation mismatch.
