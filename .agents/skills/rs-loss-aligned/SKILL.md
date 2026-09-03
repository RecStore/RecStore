---
name: rs-loss-aligned
description: Use when validating RecStore/TorchRec numerical equivalence, diagnosing loss divergence, checking sparse update visibility or ordering, or verifying changes to rs_demo training semantics.
---

# Loss Aligned

Validate numerical behavior before comparing performance. Run the same workload through RecStore and TorchRec, enable TorchRec's RecStore-compatible initialization, and compare every recorded loss by `(rank, step)`.

Default validation is **both** topologies: single-node and two-node. Run both unless the user names only one. Do not declare default validation aligned from a single topology.

## Workflow

1. Work from the RecStore repository root.
2. Confirm the alignment surface exists:

```bash
python3 model_zoo/rs_demo/run_mock_stress.py --help | rg -- '--torchrec-align-recstore-init'
rg -n 'row\["loss"\]' \
  model_zoo/rs_demo/runners/recstore_runner.py \
  model_zoo/rs_demo/runners/torchrec_runner.py
```

3. Confirm the dataset and `build/bin/ps_server` exist. Build the server and run the targeted correctness tests for the selected PS backend when needed. Two-node uses RDMA (`build/bin/petps_server`). Follow `.agents/skills/rs-benchmark-e2e/SKILL.md` for placement, routing, preflight, process cleanup, and artifacts. Do not use `tools.benchmarks.e2e.custom.cli` as-is: it omits `--seed` and `--torchrec-align-recstore-init`.
4. Use one shared workload definition for both lanes **and both topologies**. Keep these identical: dataset, batch size, embedding dimension, table cardinalities, step count, warmup count, seed, dense architecture.
5. Keep the validation path synchronous and simple: use `--read-mode direct`, disable GPU cache and lookahead prefetch, and use TorchRec HBM unless the user explicitly asks to validate another path.
6. Run RecStore first, then TorchRec with `--torchrec-align-recstore-init`. Default output layout:

```bash
OUT="results/loss_aligned_$(date +%m%d%H%M)"
COMMON=(
  --data-dir model_zoo/torchrec_dlrm/processed_day_0_data
  --batch-size 128
  --embedding-dim 128
  --num-embeddings 200000
  --steps 5
  --warmup-steps 0
  --seed 20260330
  --read-mode direct
)
```

Single-node (`$OUT/single`, run-id `single`):

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --ps-type BRPC \
  --ps-kv-backend recstore_dram \
  --recstore-index-type DRAM_PET_HASH \
  --output-root "$OUT/single" \
  --run-id single \
  "${COMMON[@]}"

python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --torchrec-memory-mode hbm \
  --torchrec-align-recstore-init \
  --output-root "$OUT/single" \
  --run-id single \
  "${COMMON[@]}"
```

Two-node (`$OUT/two_node`, run-id `two-node`). Run from `10.0.2.192`. Default placement: client `10.0.2.191/GPU0/rank0` + `10.0.2.192/GPU1/rank1`; PS shard0 `10.0.2.192:15000`, shard1 `10.0.2.191:15000`; RecStore RDMA; TorchRec HBM. The script injects `--seed` and `--torchrec-align-recstore-init`:

```bash
python3 .agents/skills/rs-loss-aligned/scripts/run_two_node.py \
  --output-dir "$OUT/two_node" \
  --run-id two-node \
  "${COMMON[@]}"
```

7. Compare all steps, including warmup-marked rows, **once per topology**:

```bash
python3 .agents/skills/rs-loss-aligned/scripts/compare_loss.py \
  "$OUT/single/outputs/single/recstore_main.csv" \
  "$OUT/single/outputs/single/torchrec_main.csv"

python3 .agents/skills/rs-loss-aligned/scripts/compare_loss.py \
  "$OUT/two_node/outputs/two-node/recstore_main.csv" \
  "$OUT/two_node/outputs/two-node/torchrec_main.csv"
```

Use the default `atol=1e-6` and `rtol=1e-5`. Override them only when the user specifies a tolerance or the numeric mode has a documented precision limit. Never declare alignment from aggregate means alone.

Do not skip two-node because single-node already aligned. Do not skip single-node because two-node already aligned. Do not treat an E2E throughput run as a substitute.

## Pass Criteria

Declare **one topology** aligned only when that topology's comparator exits zero. Require both CSVs to contain finite `loss` values for the same `(rank, step)` keys with no duplicates, and require every pair to satisfy the configured tolerance.

Declare **default validation** aligned only when **both** topologies are aligned. If either job, rank, or comparator fails, the default validation is `not aligned`.

On failure, preserve both CSVs and logs. Diagnose from the first divergent step:

- Step 0 divergence: check dataset order, seed, zero embedding initialization, dense-module initialization, dtype, shape, and device.
- Step 0 aligned but later divergence: check sparse optimizer semantics, update routing, flush completion, read-after-write visibility, and prefetch ordering.
- Rank-only or intermittent divergence: check sampler/rank mapping, distributed initialization, collective synchronization, and worker code fingerprints.

Do not hide missing rows, non-finite values, worker failures, or parent-process failures by loosening tolerances.

## Report

Write `<output_dir>/loss_alignment.md` in Chinese covering every topology that ran. For each topology record the matched workload, artifact paths, compared row count, tolerances, maximum absolute and relative differences, first mismatch if any, and `aligned` or `not aligned`. End with a single overall conclusion: default validation is `aligned` only when both topologies passed. Do not claim success unless every requested job and comparator completed successfully.
