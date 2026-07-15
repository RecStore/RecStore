---
name: loss-aligned
description: Run matched RecStore and TorchRec DLRM jobs, align their embedding and dense-model initialization, and compare per-rank per-step loss values. Use when validating RecStore/TorchRec numerical equivalence, diagnosing loss divergence, checking sparse update visibility or ordering, or verifying changes to rs_demo training semantics.
---

# Loss Aligned

Validate numerical behavior before comparing performance. Run the same workload through RecStore and TorchRec, enable TorchRec's RecStore-compatible initialization, and compare every recorded loss by `(rank, step)`.

## Workflow

1. Work from the RecStore repository root.
2. Confirm the alignment surface exists:

```bash
python3 model_zoo/rs_demo/run_mock_stress.py --help | rg -- '--torchrec-align-recstore-init'
rg -n 'row\["loss"\]' \
  model_zoo/rs_demo/runners/recstore_runner.py \
  model_zoo/rs_demo/runners/torchrec_runner.py
```

3. Confirm the dataset and `build/bin/ps_server` exist. Build the server and run the targeted correctness tests for the selected PS backend when needed. For remote, distributed, or non-BRPC placement, read `.agents/skills/benchmark-e2e/SKILL.md` and follow its placement, routing, preflight, and artifact rules.
4. Use one shared workload definition for both lanes. Keep these identical: dataset, batch size, embedding dimension, table cardinalities, step count, warmup count, seed, dense architecture, client placement, and rank count.
5. Keep the validation path synchronous and simple: use `--read-mode direct`, disable GPU cache and lookahead prefetch, and use TorchRec HBM unless the user explicitly asks to validate another path.
6. Run RecStore first, then TorchRec with `--torchrec-align-recstore-init`. For a local smoke validation, use:

```bash
OUT="results/loss_aligned_$(date +%m%d%H%M)"
RUN_ID="loss-aligned"
COMMON=(
  --output-root "$OUT"
  --run-id "$RUN_ID"
  --data-dir model_zoo/torchrec_dlrm/processed_day_0_data
  --batch-size 128
  --embedding-dim 128
  --num-embeddings 200000
  --steps 5
  --warmup-steps 0
  --seed 20260330
  --read-mode direct
)

python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --ps-type BRPC \
  --ps-kv-backend recstore_dram \
  --recstore-index-type DRAM_PET_HASH \
  "${COMMON[@]}"

python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --torchrec-memory-mode hbm \
  --torchrec-align-recstore-init \
  "${COMMON[@]}"
```

7. Compare all steps, including warmup-marked rows:

```bash
python3 .agents/skills/loss-aligned/scripts/compare_loss.py \
  "$OUT/outputs/$RUN_ID/recstore_main.csv" \
  "$OUT/outputs/$RUN_ID/torchrec_main.csv"
```

Use the default `atol=1e-6` and `rtol=1e-5`. Override them only when the user specifies a tolerance or the numeric mode has a documented precision limit. Never declare alignment from aggregate means alone.

## Pass Criteria

Declare the run aligned only when the comparator exits zero. Require both CSVs to contain finite `loss` values for the same `(rank, step)` keys with no duplicates, and require every pair to satisfy the configured tolerance.

On failure, preserve both CSVs and logs. Diagnose from the first divergent step:

- Step 0 divergence: check dataset order, seed, zero embedding initialization, dense-module initialization, dtype, shape, and device.
- Step 0 aligned but later divergence: check sparse optimizer semantics, update routing, flush completion, read-after-write visibility, and prefetch ordering.
- Rank-only or intermittent divergence: check sampler/rank mapping, distributed initialization, collective synchronization, and worker code fingerprints.

Do not hide missing rows, non-finite values, worker failures, or parent-process failures by loosening tolerances.

## Report

Write `<output_dir>/loss_alignment.md` in Chinese with the matched workload, artifact paths, compared row count, tolerances, maximum absolute and relative differences, first mismatch if any, and a clear `aligned` or `not aligned` conclusion. Do not claim success unless both jobs and the comparator completed successfully.
