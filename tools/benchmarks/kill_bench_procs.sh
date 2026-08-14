#!/usr/bin/env bash
# Kill leftover RecStore / TorchRec processes (PS binaries + related Python).
# Usage: tools/benchmarks/kill_bench_procs.sh [--dry-run]
# Remote: ssh [-p port] host 'bash -s' < tools/benchmarks/kill_bench_procs.sh
#         ssh host "docker exec -i CONTAINER bash -s" < tools/benchmarks/kill_bench_procs.sh
#
# ponytail: loop+KILL until clear; 60s ceiling if unkillable remain.

set -euo pipefail

dry_run=0
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

# PS / native bins, then Python entrypoints (training + bench runners).
# torchrun workers are covered via run_mock_stress.py / dlrm_main in their cmdline.
pattern='(^|/)(ps_server|petps_server|local_shm_ps_server|benchmark_kv_engine|ps_transport_benchmark|recstore_mixed_benchmark)( |$)|run_mock_stress\.py|dlrm_main[^ ]*\.py|tools\.benchmarks\.|tools/benchmarks/[^ ]+\.py|run_benchmark_ps\.py|run_kvengine_compare\.py|run_hierkv_recstore|run_local_shm|run_ps_dram|run_hps_|run_storage_backend|run_rdma_transport|run_main_results\.py|run_stage_breakdown\.py|run_recstore_chain\.py|run_all\.py|test_kvengine|test_recstore|test_.*ps_client'

self=$$

collect_pids() {
  local line pid
  pids=()
  mapfile -t lines < <(pgrep -af -- "$pattern" 2>/dev/null || true)
  for line in "${lines[@]}"; do
    pid=${line%% *}
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$pid" -eq "$self" ]] && continue
    [[ "$line" == *kill_bench_procs.sh* ]] && continue
    # Skip log tails / editors that only mention these names in a path arg.
    [[ "$line" == *'tail '* ]] && continue
    [[ "$line" == *'less '* ]] && continue
    [[ "$line" == *'rg '* ]] && continue
    echo "$line"
    pids+=("$pid")
  done
}

pids=()
collect_pids

if ((${#pids[@]} == 0)); then
  echo "no matching RecStore/TorchRec processes"
  exit 0
fi

if ((dry_run)); then
  echo "dry-run: would signal ${#pids[@]} process(es)"
  exit 0
fi

deadline=$((SECONDS + 60))
signaled=0
rounds=0
while ((${#pids[@]} > 0)); do
  rounds=$((rounds + 1))
  for pid in "${pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
    signaled=$((signaled + 1))
  done
  if ((SECONDS >= deadline)); then
    echo "timeout: ${#pids[@]} process(es) still alive after ${rounds} round(s)" >&2
    collect_pids
    exit 1
  fi
  sleep 0.2
  collect_pids
done

echo "done: signaled $signaled kill(s) across $rounds round(s)"
