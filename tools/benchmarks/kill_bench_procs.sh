#!/usr/bin/env bash
# Kill leftover RecStore benchmark processes (e2e / kvengine / ps).
# Usage: tools/benchmarks/kill_bench_procs.sh [--dry-run]
#
# ponytail: local only; pass SSH hosts later if cross-host cleanup is needed.

set -euo pipefail

dry_run=0
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

# Binary / runner names from benchmark-e2e, benchmark-kvengine, benchmark-ps.
pattern='(^|/)(ps_server|petps_server|benchmark_kv_engine|ps_transport_benchmark)( |$)|run_mock_stress\.py|run_benchmark_ps\.py|run_kvengine_compare\.py|tools\.benchmarks\.e2e\.custom|test_kvengine'

self=$$
mapfile -t lines < <(pgrep -af -- "$pattern" 2>/dev/null || true)

signaled=0
declare -a pids=()
for line in "${lines[@]}"; do
  pid=${line%% *}
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  [[ "$pid" -eq "$self" ]] && continue
  [[ "$line" == *kill_bench_procs.sh* ]] && continue
  echo "$line"
  pids+=("$pid")
done

if ((${#pids[@]} == 0)); then
  echo "no matching benchmark processes"
  exit 0
fi

if ((dry_run)); then
  echo "dry-run: would signal ${#pids[@]} process(es)"
  exit 0
fi

for pid in "${pids[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
  signaled=$((signaled + 1))
done

sleep 1
for pid in "${pids[@]}"; do
  [[ -d "/proc/$pid" ]] || continue
  kill -KILL "$pid" 2>/dev/null || true
done

echo "done: signaled $signaled process(es)"
