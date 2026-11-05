#!/usr/bin/env bash
# Kill leftover RecStore / TorchRec processes (PS binaries + related Python).
# Usage: RECSTORE_BENCH_PROCESS_ROOT=/path/to/run tools/benchmarks/kill_bench_procs.sh [--dry-run]
# Set the optional root on shared hosts to preserve unrelated experiments.
# Remote: ssh [-p port] host 'bash -s' < tools/benchmarks/kill_bench_procs.sh
#         ssh host "docker exec -i CONTAINER bash -s" < tools/benchmarks/kill_bench_procs.sh
#
# ponytail: loop+KILL until clear; 60s ceiling if unkillable remain.

set -euo pipefail

dry_run=0
[[ "${1:-}" == "--dry-run" ]] && dry_run=1

# PS / native bins, then Python entrypoints (training + bench runners).
# torchrun workers are covered via run_mock_stress.py / dlrm_main in their cmdline.
pattern='(^|/)(ps_server|petps_server|local_shm_ps_server|benchmark_kv_engine|ps_transport_benchmark|recstore_mixed_benchmark)( |$)|run_mock_stress\.py|dlrm_main[^ ]*\.py|tools\.benchmarks\.|tools/benchmarks/[^ ]+\.py|run_benchmark_ps\.py|run_kvengine_compare\.py|run_hierkv_recstore|run_local_shm|run_ps_dram|run_hps_|run_storage_backend|run_rdma_transport|run_main_results\.py|run_stage_breakdown\.py|run_recstore_chain\.py|run_all\.py|(^|/)(test_kvengine|test_recstore|test_[^ /]*ps_client)( |$)'

self=$$
process_root=${RECSTORE_BENCH_PROCESS_ROOT:-}
process_root=${process_root%/}

# pgrep -af can emit the same TGID once per thread comm. Kill each PID once.
collect_from_lines() {
  local line pid process_cwd
  pids=()
  local -A seen=()
  # bash 4.2 (CentOS 7) errors on expanding a never-declared array under
  # `set -u`; the `${var+x}` guard keeps this working when lines is unset.
  for line in ${lines[@]+"${lines[@]}"}; do
    pid=${line%% *}
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "$pid" -eq "$self" ]] && continue
    [[ -n ${seen[$pid]+x} ]] && continue
    [[ "$line" == *kill_bench_procs.sh* ]] && continue
    # Skip log tails / editors that only mention these names in a path arg.
    [[ "$line" == *'tail '* ]] && continue
    [[ "$line" == *'less '* ]] && continue
    [[ "$line" == *'rg '* ]] && continue
    # A build target may contain a benchmark name without running that binary.
    [[ "$line" =~ (^|[[:space:]/])(cmake|gmake|make|ninja|ctest|cc|c\+\+|gcc|g\+\+|cc1|cc1plus|clang|clang\+\+|nvcc|cicc|ptxas|collect2|ld|ar|ranlib)([[:space:]]|$) ]] && continue
    if [[ -n "$process_root" && "$line" != *"$process_root/"* ]]; then
      process_cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
      [[ "$process_cwd" == "$process_root" || "$process_cwd" == "$process_root/"* ]] || continue
    fi
    seen[$pid]=1
    echo "$line"
    pids+=("$pid")
  done
}

collect_pids() {
  # bash 4.2 (CentOS 7) errors on expanding an empty array under `set -u`;
  # `${var+x}` keeps it defined even when pgrep matches nothing.
  lines=()
  mapfile -t lines < <(pgrep -af -- "$pattern" 2>/dev/null || true)
  collect_from_lines
}

if [[ "${1:-}" == "--self-check" ]]; then
  process_root=""
  lines=(
    "111111 /app/RecStore/build/bin/petps_server --config_path x"
    "111111 petps_server"
    "111111 petps_server"
    "222222 /app/RecStore/build/bin/petps_server --config_path y"
    "333333 cmake --build build_release --target test_kvengine -j2"
    "444444 /usr/bin/gmake -f Makefile test_kvengine"
    "555555 ninja test_rdma_ps_client"
    "666666 ctest -R test_rdma_ps_client"
    "777777 /bin/sh -c /usr/bin/c++ -o build/bin/test_kvengine test_kvengine.cpp"
    "888888 /usr/bin/c++ -c src/test/test_kvengine.cpp"
    "999999 /usr/lib/gcc/cc1plus src/test/test_kvengine.cpp"
  )
  collect_from_lines >/dev/null
  if [[ ${#pids[@]} -ne 2 || ${pids[0]} -ne 111111 || ${pids[1]} -ne 222222 ]]; then
    echo "self-check failed: pids=(${pids[*]})" >&2
    exit 1
  fi
  process_root=/app/RecStore
  lines=("111111 /app/RecStore/build/bin/petps_server" "222222 /other/build/bin/petps_server")
  collect_from_lines >/dev/null
  if [[ ${#pids[@]} -ne 1 || ${pids[0]} -ne 111111 ]]; then
    echo "scoped self-check failed" >&2
    exit 1
  fi
  echo "self-check ok"
  exit 0
fi

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
