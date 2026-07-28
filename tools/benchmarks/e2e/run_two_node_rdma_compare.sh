#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

local_ip="10.0.2.192"
local_gpu_id="3"

remote_host="10.0.2.191"
remote_ssh_port="22222"
remote_gpu_id="0"

ps_port="15000"
master_port="29500"
output_dir="${OUTPUT_DIR:-results/e2e_$(date +%m%d%H%M)}"
jobs="${JOBS:-$(nproc)}"

for command in cmake ctest python3 ssh sshpass nvidia-smi sha256sum; do
    command -v "$command" >/dev/null || {
        echo "Missing required command: $command" >&2
        exit 1
    }
done

if [[ -e "$output_dir" ]]; then
    echo "Output path already exists: $output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir/logs"

SSHPASS=1234

if [[ -z "${SSHPASS:-}" ]]; then
    read -r -s -p "Password for root@${remote_host}: " SSHPASS
    echo
fi
export SSHPASS

# The project runner invokes ssh directly. Keep the password out of commands and logs.
ssh_wrapper_dir="$(mktemp -d)"
cleanup() {
    rm -f "$ssh_wrapper_dir/ssh"
    rmdir "$ssh_wrapper_dir"
}
trap cleanup EXIT
printf '%s\n' '#!/bin/sh' 'exec sshpass -e /usr/bin/ssh "$@"' >"$ssh_wrapper_dir/ssh"
chmod 700 "$ssh_wrapper_dir/ssh"
export PATH="$ssh_wrapper_dir:$PATH"

ssh_remote=(
    ssh -p "$remote_ssh_port"
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=10
    "root@${remote_host}"
)

sep='========================='
log_section() {
    # ponytail: banner + path only; full body already streamed via tee
    echo "$sep"
    echo "$1"
    echo "log: $2"
    echo "$sep"
}
# Dump per-system/per-repeat client (and RDMA server) logs with separators.
dump_run_logs() {
    local title=$1
    shift
    echo "$sep"
    echo "$title"
    for f in "$@"; do
        [[ -f "$f" ]] || continue
        echo "---- $f ----"
        cat "$f"
    done
    echo "$sep"
}

log_section "[1/4] Preflight" "$output_dir/logs/preflight.log"
{
    date -Is
    git rev-parse HEAD
    hostname
    nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
    test -d model_zoo/torchrec_dlrm/processed_day_0_data
    test -e /sys/class/infiniband/mlx5_0
    grep -q ACTIVE /sys/class/infiniband/mlx5_0/ports/1/state
    grep -q InfiniBand /sys/class/infiniband/mlx5_0/ports/1/link_layer
    "${ssh_remote[@]}" \
        'cd /app/RecStore && hostname && git rev-parse HEAD && nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader && test -d model_zoo/torchrec_dlrm/processed_day_0_data && test -e /sys/class/infiniband/mlx5_0 && grep -q ACTIVE /sys/class/infiniband/mlx5_0/ports/1/state && grep -q InfiniBand /sys/class/infiniband/mlx5_0/ports/1/link_layer'
} 2>&1 | tee "$output_dir/logs/preflight.log"

local_fingerprint="$(sha256sum \
    model_zoo/rs_demo/run_mock_stress.py \
    tools/benchmarks/e2e/custom/cli.py \
    tools/benchmarks/e2e/custom/runner.py \
    tools/benchmarks/e2e/custom/runtime.py \
    tools/benchmarks/e2e/custom/report.py)"
remote_fingerprint="$("${ssh_remote[@]}" \
    'cd /app/RecStore && sha256sum model_zoo/rs_demo/run_mock_stress.py tools/benchmarks/e2e/custom/cli.py tools/benchmarks/e2e/custom/runner.py tools/benchmarks/e2e/custom/runtime.py tools/benchmarks/e2e/custom/report.py')"
if [[ "$local_fingerprint" != "$remote_fingerprint" ]]; then
    echo "Local and remote runner fingerprints differ" >&2
    exit 1
fi

log_section "[2/4] cmake configure" "$output_dir/logs/cmake_configure.log"
# The current custom runner resolves binaries from build/, so configure that tree as Release.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tee "$output_dir/logs/cmake_configure.log"

log_section "[2/4] Release build" "$output_dir/logs/build.log"
cmake --build build --target \
    ps_server recstore_torch_ops petps_server \
    test_rdma_rc_protocol test_raw_verbs_allocator \
    test_rdmaps_client_adapter test_allshards_ps_client \
    -j "$jobs" 2>&1 | tee "$output_dir/logs/build.log"
grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' build/CMakeCache.txt

log_section "[2/4] RDMA correctness tests" "$output_dir/logs/ctest_rdma.log"
ctest --test-dir build \
    -R 'test_rdma_rc_protocol|test_raw_verbs_allocator|test_rdmaps_client_adapter|test_allshards_ps_client' \
    --output-on-failure 2>&1 | tee "$output_dir/logs/ctest_rdma.log"

log_section "[3/4] RecStore-RDMA and TorchRec-HBM benchmark" "$output_dir/logs/runner.log"
benchmark_args=(
    --client "ssh=root@${remote_host},ssh_port=${remote_ssh_port},repo=/app/RecStore,ip=${remote_host},gpu=${remote_gpu_id},node_rank=0,nproc=1"
    --client "ssh=local,ssh_port=${remote_ssh_port},repo=/app/RecStore,ip=${local_ip},gpu=${local_gpu_id},node_rank=1,nproc=1"
    # Keep shard 0 local because the current RDMA runner allocates its control-plane port locally.
    --ps "ssh=local,ssh_port=${remote_ssh_port},repo=/app/RecStore,ip=${local_ip},port=${ps_port},shard=0"
    --ps "ssh=root@${remote_host},ssh_port=${remote_ssh_port},repo=/app/RecStore,ip=${remote_host},port=${ps_port},shard=1"
    --output-dir "$output_dir"
    --transports rdma
    --batch-size 1024
    --embedding-dim 128
    --num-embeddings 200000
    --steps 80
    --warmup-steps 5
    --repeat 3
    --read-mode prefetch
    --prefetch-depth 0
    --index-type DRAM_PET_HASH
    --master-port "$master_port"
    --skip-build
    --skip-tests
)
python3 -m tools.benchmarks.e2e.custom.cli "${benchmark_args[@]}" \
    2>&1 | tee "$output_dir/logs/runner.log"

bs=1024
dim=128
repeats=3
for ((r = 0; r < repeats; r++)); do
    dump_run_logs "RecStore-RDMA repeat_${r}" \
        "$output_dir/logs/rdma_server_shard0_r${r}.log" \
        "$output_dir/logs/rdma_b${bs}_d${dim}_r${r}_n0.log" \
        "$output_dir/logs/rdma_b${bs}_d${dim}_r${r}_n1.log"
done
for ((r = 0; r < repeats; r++)); do
    dump_run_logs "TorchRec-HBM repeat_${r}" \
        "$output_dir/logs/torchrec_hbm_b${bs}_d${dim}_r${r}_n0.log" \
        "$output_dir/logs/torchrec_hbm_b${bs}_d${dim}_r${r}_n1.log"
done

log_section "[4/4] Artifact and RDMA verification" "$output_dir/rdma_verification.txt"
python3 - "$output_dir" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8")))
if len(manifest) != 12 or any(row["status"] != "ok" for row in manifest):
    raise SystemExit("manifest must contain 12 successful client runs")

for prefix, rank_dir in (("rdma_b1024_d128", "recstore_ranks"),
                         ("torchrec_hbm_b1024_d128", "torchrec_ranks")):
    for repeat in range(3):
        for rank in range(2):
            path = root / "outputs" / f"{prefix}_r{repeat}" / rank_dir / f"rank{rank}.csv"
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            if len(rows) != 80 or rows[-1]["step"] != "79":
                raise SystemExit(f"incomplete rank CSV: {path}")
            warmup = sum(row.get("warmup_excluded") in {"1", "true", "True"} for row in rows)
            if warmup != 5:
                raise SystemExit(f"unexpected warmup rows in {path}: {warmup}")
PY

mapfile -t torchrec_nccl_logs < <(
    find "$output_dir/outputs" -path '*/torchrec_nccl_rank*.log' -type f | sort
)
if [[ "${#torchrec_nccl_logs[@]}" -ne 6 ]]; then
    echo "Expected 6 TorchRec NCCL logs, found ${#torchrec_nccl_logs[@]}" >&2
    exit 1
fi
for log_path in "${torchrec_nccl_logs[@]}"; do
    grep -q 'NET/IB' "$log_path"
    grep -q 'mlx5_0:1/IB' "$log_path"
done

for shard in 0 1; do
    grep -E "component=rdma_rc_server_profile shard=${shard} .*handled_get=[1-9]" \
        "$output_dir/logs/runner.log" >/dev/null
    grep -E "component=rdma_rc_transport_profile role=server shard=${shard} .*response_payload_bytes=[1-9]" \
        "$output_dir/logs/runner.log" >/dev/null
done

{
    echo "RecStore-RDMA：shard 0 和 shard 1 均确认存在非零 GET 流量与响应字节。"
    echo "TorchRec-HBM：全部 6 份 rank/repeat NCCL 日志均确认通过 mlx5_0 使用 NET/IB。"
    echo "Job throughput：每个 step 使用最慢 rank 的延迟反推任务吞吐。"
} >"$output_dir/rdma_verification.txt"
sed -i '/本次测试模型为 /a\通信验证：RecStore 双 shard 已观察到非零 RDMA GET 与响应字节；TorchRec 全部 rank/repeat 均观察到 mlx5_0 NET/IB。' \
    "$output_dir/summary.md"
echo "$sep"
echo "[4/4] done"
echo "summary.md: $output_dir/summary.md"
echo "$sep"

echo "$sep"
echo "Benchmark complete: $output_dir"
echo "Summary: $output_dir/summary.md"
echo "RDMA evidence: $output_dir/rdma_verification.txt"
echo "Logs: $output_dir/logs/"
echo "$sep"
