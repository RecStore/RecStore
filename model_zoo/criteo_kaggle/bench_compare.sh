#!/usr/bin/env bash
set -euo pipefail

CRITEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.sh
source "${CRITEO_DIR}/cluster.sh"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/criteo_kaggle_e2e_$(date +%m%d%H%M)}"

criteo_init_ssh
cleanup() {
    criteo_kill_both >/dev/null 2>&1 || true
    criteo_cleanup_ssh
}
trap cleanup EXIT

criteo_require_processed
criteo_preflight_ib
[[ -x "${REPO_ROOT}/build/bin/petps_server" ]] || { echo "missing build/bin/petps_server" >&2; exit 1; }

cd "${REPO_ROOT}"
criteo_kill_both

python3 -m tools.benchmarks.e2e.custom.cli \
    --data-dir "${CRITEO_PROCESSED}" \
    --client "ssh=${CRITEO_REMOTE_SSH},ssh_port=${CRITEO_REMOTE_SSH_PORT},repo=${REPO_ROOT},ip=${CRITEO_REMOTE_IP},gpu=${CRITEO_REMOTE_GPU},node_rank=0,nproc=1" \
    --client "ssh=local,ssh_port=${CRITEO_REMOTE_SSH_PORT},repo=${REPO_ROOT},ip=${CRITEO_LOCAL_IP},gpu=${CRITEO_LOCAL_GPU},node_rank=1,nproc=1" \
    --ps "ssh=local,ssh_port=${CRITEO_REMOTE_SSH_PORT},repo=${REPO_ROOT},ip=${CRITEO_LOCAL_IP},port=${CRITEO_PS_PORT},shard=0" \
    --ps "ssh=${CRITEO_REMOTE_SSH},ssh_port=${CRITEO_REMOTE_SSH_PORT},repo=${REPO_ROOT},ip=${CRITEO_REMOTE_IP},port=${CRITEO_PS_PORT},shard=1" \
    --transports rdma \
    --batch-size 2048 --num-embeddings 800000 \
    --steps 80 --warmup-steps 5 --repeat 3 \
    --index-type DRAM_PET_HASH \
    --master-port "${CRITEO_MASTER_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --skip-build --skip-tests \
    "$@"

mapfile -t torchrec_nccl_logs < <(
    find "${OUTPUT_DIR}/outputs" -path '*/torchrec_nccl_rank*.log' -type f | sort
)
if [[ ${#torchrec_nccl_logs[@]} -eq 0 ]]; then
    echo "no TorchRec NCCL logs under ${OUTPUT_DIR}/outputs" >&2
    exit 1
fi
for log_path in "${torchrec_nccl_logs[@]}"; do
    grep -q 'NET/IB' "$log_path"
    grep -q 'mlx5_0:1/IB' "$log_path"
done
echo "TorchRec NCCL IB ok on ${#torchrec_nccl_logs[@]} logs"
echo "summary: ${OUTPUT_DIR}/summary.md"
