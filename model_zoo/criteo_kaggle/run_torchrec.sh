#!/usr/bin/env bash
set -euo pipefail

CRITEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.sh
source "${CRITEO_DIR}/cluster.sh"

BATCH_SIZE="${CRITEO_BATCH_SIZE:-2048}"
LR="${CRITEO_LR:-0.05}"
EPOCHS="${CRITEO_EPOCHS:-2}"
LOG_DIR="${CRITEO_LOG_DIR:-${REPO_ROOT}/results/criteo_kaggle_train/logs}"
PY="${REPO_ROOT}/model_zoo/torchrec_dlrm/tests/dlrm_main_torchrec_single.py"

criteo_init_ssh
cleanup() {
    criteo_kill_both >/dev/null 2>&1 || true
    criteo_cleanup_ssh
}
trap cleanup EXIT

criteo_require_processed
criteo_preflight_ib
criteo_kill_both

mkdir -p "$LOG_DIR"
echo "TorchRec-HBM two-node train logs: $LOG_DIR"

criteo_run_node remote 0 "${CRITEO_REMOTE_GPU}" "${LOG_DIR}/torchrec_n0.log" "$PY" \
    --single_day_mode \
    --in_memory_binary_criteo_path "${CRITEO_PROCESSED}" \
    --mmap_mode --pin_memory --adagrad --allow_tf32 \
    --embedding_storage hbm --embedding_dim 128 \
    --batch_size "${BATCH_SIZE}" --learning_rate "${LR}" --epochs "${EPOCHS}" \
    "$@"
pid0=$CRITEO_LAST_PID
criteo_run_node local 1 "${CRITEO_LOCAL_GPU}" "${LOG_DIR}/torchrec_n1.log" "$PY" \
    --single_day_mode \
    --in_memory_binary_criteo_path "${CRITEO_PROCESSED}" \
    --mmap_mode --pin_memory --adagrad --allow_tf32 \
    --embedding_storage hbm --embedding_dim 128 \
    --batch_size "${BATCH_SIZE}" --learning_rate "${LR}" --epochs "${EPOCHS}" \
    "$@"
pid1=$CRITEO_LAST_PID

criteo_wait_pair "$pid0" "$pid1"
criteo_check_nccl_ib "${LOG_DIR}/torchrec_n0.log"
criteo_check_nccl_ib "${LOG_DIR}/torchrec_n1.log"
echo "NCCL IB ok (mlx5_0 NET/IB). rank0 val AUC: grep Validation ${LOG_DIR}/torchrec_n0.log"
