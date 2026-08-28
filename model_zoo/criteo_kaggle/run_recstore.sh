#!/usr/bin/env bash
set -euo pipefail

CRITEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.sh
source "${CRITEO_DIR}/cluster.sh"

BATCH_SIZE="${CRITEO_BATCH_SIZE:-2048}"
LR="${CRITEO_LR:-0.05}"
EPOCHS="${CRITEO_EPOCHS:-2}"
LOG_DIR="${CRITEO_LOG_DIR:-${REPO_ROOT}/results/criteo_kaggle_train/logs}"
PY="${REPO_ROOT}/model_zoo/torchrec_dlrm/tests/dlrm_main_single_day.py"
CONFIG="${CRITEO_DIR}/recstore_config.json"
PETPS="${REPO_ROOT}/build/bin/petps_server"

criteo_init_ssh
rdma_pid=""
cleanup() {
    if [[ -n "${rdma_pid}" ]]; then
        kill "${rdma_pid}" >/dev/null 2>&1 || true
        wait "${rdma_pid}" >/dev/null 2>&1 || true
    fi
    criteo_kill_both >/dev/null 2>&1 || true
    criteo_cleanup_ssh
}
trap cleanup EXIT

[[ -x "$PETPS" ]] || { echo "missing $PETPS" >&2; exit 1; }
criteo_require_processed
criteo_preflight_ib
criteo_kill_both

mkdir -p "$LOG_DIR"
export CRITEO_OUTPUT_DIR="${REPO_ROOT}/results/criteo_kaggle_train"
export CRITEO_BATCH_SIZE="${BATCH_SIZE}"
env_file="${LOG_DIR}/rdma.env"
rm -f "$env_file"
python3 "${CRITEO_DIR}/start_rdma_ps.py" \
    --config "$CONFIG" \
    --env-out "$env_file" \
    --log "${LOG_DIR}/petps.log" &
rdma_pid=$!

for _ in $(seq 1 180); do
    [[ -f "$env_file" ]] && break
    kill -0 "$rdma_pid" 2>/dev/null || { echo "petps cluster exited early; see ${LOG_DIR}/petps.log" >&2; exit 1; }
    sleep 1
done
[[ -f "$env_file" ]] || { echo "timeout waiting for RDMA env" >&2; exit 1; }
# shellcheck disable=SC1090
source "$env_file"

CRITEO_CLIENT_ENV=(
    "RECSTORE_CONFIG=${CONFIG}"
    "RECSTORE_RDMA_RC_NAMESPACE=${RECSTORE_RDMA_RC_NAMESPACE}"
    "RECSTORE_RDMA_CONTROL_PLANE_HOST=${RECSTORE_RDMA_CONTROL_PLANE_HOST}"
    "RECSTORE_RDMA_CONTROL_PLANE_PORT=${RECSTORE_RDMA_CONTROL_PLANE_PORT}"
    "RECSTORE_RDMA_GET_RESPONSE_MODE=${RECSTORE_RDMA_GET_RESPONSE_MODE}"
)
[[ -n "${RECSTORE_RDMA_CONTROL_PLANE_TIMEOUT_MS:-}" ]] && CRITEO_CLIENT_ENV+=("RECSTORE_RDMA_CONTROL_PLANE_TIMEOUT_MS=${RECSTORE_RDMA_CONTROL_PLANE_TIMEOUT_MS}")
[[ -n "${RECSTORE_RDMA_WAIT_TIMEOUT_MS:-}" ]] && CRITEO_CLIENT_ENV+=("RECSTORE_RDMA_WAIT_TIMEOUT_MS=${RECSTORE_RDMA_WAIT_TIMEOUT_MS}")
[[ -n "${RECSTORE_RDMA_RC_QPS_PER_CLIENT_PER_SHARD:-}" ]] && CRITEO_CLIENT_ENV+=("RECSTORE_RDMA_RC_QPS_PER_CLIENT_PER_SHARD=${RECSTORE_RDMA_RC_QPS_PER_CLIENT_PER_SHARD}")
[[ -n "${RECSTORE_RDMA_RC_SLOTS_PER_QP:-}" ]] && CRITEO_CLIENT_ENV+=("RECSTORE_RDMA_RC_SLOTS_PER_QP=${RECSTORE_RDMA_RC_SLOTS_PER_QP}")
[[ -n "${RECSTORE_RDMA_RC_SERVER_COROUTINES_PER_THREAD:-}" ]] && CRITEO_CLIENT_ENV+=("RECSTORE_RDMA_RC_SERVER_COROUTINES_PER_THREAD=${RECSTORE_RDMA_RC_SERVER_COROUTINES_PER_THREAD}")
[[ -n "${RECSTORE_RDMA_RC_SERVER_GET_WORKERS:-}" ]] && CRITEO_CLIENT_ENV+=("RECSTORE_RDMA_RC_SERVER_GET_WORKERS=${RECSTORE_RDMA_RC_SERVER_GET_WORKERS}")

echo "RecStore-RDMA two-node train logs: $LOG_DIR"
criteo_run_node remote 0 "${CRITEO_REMOTE_GPU}" "${LOG_DIR}/recstore_n0.log" "$PY" \
    --single_day_mode \
    --in_memory_binary_criteo_path "${CRITEO_PROCESSED}" \
    --mmap_mode --pin_memory --adagrad --allow_tf32 \
    --embedding_dim 128 \
    --batch_size "${BATCH_SIZE}" --learning_rate "${LR}" --epochs "${EPOCHS}" \
    "$@"
pid0=$CRITEO_LAST_PID
criteo_run_node local 1 "${CRITEO_LOCAL_GPU}" "${LOG_DIR}/recstore_n1.log" "$PY" \
    --single_day_mode \
    --in_memory_binary_criteo_path "${CRITEO_PROCESSED}" \
    --mmap_mode --pin_memory --adagrad --allow_tf32 \
    --embedding_dim 128 \
    --batch_size "${BATCH_SIZE}" --learning_rate "${LR}" --epochs "${EPOCHS}" \
    "$@"
pid1=$CRITEO_LAST_PID

criteo_wait_pair "$pid0" "$pid1"
criteo_check_nccl_ib "${LOG_DIR}/recstore_n0.log"
criteo_check_nccl_ib "${LOG_DIR}/recstore_n1.log"
echo "NCCL IB ok (mlx5_0 NET/IB). rank0 val AUC: grep Validation ${LOG_DIR}/recstore_n0.log"
