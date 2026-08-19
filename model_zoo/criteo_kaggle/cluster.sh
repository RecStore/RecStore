# Shared two-node topology for criteo_kaggle train/e2e scripts.
# Source from those scripts; do not execute directly.

CRITEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CRITEO_DIR}/../.." && pwd)"
export REPO_ROOT

export CRITEO_LOCAL_IP="${CRITEO_LOCAL_IP:-10.0.2.192}"
export CRITEO_LOCAL_GPU="${CRITEO_LOCAL_GPU:-3}"
export CRITEO_REMOTE_IP="${CRITEO_REMOTE_IP:-10.0.2.191}"
export CRITEO_REMOTE_GPU="${CRITEO_REMOTE_GPU:-0}"
export CRITEO_REMOTE_SSH="${CRITEO_REMOTE_SSH:-root@10.0.2.191}"
export CRITEO_REMOTE_SSH_PORT="${CRITEO_REMOTE_SSH_PORT:-22222}"
export CRITEO_PS_PORT="${CRITEO_PS_PORT:-15000}"
export CRITEO_MASTER_PORT="${CRITEO_MASTER_PORT:-29500}"
export CRITEO_PROCESSED="${CRITEO_PROCESSED:-${CRITEO_DIR}/processed}"
export SSHPASS="${SSHPASS:-1234}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp3s0f0,eno8303}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-NET}"
export PYTHONPATH="${REPO_ROOT}/src/python/pytorch:${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
CRITEO_CLIENT_ENV=()

_CRITEO_SSH_WRAP=""

criteo_init_ssh() {
    command -v sshpass >/dev/null || {
        echo "missing sshpass" >&2
        return 1
    }
    _CRITEO_SSH_WRAP="$(mktemp -d)"
    printf '%s\n' '#!/bin/sh' 'exec sshpass -e /usr/bin/ssh "$@"' >"${_CRITEO_SSH_WRAP}/ssh"
    chmod 700 "${_CRITEO_SSH_WRAP}/ssh"
    export PATH="${_CRITEO_SSH_WRAP}:${PATH}"
}

criteo_cleanup_ssh() {
    if [[ -n "${_CRITEO_SSH_WRAP}" ]]; then
        rm -f "${_CRITEO_SSH_WRAP}/ssh"
        rmdir "${_CRITEO_SSH_WRAP}" 2>/dev/null || true
        _CRITEO_SSH_WRAP=""
    fi
}

criteo_ssh_remote() {
    ssh -p "${CRITEO_REMOTE_SSH_PORT}" \
        -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=10 \
        "${CRITEO_REMOTE_SSH}" "$@"
}

criteo_kill_both() {
    "${REPO_ROOT}/tools/benchmarks/kill_bench_procs.sh"
    criteo_ssh_remote 'bash -s' <"${REPO_ROOT}/tools/benchmarks/kill_bench_procs.sh"
}

criteo_require_processed() {
    if [[ ! -e "${CRITEO_PROCESSED}/day_0_labels.npy" ]]; then
        echo "missing ${CRITEO_PROCESSED}/day_0_labels.npy; run: python3 ${CRITEO_DIR}/preprocess.py" >&2
        return 1
    fi
}

criteo_preflight_ib() {
    local check
    check='test -e /sys/class/infiniband/mlx5_0 && grep -q ACTIVE /sys/class/infiniband/mlx5_0/ports/1/state && grep -q InfiniBand /sys/class/infiniband/mlx5_0/ports/1/link_layer'
    bash -c "$check"
    criteo_ssh_remote "bash -lc $(printf '%q' "$check")"
}

criteo_check_nccl_ib() {
    local log=$1
    grep -q 'NET/IB' "$log" && grep -q 'mlx5_0:1/IB' "$log"
}

# CRITEO_CLIENT_ENV: optional array of KEY=VAL extra exports (RDMA, RECSTORE_CONFIG).
criteo_run_node() {
    local where=$1 node_rank=$2 gpu=$3 logfile=$4 py=$5
    shift 5
    local -a cmd=(
        env "CUDA_VISIBLE_DEVICES=${gpu}"
        "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
        "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
        "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY}"
        "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
        "NCCL_IB_HCA=${NCCL_IB_HCA}"
        "NCCL_DEBUG=${NCCL_DEBUG}"
        "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
        "PYTHONPATH=${PYTHONPATH}"
        "PYTHONUNBUFFERED=1"
    )
    if ((${#CRITEO_CLIENT_ENV[@]})); then
        cmd+=("${CRITEO_CLIENT_ENV[@]}")
    fi
    cmd+=(
        python3 -m torch.distributed.run
        --nnodes 2 --nproc_per_node 1 --node_rank "${node_rank}"
        --master_addr "${CRITEO_REMOTE_IP}" --master_port "${CRITEO_MASTER_PORT}"
        "${py}"
        "$@"
    )
    mkdir -p "$(dirname "$logfile")"
    if [[ "$where" == remote ]]; then
        local remote_cmd="cd $(printf '%q' "${REPO_ROOT}") &&"
        local part
        for part in "${cmd[@]}"; do
            remote_cmd+=" $(printf '%q' "$part")"
        done
        criteo_ssh_remote "bash -lc $(printf '%q' "$remote_cmd")" >"$logfile" 2>&1 &
    else
        (cd "${REPO_ROOT}" && "${cmd[@]}") >"$logfile" 2>&1 &
    fi
    CRITEO_LAST_PID=$!
}

criteo_wait_pair() {
    local pid0=$1 pid1=$2
    local e0=0 e1=0
    wait "$pid0" || e0=$?
    wait "$pid1" || e1=$?
    if ((e0 != 0 || e1 != 0)); then
        echo "trainer failed: rank0=${e0} rank1=${e1}" >&2
        return 1
    fi
}
