#!/bin/bash
# 3-machine RDMA benchmark: RecStore (PS + bagpipe) vs QuantaRec (local emb + sparse grad sync)
# Usage: bash run_3node_bench.sh [quanta|recstore|all]
set -euo pipefail

REPO=~/VSCodeProjects/recstore_shanhongqi
MACHINES=("10.164.95.159:0" "10.166.158.20:1" "10.164.94.5:2")
PS_IP="10.164.95.159"
MASTER_PORT=29500
RDZV_ID="bench3_$(date +%s)"

# Benchmark config (same for both architectures for fair comparison)
STEPS=40
WARMUP=5
BATCH=1024
EMB_DIM=64
NUM_EMB=200000
TOKENS_SPLIT=2400
RANKMIXER_BLOCKS=2
GATE_NUM=6
GPU_CACHE_CAP=500000
LOOKAHEAD=4

ENV_EXPORT="export LD_LIBRARY_PATH=$REPO/third_party/deps/usr/local/lib:$REPO/build/lib:\$LD_LIBRARY_PATH"
ENV_EXPORT+=" && export NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_GID_INDEX=7 NCCL_DEBUG=WARN NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=eth0"
ENV_EXPORT+=" && export PYTHONPATH=$REPO/build/lib:$REPO/src/python:$REPO/src"
ENV_EXPORT+=" && cd $REPO"

run_quanta() {
  local run_id="quanta3_$(date +%s)"
  local rdzv="${run_id}_$(date +%s)"
  echo "═══ QUANTA (local emb + sparse grad sync) run_id=$run_id ═══"
  
  local COMMON="$ENV_EXPORT && CUDA_VISIBLE_DEVICES=0 python3 model_zoo/rs_demo/run_mock_stress.py"
  COMMON+=" --backend quanta --model rankmixer"
  COMMON+=" --nnodes 3 --nproc-per-node 1 --no-start-server"
  COMMON+=" --steps $STEPS --warmup-steps $WARMUP --batch-size $BATCH"
  COMMON+=" --num-embeddings $NUM_EMB --embedding-dim $EMB_DIM"
  COMMON+=" --rankmixer-tokens-split-dim $TOKENS_SPLIT --rankmixer-blocks $RANKMIXER_BLOCKS --rankmixer-gate-num $GATE_NUM"
  COMMON+=" --read-mode direct --no-read-before-update"
  COMMON+=" --data-dir model_zoo/torchrec_dlrm/processed_day_0_data"
  COMMON+=" --output-root results/bench3node --run-id $run_id"
  COMMON+=" --master-addr $PS_IP --master-port $MASTER_PORT --rdzv-id $rdzv"
  
  launch_all "$COMMON" "$run_id"
}

run_recstore() {
  local run_id="recstore3_$(date +%s)"
  local rdzv="${run_id}_$(date +%s)"
  echo "═══ RECSTORE (PS + bagpipe) run_id=$run_id ═══"
  
  # Start PS server on machine B
  echo "  Starting PS server on $PS_IP..."
  ssh -o StrictHostKeyChecking=no hadoop-quanta@$PS_IP "pkill -9 ps_server 2>/dev/null; sleep 1; rm -rf /dev/shm/recstore_kv /dev/shm/rs_demo_kv 2>/dev/null; cd $REPO && setsid ./build/bin/ps_server --config_path recstore_config.json > /tmp/ps_server.log 2>&1 < /dev/null &" 
  sleep 5
  # Verify PS is up
  ssh -o StrictHostKeyChecking=no hadoop-quanta@$PS_IP "ss -tln | grep -E '1500[01]' | wc -l" 2>&1 | tail -1
  echo "  PS server started."
  
  # Create shared runtime dir on all machines
  for m in "${MACHINES[@]}"; do
    local ip="${m%%:*}"
    ssh -o StrictHostKeyChecking=no hadoop-quanta@$ip "mkdir -p $REPO/results/bench3node/runtime/recstore3node" 2>/dev/null || true
  done
  # Copy recstore_config.json to the runtime dir on all machines
  for m in "${MACHINES[@]}"; do
    local ip="${m%%:*}"
    ssh -o StrictHostKeyChecking=no hadoop-quanta@$ip "cp $REPO/recstore_config.json $REPO/results/bench3node/runtime/recstore3node/ 2>/dev/null" || true
  done
  
  local COMMON="$ENV_EXPORT && CUDA_VISIBLE_DEVICES=0 python3 model_zoo/rs_demo/run_mock_stress.py"
  COMMON+=" --backend recstore --model rankmixer --ps-type BRPC"
  COMMON+=" --recstore-index-type DRAM_PET_HASH --ps-kv-backend recstore_dram"
  COMMON+=" --nnodes 3 --nproc-per-node 1 --no-start-server"
  COMMON+=" --steps $STEPS --warmup-steps $WARMUP --batch-size $BATCH"
  COMMON+=" --num-embeddings $NUM_EMB --embedding-dim $EMB_DIM"
  COMMON+=" --rankmixer-tokens-split-dim $TOKENS_SPLIT --rankmixer-blocks $RANKMIXER_BLOCKS --rankmixer-gate-num $GATE_NUM"
  COMMON+=" --read-mode prefetch --prefetch-depth 0"
  COMMON+=" --enable-gpu-cache --gpu-cache-capacity $GPU_CACHE_CAP"
  COMMON+=" --enable-bagpipe-cache --bagpipe-lookahead $LOOKAHEAD"
  COMMON+=" --data-dir model_zoo/torchrec_dlrm/processed_day_0_data"
  COMMON+=" --output-root results/bench3node --run-id $run_id"
  COMMON+=" --recstore-runtime-dir results/bench3node/runtime/recstore3node"
  COMMON+=" --server-host $PS_IP --server-port0 15000 --server-port1 15001"
  COMMON+=" --master-addr $PS_IP --master-port $MASTER_PORT --rdzv-id $rdzv"
  
  launch_all "$COMMON" "$run_id"
  
  # Stop PS server
  echo "  Stopping PS server..."
  ssh -o StrictHostKeyChecking=no hadoop-quanta@$PS_IP "pkill -9 ps_server 2>/dev/null" || true
}

launch_all() {
  local cmd="$1"
  local run_id="$2"
  local procs=()
  for m in "${MACHINES[@]}"; do
    local ip="${m%%:*}"
    local rank="${m##*:}"
    local full_cmd="$cmd --node-rank $rank"
    local log="/tmp/${run_id}_rank${rank}.log"
    echo "  Launching rank $rank on $ip..."
    ssh -o StrictHostKeyChecking=no hadoop-quanta@$ip "$full_cmd" > "$log" 2>&1 &
    procs+=($!)
  done
  echo "  Waiting for all ranks..."
  local failed=0
  for pid in "${procs[@]}"; do
    if ! wait $pid; then
      failed=1
    fi
  done
  if [ $failed -ne 0 ]; then
    echo "  WARNING: some ranks failed. Check logs."
  fi
  # Collect CSVs from all machines
  echo "  Collecting CSVs..."
  mkdir -p /tmp/$run_id
  for m in "${MACHINES[@]}"; do
    local ip="${m%%:*}"
    local rank="${m##*:}"
    scp -o StrictHostKeyChecking=no -q hadoop-quanta@$ip:$REPO/results/bench3node/outputs/$run_id/recstore_main.csv /tmp/$run_id/recstore_rank${rank}.csv 2>/dev/null || true
  done
  # Show summary
  for m in "${MACHINES[@]}"; do
    local rank="${m##*:}"
    local log="/tmp/${run_id}_rank${rank}.log"
    echo "  rank$rank: $(grep -c 'step' $log 2>/dev/null || echo '?') steps, last: $(tail -3 $log 2>/dev/null | head -1)"
  done
  echo "  CSVs collected to /tmp/$run_id/"
}

LANE="${1:-all}"
case "$LANE" in
  quanta)    run_quanta ;;
  recstore)  run_recstore ;;
  all)       run_quanta; echo; run_recstore ;;
  *)         echo "Usage: $0 [quanta|recstore|all]"; exit 1 ;;
esac
echo "=== Benchmark complete ==="
