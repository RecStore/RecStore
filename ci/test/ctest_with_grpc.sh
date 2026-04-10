#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIB_PATH="${REPO_ROOT}/build/lib/lib_recstore_ops.so"
PY_CLIENT_DIR="${REPO_ROOT}/src/framework/pytorch/python_client"
PY_PKG_ROOT="${REPO_ROOT}/src/python/pytorch"

GRPC_SERVER_PATH="${REPO_ROOT}/build/bin/grpc_ps_server"

CONFIG_JSON_PATH="${REPO_ROOT}/recstore_config.json"
if [[ -f "${CONFIG_JSON_PATH}" ]]; then
    if command -v jq >/dev/null 2>&1; then
        TMP_JSON="${CONFIG_JSON_PATH}.tmp"
        jq '.cache_ps.base_kv_config.capacity = 65536' "${CONFIG_JSON_PATH}" > "${TMP_JSON}" && mv "${TMP_JSON}" "${CONFIG_JSON_PATH}"
        echo "Updated capacity in recstore_config.json using jq."
    else
        export RECSTORE_REPO_ROOT="${REPO_ROOT}"
        python3 - <<'PY'
import json, sys, os
root = os.environ.get('RECSTORE_REPO_ROOT', os.getcwd())
path = os.path.join(root, 'recstore_config.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
try:
    data['cache_ps']['base_kv_config']['capacity'] = 65536
except Exception as e:
    print(f"Failed to set capacity: {e}")
    sys.exit(1)
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
print('Updated capacity in recstore_config.json using Python.')
PY
    fi
else
    echo "recstore_config.json not found at ${CONFIG_JSON_PATH}; skipping capacity update."
fi

export RECSTORE_REPO_ROOT="${REPO_ROOT}"
KV_PATH="$(python3 - <<'PY'
import json, os
root = os.environ.get('RECSTORE_REPO_ROOT', os.getcwd())
path = os.path.join(root, 'recstore_config.json')
try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(data.get('cache_ps', {}).get('base_kv_config', {}).get('path', '/tmp/recstore_data'))
except Exception:
    print('/tmp/recstore_data')
PY
)"
mkdir -p "${KV_PATH}" || true
echo "Ensured KV path exists: ${KV_PATH}"

export LD_LIBRARY_PATH="${REPO_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PY_PKG_ROOT}:${PYTHONPATH:-}"

if [[ "$@" == *"--mock"* ]]; then
    echo "Mock mode enabled; skipping this script."
    exit 0
fi

# Ensure ctest is available (fallback to pip cmake bin)
PY_CMAKE_BIN=$(python3 - <<'PY'
import cmake, os
print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin'))
PY
)
export PATH="${PY_CMAKE_BIN}:$PATH"

LOG_DIR="${REPO_ROOT}/build/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/grpc_ps_server.log"

echo "Starting grpc_ps_server... (log: ${SERVER_LOG})"
"${GRPC_SERVER_PATH}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 ${SERVER_PID} >/dev/null 2>&1; then
        kill ${SERVER_PID} || true
        wait ${SERVER_PID} 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server ready signal
READY_TIMEOUT=1200
echo "Waiting for grpc_ps_server to be ready..."
START_TS=$(date +%s)
while true; do
    if grep -q 'listening on' "${SERVER_LOG}"; then
        echo "grpc_ps_server is ready. Running ctest --verbose"
        break
    fi
    if ! kill -0 ${SERVER_PID} >/dev/null 2>&1; then
        echo "grpc_ps_server exited before readiness signal."
        echo "==== grpc_ps_server log ===="
        cat "${SERVER_LOG}"
        exit 1
    fi
    NOW_TS=$(date +%s)
    if (( NOW_TS - START_TS >= READY_TIMEOUT )); then
        echo "grpc_ps_server did not report 'listening on' within ${READY_TIMEOUT}s"
        echo "==== grpc_ps_server log ===="
        cat "${SERVER_LOG}"
        exit 1
    fi
    sleep 1
done
echo "==== grpc_ps_server log ===="
cat "${SERVER_LOG}"

cd "${REPO_ROOT}/build"
export LD_LIBRARY_PATH="${REPO_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
ctest --verbose

echo "All tests finished successfully."