#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_JSON_PATH="${REPO_ROOT}/recstore_config.json"
if [[ -f "${CONFIG_JSON_PATH}" ]]; then
    if command -v jq >/dev/null 2>&1; then
        TMP_JSON="${CONFIG_JSON_PATH}.tmp"
        jq '.cache_ps.base_kv_config.capacity = 512
            | .cache_ps.max_batch_keys_size = 128
            | .cache_ps.num_threads = 4
            | .distributed_client.max_keys_per_request = 32
            | .cache_ps.base_kv_config.path = "/tmp/recstore_data"
            | .cache_ps.base_kv_config.index_type = "DRAM"
            | .cache_ps.base_kv_config.value_type = "SSD"
            | .cache_ps.base_kv_config.type = "DRAM"
            | .cache_ps.base_kv_config.queue_size = 1024' "${CONFIG_JSON_PATH}" > "${TMP_JSON}" && mv "${TMP_JSON}" "${CONFIG_JSON_PATH}"
        echo "Updated config fields in recstore_config.json using jq."
    else
        export RECSTORE_REPO_ROOT="${REPO_ROOT}"
        python3 - <<'PY'
import json, sys, os
root = os.environ.get('RECSTORE_REPO_ROOT', os.getcwd())
path = os.path.join(root, 'recstore_config.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
try:
    data['cache_ps']['base_kv_config']['capacity'] = 512
    data['cache_ps']['max_batch_keys_size'] = 128
    data['cache_ps']['num_threads'] = 4
    data['distributed_client']['max_keys_per_request'] = 32
    data['cache_ps']['base_kv_config']['path'] = '/tmp/recstore_data'
    data['cache_ps']['base_kv_config']['index_type'] = "DRAM"
    data['cache_ps']['base_kv_config']['value_type'] = "SSD"
    data['cache_ps']['base_kv_config']['type'] = "DRAM"
    data['cache_ps']['base_kv_config']['queue_size'] = 1024
except Exception as e:
    print(f"Failed to set capacity: {e}")
    sys.exit(1)
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
print('Updated config fields in recstore_config.json using Python.')
PY
    fi
else
    echo "recstore_config.json not found at ${CONFIG_JSON_PATH}; skipping capacity update."
fi

mkdir -p /tmp/recstore_data || true