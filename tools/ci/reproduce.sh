#!/usr/bin/env bash
# Reproduce the GitHub Actions "Docker Test" lane locally.
#
# Starts the same devel image CI uses, mounts the current checkout, restores
# the cached third_party from the image, and runs the same cmake/ctest
# commands as .github/workflows/test.yml.
#
# Usage:
#   tools/ci/reproduce.sh                  # full CI run (build + all lanes)
#   tools/ci/reproduce.sh -R test_op       # extra args replace the lanes and
#                                          # are passed to ctest verbatim
#   tools/ci/reproduce.sh --shell          # drop into a shell in the container
#
# Environment variables:
#   RECSTORE_CI_IMAGE   image to run (default: ghcr.io/recstore/recstore-devel:latest)
#   RECSTORE_CI_BUILD   build directory inside the container (default: build)
#   RECSTORE_CI_SKIP_THIRD_PARTY_RESTORE=1
#                       keep the checkout's third_party instead of restoring
#                       the image cache
#
# Notes:
# - The container runs as root, so files it creates in the mounted checkout
#   (build/, runner/logs/) will be owned by root afterwards.
# - Pulling the image may require `docker login ghcr.io` if it is not public.

set -euo pipefail

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

IMAGE="${RECSTORE_CI_IMAGE:-ghcr.io/recstore/recstore-devel:latest}"
BUILD_DIR="${RECSTORE_CI_BUILD:-build}"
MODE="ci"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
fi
if [[ "${1:-}" == "--shell" ]]; then
  MODE="shell"
  shift
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

CONTAINER_ARGS=(
  --rm
  --network host
  --security-opt seccomp=unconfined
  -v "${REPO_ROOT}:/workspace"
  -w /workspace
)

if [[ "${MODE}" == "shell" ]]; then
  echo "Starting shell in ${IMAGE} (checkout mounted at /workspace)"
  exec docker run -it "${CONTAINER_ARGS[@]}" "${IMAGE}" bash -l
fi

echo "Running CI-equivalent build and tests in ${IMAGE}"

# Any extra arguments are forwarded as ctest arguments inside the container
# (they replace the lane selection, e.g. -R <test> to focus on one test).
docker run "${CONTAINER_ARGS[@]}" "${IMAGE}" bash -lc '
  set -euo pipefail
  # The trailing "_" argument is $0; any extra args become $@ (forwarded to
  # ctest below when the caller wants to focus on specific tests).
  export CI=true
  export GLOG_logtostderr=1
  export GLOG_v=1
  BUILD_DIR="'"${BUILD_DIR}"'"
  SKIP_RESTORE="'"${RECSTORE_CI_SKIP_THIRD_PARTY_RESTORE:-0}"'"

  cd /workspace
  mkdir -p runner/logs

  if [ "${SKIP_RESTORE}" != "1" ]; then
    CACHE_DIR="/opt/recstore-cache/third_party"
    if [ -d "${CACHE_DIR}" ]; then
      echo "Restoring initialized third_party from ${CACHE_DIR} ..."
      rm -rf /workspace/third_party
      cp -a "${CACHE_DIR}" /workspace/third_party
    else
      echo "No third_party cache in image; using the checkout as-is."
    fi
  fi

  if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake -DENABLE_CUDA=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      ..
  else
    cd "${BUILD_DIR}"
  fi
  make -j"$(nproc)"

  cd "/workspace/${BUILD_DIR}"

  if [ "$#" -gt 0 ]; then
    echo "===== ctest (custom args) ====="
    : > /workspace/runner/logs/ctest.log
    ctest --output-on-failure --timeout 300 "$@"
    exit $?
  fi

  CTEST_EXCLUDE_REGEX=""
  if ! command -v memcached >/dev/null 2>&1; then
    echo "memcached not found; skipping RDMA integration tests."
    CTEST_EXCLUDE_REGEX="pytorch_client_test_rdma_basic|pytorch_client_test_rdma"
  fi

  run_lane() {
    echo "===== ctest lane: $1 ====="
    ctest --output-on-failure --progress --timeout 300 \
      -E "${CTEST_EXCLUDE_REGEX}" "${@:2}"
  }

  : > /workspace/runner/logs/ctest.log
  status=0
  run_lane unit -LE "python|rdma_integration" \
    2>&1 | tee -a /workspace/runner/logs/ctest.log || status=1
  run_lane python -L "python" \
    2>&1 | tee -a /workspace/runner/logs/ctest.log || status=1
  run_lane rdma_integration -L "rdma_integration" \
    2>&1 | tee -a /workspace/runner/logs/ctest.log || status=1

  python3 /workspace/tools/ci/ctest_annotations.py \
    /workspace/runner/logs/ctest.log || true

  if [ "${status}" -ne 0 ]; then
    echo "ctest failed; log: /workspace/runner/logs/ctest.log"
    exit 1
  fi
' _ "$@"
