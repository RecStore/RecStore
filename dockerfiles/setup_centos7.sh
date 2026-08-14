#!/bin/bash
# =============================================================================
# RecStore CentOS 7 host setup script (no Docker, no root password required)
#
# Builds recstore_ops (_recstore_ops.so) + ps_server + RDMA (petps_server)
# on a bare CentOS 7 host that already has:
#   - gcc >= 11  (this host: 11.5.0)   - cmake >= 3.10 (this host: 3.31.5)
#   - CUDA 12.x + nvcc                 - A100 (sm_80)
#   - pip torch 2.7.1+cu126, torchrec, fbgemm-gpu (cu126, cxx11_abi=True)
#   - passwordless `sudo yum` (NOPASSWD allowlist); other sudo NOT available,
#     so every dependency is installed under a user-writable prefix.
#
# Usage:
#   bash dockerfiles/setup_centos7.sh            # full setup
#   bash dockerfiles/setup_centos7.sh --build    # only the RecStore cmake/build step
#
# Re-runnable: each stage is idempotent. Set RECSTORE_SKIP_SUBMODULES=1 to skip
# the (slow) git submodule init if already done.
# =============================================================================
set -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEPS_PREFIX="${RECSTORE_DEPS_PREFIX:-$HOME/recstore_deps}"
PIP_SITE=/usr/local/python3.12.11/lib/python3.12/site-packages
NPROC=$(nproc)
export MAKEFLAGS="-j${NPROC}"

# CentOS installs 64-bit libs under lib64/; some build systems use lib/.
# We keep both on every search path.
DEPS_LIB="$DEPS_PREFIX/lib64;$DEPS_PREFIX/lib"
DEPS_INC="$DEPS_PREFIX/include"

mkdir -p "$DEPS_PREFIX"
echo "[setup] REPO=$REPO  DEPS_PREFIX=$DEPS_PREFIX  nproc=$NPROC"

# -----------------------------------------------------------------------------
# ensure_git_ref <dir> <tag>   (checkout a tag, fetching it shallow if needed)
# -----------------------------------------------------------------------------
ensure_git_ref() {
  local dir="$1" ref="$2"
  ( cd "$dir" && git rev-parse --verify --quiet "${ref}^{commit}" >/dev/null 2>&1 && git checkout -q "${ref}" ) \
    || ( cd "$dir" && git fetch --depth 1 origin "refs/tags/${ref}:refs/tags/${ref}" && git checkout -q "${ref}" )
}

# =============================================================================
# Stage 1 — system packages (yum). sudo is NOPASSWD for yum only.
# =============================================================================
stage_system() {
  /bin/sudo yum -y install \
    make autoconf automake libtool wget unzip patch pkgconfig perl \
    boost-devel double-conversion-devel libevent-devel libunwind-devel \
    libzstd-devel snappy-devel lz4-devel bzip2-devel xz-devel \
    libibverbs-devel libaio-devel openssl-devel \
    leveldb-devel libcurl-devel elfutils-libelf-devel binutils-devel gflags-devel
}

# =============================================================================
# Stage 2 — git submodules (skip SSD-only: spdk, HugeCTR, gpudirect-nvme)
# =============================================================================
stage_submodules() {
  if [ "${RECSTORE_SKIP_SUBMODULES:-0}" = "1" ]; then echo "[setup] skipping submodules"; return; fi
  cd "$REPO"
  git submodule update --init --recursive \
    third_party/fmt third_party/folly third_party/glog third_party/googletest \
    third_party/json third_party/grpc third_party/jemalloc third_party/liburing \
    third_party/brpc third_party/cityhash third_party/cpptrace third_party/faster
}

# =============================================================================
# Stage 3 — build third-party libraries from source
# =============================================================================
stage_thirdparty() {
  # ---- liburing (in-tree; folly uses src/liburing.a directly) ----
  cd "$REPO/third_party/liburing"
  [ -x configure ] || autoreconf -fi
  ./configure --cc=gcc --cxx=g++
  make -C src
  test -f src/liburing.a

  # ---- fmt -> $DEPS_PREFIX ----
  cd "$REPO/third_party/fmt"; rm -rf _build; mkdir _build; cd _build
  CXXFLAGS="-fPIC" cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="$DEPS_PREFIX"
  make && make install

  # ---- cityhash -> in-tree include/lib (FindCityhash looks in third_party/cityhash) ----
  cd "$REPO/third_party/cityhash"
  [ -f configure ] || autoreconf -fi
  ./configure --prefix="$PWD"
  make && make install
  test -f include/city.h && test -f lib/libcityhash.a

  # ---- gflags 2.2.2 -> $DEPS_PREFIX  (CentOS gflags 2.1.1 lacks GFLAGS_NAMESPACE) ----
  cd "$REPO/third_party"
  [ -d gflags-2.2.2 ] || { wget -q https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz -O gflags-2.2.2.tar.gz; tar xzf gflags-2.2.2.tar.gz; }
  cd gflags-2.2.2; rm -rf _build; mkdir _build; cd _build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="$DEPS_PREFIX" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF
  make && make install

  # ---- glog v0.5.0 -> third_party/glog/glog-install-fPIC (built against gflags 2.2.2) ----
  cd "$REPO/third_party/glog"; ensure_git_ref . v0.5.0
  rm -rf _build; mkdir _build; cd _build
  CXXFLAGS="-fPIC" cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTING=OFF -DWITH_GFLAGS=ON -DWITH_GTEST=OFF \
    -DCMAKE_PREFIX_PATH="$DEPS_PREFIX"
  make
  rm -rf "$REPO/third_party/glog/glog-install-fPIC"
  make DESTDIR="$REPO/third_party/glog/glog-install-fPIC" install
  # CMake hard-codes usr/local/lib/cmake/glog; CentOS installs to lib64 -> symlink
  ( cd "$REPO/third_party/glog/glog-install-fPIC/usr/local" && [ -e lib ] || ln -s lib64 lib )

  # ---- cpptrace v0.3.1 -> $DEPS_PREFIX ----
  cd "$REPO/third_party/cpptrace"; ensure_git_ref . v0.3.1
  rm -rf build; mkdir build; cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="$DEPS_PREFIX" -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  make && make install

  # ---- oneTBB v2021.13.0 -> $DEPS_PREFIX  (code uses oneapi/tbb + TBB::tbb) ----
  cd "$REPO/third_party"
  [ -d onetbb ] || git clone --depth 1 --branch v2021.13.0 https://github.com/oneapi-src/oneTBB.git onetbb
  cd onetbb; rm -rf build; mkdir build; cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="$DEPS_PREFIX" -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF
  make && make install

  # ---- OpenSSL 1.1.1w -> $DEPS_PREFIX  (CentOS system openssl 1.0.2k too old for folly/grpc) ----
  cd "$REPO/third_party"
  [ -d openssl-1.1.1w ] || { wget -q https://github.com/openssl/openssl/releases/download/OpenSSL_1_1_1w/openssl-1.1.1w.tar.gz; tar xzf openssl-1.1.1w.tar.gz; }
  cd openssl-1.1.1w
  ./config --prefix="$DEPS_PREFIX" --openssldir="$DEPS_PREFIX/ssl" shared zlib
  make -j"$NPROC" && make install_sw

  # ---- boost 1.75.0 -> $DEPS_PREFIX  (CentOS boost 1.53 lacks headers folly 2023 needs) ----
  cd "$REPO/third_party"
  [ -d boost_1_75_0 ] || { wget -q https://archives.boost.org/release/1.75.0/source/boost_1_75_0.tar.gz; tar xzf boost_1_75_0.tar.gz; }
  cd boost_1_75_0
  ./bootstrap.sh --prefix="$DEPS_PREFIX" \
    --with-libraries=context,coroutine,filesystem,program_options,regex,system,thread,chrono,date_time,atomic,serialization
  ./b2 -j"$NPROC" cxxstd=17 cflags=-fPIC cxxflags=-fPIC install

  # ---- folly v2023.09.11.00 -> third_party/folly/folly-install-fPIC ----
  cd "$REPO/third_party/folly"; ensure_git_ref . v2023.09.11.00
  rm -rf _build; mkdir _build; cd _build
  export CC=/usr/bin/gcc CXX=/usr/bin/g++ OPENSSL_ROOT_DIR="$DEPS_PREFIX"
  CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
    -DCMAKE_INCLUDE_PATH="$REPO/third_party/glog/glog-install-fPIC/usr/local/include" \
    -DCMAKE_LIBRARY_PATH="$REPO/third_party/glog/glog-install-fPIC/usr/local/lib64" \
    -DLIBURING_INCLUDE_DIR="$REPO/third_party/liburing/src/include" \
    -DLIBURING_LIBRARY="$REPO/third_party/liburing/src/liburing.a" \
    -DOPENSSL_ROOT_DIR="$DEPS_PREFIX" -DBOOST_ROOT="$DEPS_PREFIX" -DBoost_NO_SYSTEM_PATHS=ON \
    -DBUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  make -j8
  rm -rf "$REPO/third_party/folly/folly-install-fPIC"
  make DESTDIR="$REPO/third_party/folly/folly-install-fPIC" install
  ( cd "$REPO/third_party/folly/folly-install-fPIC/usr/local" && [ -e lib ] || ln -s lib64 lib )

  # ---- gRPC -> third_party/grpc-install  (uses bundled boringssl; no system openssl dep) ----
  cd "$REPO/third_party/grpc"
  rm -rf cmake/build; mkdir -p cmake/build; cd cmake/build
  cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$REPO/third_party/grpc-install" \
    -DOPENSSL_ROOT_DIR="$DEPS_PREFIX" ../..
  make -j8 && make install

  # ---- protobuf (from grpc submodule) -> third_party/protobuf-install ----
  PB_SRC="$REPO/third_party/grpc/third_party/protobuf"
  mkdir -p "$PB_SRC/_build"; cd "$PB_SRC/_build"
  cmake "$PB_SRC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$REPO/third_party/protobuf-install" \
    -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=ON
  make && make install
  ( cd "$REPO/third_party/protobuf-install" && [ -e lib ] || ln -s lib64 lib )

  # ---- leveldb 1.23 (cxx11 ABI, static, NO snappy) -> $DEPS_PREFIX + third_party/leveldb
  #      CentOS leveldb 1.12 is old-ABI (brpc_ps_client needs ToString[abi:cxx11]).
  #      Snappy is disabled: a static libleveldb.a with snappy leaves undefined snappy
  #      symbols when ps_server links it. ----
  cd "$REPO/third_party"
  [ -d leveldb ] || git clone --depth 1 --branch 1.23 https://github.com/google/leveldb.git leveldb
  # Force-disable snappy detection (check_library_exists ignores -D overrides).
  sed -i 's/^check_library_exists(snappy snappy_compress "" HAVE_SNAPPY)/set(HAVE_SNAPPY 0 CACHE INTERNAL "snappy disabled for RecStore")/' \
    "$REPO/third_party/leveldb/CMakeLists.txt"
  cd leveldb; rm -rf build; mkdir build; cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=1" \
    -DCMAKE_INSTALL_PREFIX="$DEPS_PREFIX" -DLEVELDB_BUILD_TESTS=OFF -DLEVELDB_BUILD_BENCHMARKS=OFF
  make && make install
  # also install under third_party/leveldb (src/ps/brpc/CMakeLists link_directories uses it)
  cmake .. -DCMAKE_INSTALL_PREFIX="$REPO/third_party/leveldb" -DLEVELDB_BUILD_TESTS=OFF -DLEVELDB_BUILD_BENCHMARKS=OFF
  make install
  ( cd "$REPO/third_party/leveldb" && [ -e lib ] || ln -s lib64 lib )

  # ---- brpc -> third_party/brpc-install  (build against $DEPS openssl 1.1.1w; the
  #      src/ps/brpc/CMakeLists.txt patch below makes the PS client link the SAME 1.1.1.
  #      folly also needs 1.1.1, so the whole link must agree on one openssl.) ----
  BRPC_SRC="$REPO/third_party/brpc"
  GLOG_INC="$REPO/third_party/glog/glog-install-fPIC/usr/local/include"
  GLOG_LIB="$REPO/third_party/glog/glog-install-fPIC/usr/local/lib64"
  PB_INC="$REPO/third_party/protobuf-install/include"; PB_LIB="$REPO/third_party/protobuf-install/lib64"
  rm -rf "$BRPC_SRC/_build"; mkdir -p "$BRPC_SRC/_build"; cd "$BRPC_SRC/_build"
  cmake "$BRPC_SRC" \
    -DProtobuf_INCLUDE_DIR="$PB_INC" -DProtobuf_LIBRARIES="$PB_LIB/libprotobuf.so" \
    -DProtobuf_PROTOC_EXECUTABLE="$REPO/third_party/protobuf-install/bin/protoc" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_PREFIX="$REPO/third_party/brpc-install" \
    -DWITH_GLOG=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_BRPC_TOOLS=OFF \
    -DOPENSSL_ROOT_DIR="$DEPS_PREFIX" -DOPENSSL_INCLUDE_DIR="$DEPS_PREFIX/include" \
    -DCMAKE_INCLUDE_PATH="$GLOG_INC;$PB_INC;$DEPS_INC" \
    -DCMAKE_LIBRARY_PATH="$GLOG_LIB;$PB_LIB;$DEPS_PREFIX/lib;$DEPS_PREFIX/lib64"
  make && make install
  ( cd "$REPO/third_party/brpc-install" && [ -e lib ] || (ls lib64 >/dev/null 2>&1 && ln -s lib64 lib) )

  # ---- patch src/ps/brpc/CMakeLists.txt: the openssl find_library uses NO_DEFAULT_PATH
  #      against /usr/lib64 (1.0.2k on CentOS 7). Add ${OPENSSL_ROOT_DIR}/lib64 so it
  #      finds the 1.1.1w that brpc/folly were built against. (Idempotent.) ----
  BRPC_PS_CMAKE="$REPO/src/ps/brpc/CMakeLists.txt"
  if ! grep -q 'OPENSSL_ROOT_DIR}/lib64' "$BRPC_PS_CMAKE"; then
    sed -i 's#PATHS /usr/include#PATHS ${OPENSSL_ROOT_DIR}/include /usr/include#' "$BRPC_PS_CMAKE"
    sed -i 's#PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu#PATHS ${OPENSSL_ROOT_DIR}/lib64 ${OPENSSL_ROOT_DIR}/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu#g' "$BRPC_PS_CMAKE"
  fi

  # ---- libtorch: point the hard-coded third_party/libtorch path at the pip torch ----
  PIP_TORCH="$PIP_SITE/torch"
  mkdir -p "$REPO/third_party/libtorch"
  [ -e "$REPO/third_party/libtorch/libtorch" ] || ln -s "$PIP_TORCH" "$REPO/third_party/libtorch/libtorch"
}

# =============================================================================
# Stage 4 — configure & build RecStore
# =============================================================================
stage_build() {
  cd "$REPO"
  rm -rf build; mkdir build; cd build
  # -L$DEPS/lib[64] + rpath so BARE -l names (boost_context, leveldb, ...) resolve to
  # the $DEPS builds, not the old CentOS 7 system libs (boost 1.53, leveldb old-ABI).
  local LDFLAGS="-L$DEPS_PREFIX/lib -L$DEPS_PREFIX/lib64 -Wl,-rpath,$DEPS_PREFIX/lib64 -Wl,-rpath,$DEPS_PREFIX/lib"
  cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DPYBIND11_CMAKE_DIR="$PIP_SITE/pybind11/share/cmake/pybind11" \
    -DTBB_DIR="$DEPS_PREFIX/lib64/cmake/TBB" \
    -Dcpptrace_DIR="$DEPS_PREFIX/lib64/cmake/cpptrace" \
    -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" -DOPENSSL_ROOT_DIR="$DEPS_PREFIX" \
    -DCURL_LIBRARY=/usr/lib64/libcurl.so -DCURL_INCLUDE_DIR=/usr/include \
    -DPython_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
    -DCMAKE_MODULE_LINKER_FLAGS="$LDFLAGS" \
    -DCMAKE_BUILD_RPATH="$DEPS_PREFIX/lib64:$DEPS_PREFIX/lib" \
    ..
  make -j"$NPROC" recstore_torch_ops ps_server petps_server
  echo "[setup] _recstore_ops.so: $(ls -la lib/_recstore_ops.so 2>/dev/null)"
  echo "[setup] ps_server:        $(ls -la bin/ps_server 2>/dev/null)"
  echo "[setup] petps_server:     $(ls -la bin/petps_server 2>/dev/null)"
}

case "${1:-all}" in
  --system)     stage_system ;;
  --submodules) stage_submodules ;;
  --thirdparty) stage_thirdparty ;;
  --build)      stage_build ;;
  all) stage_system; stage_submodules; stage_thirdparty; stage_build ;;
  *) echo "usage: $0 [--system|--submodules|--thirdparty|--build|all]"; exit 1 ;;
esac
