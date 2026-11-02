#!/bin/bash
# =============================================================================
# RecStore SPDK build setup for this CentOS 7 host (no root password).
#
# Builds SPDK + DPDK from third_party/spdk into the repo tree and enables
# the RecStore SPDK IO backend (RECSTORE_ENABLE_SPDK=ON) in build/.
#
# Notes / deviations from the in-tree Ubuntu flow (dockerfiles/init_env_inside_docker.sh):
#   - sudo is only allowlisted for yum/pip/etc, so SPDK is built in-tree
#     (not `make install`ed to /usr/local) and CMake finds it via pkg-config.
#   - nasm on this host is 2.10 (<2.11), so a newer nasm is built into ~/.local.
#   - The system libarchive lacks zstd, which breaks DPDK; a pkg-config shim
#     hides libarchive from the DPDK build.
#   - SPDK's generated spdk_sock<->spdk_sock_posix .pc Requires cycle segfaults
#     the ancient pkg-config 0.27.1; a curated .pc shim breaks the cycle.
#   - SPDK/DPDK are static; DPDK's shared libs are stashed so the linker uses
#     the static ones, and the DPDK lib set is passed via CMAKE_CXX_STANDARD_LIBRARIES
#     because the in-tree CMake list predates current DPDK.
#   - SPDK nvme_auth needs OpenSSL >= 3.0 (host has 1.0.2); an existing conda
#     OpenSSL 3.x is symlinked and put first on the linker search path.
#
# Runtime (NOT done here -- requires privileges this host does not have):
#   - hugepages (e.g. nr_hugepages) and /dev/hugepages
#   - an NVMe device bound to vfio-pci/uio_pci_generic and its PCIe BDF set in
#     src/storage/io_backend/spdk/spdk_backend.cpp (pcie_address, default 0000:c2:00.0)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="${RECSTORE_DEPS_PREFIX:-$HOME/recstore_deps}"
SPDK="$REPO/third_party/spdk"
OSSL_SRC="${RECSTORE_OPENSSL3_DIR:-$HOME/.conda/envs/quanta_rankmixer}"
OSSL_LINK="$DEPS/openssl3"
PC_SHIM="$DEPS/spdk_pc"
NPROC=$(nproc)
JOBS="${JOBS:-$NPROC}"
export PATH="$HOME/.local/bin:$PATH"

echo "[spdk] repo=$REPO deps=$DEPS jobs=$JOBS"

# --- 0. host build deps -------------------------------------------------------
sudo -n yum -y install fuse3-devel nasm python3-devel numactl-devel openssl-devel libiscsi-devel
pip3 install --user -i https://pypi.tuna.tsinghua.edu.cn/simple meson pyelftools

# --- 1. submodules ------------------------------------------------------------
git -C "$REPO" submodule update --init --depth 1 third_party/spdk
git -C "$SPDK" submodule update --init --depth 1 dpdk isa-l intel-ipsec-mb

# --- 2. nasm >= 2.11 into ~/.local (2.15.05 builds with autoconf 2.69) ---------
if [ "$("$HOME/.local/bin/nasm" -v 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" \< "2.11.01" ] 2>/dev/null || ! command -v "$HOME/.local/bin/nasm" >/dev/null; then
  TMP=$(mktemp -d)
  curl -fsSL -o "$TMP/nasm.tgz" https://github.com/netwide-assembler/nasm/archive/refs/tags/nasm-2.15.05.tar.gz
  tar xzf "$TMP/nasm.tgz" -C "$TMP" --strip-components=1
  ( cd "$TMP" && ./autogen.sh && ./configure --prefix="$HOME/.local" && make -j"$JOBS" && make install || true )
fi

# --- 3. isa-l (CMakeLists links third_party/spdk/isa-l/.libs/libisal.a) --------
( cd "$SPDK/isa-l" && ./autogen.sh && ./configure --prefix="$PWD" && make -j"$JOBS" )
test -f "$SPDK/isa-l/.libs/libisal.a"

# --- 4. build SPDK/DPDK with libarchive hidden --------------------------------
SHIM=$(mktemp -d)
for f in /usr/lib64/pkgconfig/*.pc /usr/share/pkgconfig/*.pc; do
  b=$(basename "$f"); [ "$b" = libarchive.pc ] && continue; ln -sf "$f" "$SHIM/$b"
done
[ -d "$OSSL_LINK" ] || { mkdir -p "$OSSL_LINK"; ln -sfn "$OSSL_SRC/include" "$OSSL_LINK/include"; ln -sf "$OSSL_SRC/lib/libssl.so" "$OSSL_LINK/libssl.so"; ln -sf "$OSSL_SRC/lib/libcrypto.so" "$OSSL_LINK/libcrypto.so"; }
( cd "$SPDK" && PKG_CONFIG_LIBDIR="$SHIM" ./configure --prefix="$DEPS/spdk" --with-openssl="$OSSL_LINK" \
    --disable-tests --disable-unit-tests --disable-examples --disable-apps --without-idxd --without-crypto \
  && PKG_CONFIG_LIBDIR="$SHIM" make -j"$JOBS" )

# --- 5. .pc shim (break spdk_sock<->spdk_sock_posix Requires cycle) -----------
mkdir -p "$PC_SHIM"
cp "$SPDK"/build/lib/pkgconfig/*.pc "$PC_SHIM"/
python3 - "$PC_SHIM/spdk_sock_modules.pc" <<'PY'
import sys
p = sys.argv[1]
open(p, "w").write("".join(l for l in open(p) if not l.startswith("Requires:")))
PY

# --- 6. use static DPDK (stash shared libs) -----------------------------------
if ls "$SPDK"/dpdk/build/lib/*.so >/dev/null 2>&1; then
  mkdir -p "$SPDK/dpdk/build/lib/shared-stash"
  for f in "$SPDK"/dpdk/build/lib/*.so "$SPDK"/dpdk/build/lib/*.so.*; do
    [ -e "$f" ] && mv "$f" "$SPDK/dpdk/build/lib/shared-stash/"
  done
fi

# --- 6b. runtime OpenSSL 3 libs (SPDK nvme_auth links libssl.so.3/libcrypto.so.3)
# recstore_deps/lib64 is already on the binaries' rpath, so no LD_LIBRARY_PATH needed.
cp -L "$OSSL_SRC/lib/libssl.so.3" "$OSSL_SRC/lib/libcrypto.so.3" "$DEPS/lib64/"

# --- 7. configure RecStore with SPDK enabled ----------------------------------
DPDKLIBS=$(PKG_CONFIG_PATH="$PC_SHIM" pkg-config --libs spdk_dpdklibs)
PKG_CONFIG_PATH="$PC_SHIM" cmake -S "$REPO" -B "$REPO/build" -DCMAKE_BUILD_TYPE=Release \
  -DRECSTORE_ENABLE_SPDK=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-L$OSSL_LINK -L$DEPS/lib -L$DEPS/lib64" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-Wl,--start-group $DPDKLIBS -Wl,--end-group"
cmake --build "$REPO/build" --target benchmark_kv_engine test_io_backend -j"$JOBS"

echo "[spdk] done. binary: $REPO/build/bin/benchmark_kv_engine"
echo "[spdk] runtime still needs hugepages + a vfio/uio-bound NVMe; see header."
