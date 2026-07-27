# -*- coding: utf-8 -*-
"""RecStore Python package build script.

Builds the ``recstore._recstore_ops`` native extension (a single shared
object that registers ``torch.ops.recstore_ops`` via ``TORCH_LIBRARY``) and
packages the pure-Python ``recstore`` package located under
``src/python/pytorch/recstore``.

The native extension reuses the C++ static/shared libraries already produced
by the project's top-level CMake build (``build/lib/*.a`` / ``*.so``).  The
canonical link line CMake emits lives at
``build/src/framework/pytorch/CMakeFiles/recstore_torch_ops.dir/link.txt``;
``setup.py`` reproduces that link line as data so the resulting ``.so`` is
byte-for-byte compatible with the CMake-produced ``lib_recstore_ops.so``.

Usage:
    # Development (in-place build; .so lands next to the package)
    pip install -e .                # or: python setup.py build_ext --inplace

    # Wheel (for distribution)
    pip wheel .                     # or: python setup.py bdist_wheel
"""

import glob
import os
import shutil
import subprocess
import sys
import time
import socket
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# setup.py lives at src/python/setup.py.  BASEDIR is this directory
# (used for package-relative paths).  REPO_ROOT is two levels up and is
# used for C++ sources, third_party/, and the CMake build/ dir.
BASEDIR = os.path.dirname(os.path.realpath(__file__))  # src/python/
REPO_ROOT = os.path.abspath(os.path.join(BASEDIR, "..", ".."))
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party")
BUILD_DIR = os.path.join(REPO_ROOT, "build")  # CMake build root (already populated)
BUILD_LIB = os.path.join(BUILD_DIR, "lib")  # CMake artifact dir (libbase.a, etc.)


def _env_path(env_var: str, default: str) -> str:
    """Allow overriding key directories via env vars (for worktree/CI use)."""
    val = os.environ.get(env_var)
    if val:
        return val
    return default


# Allow pointing at a sibling checkout when this worktree doesn't have
# submodules initialized (saves re-cloning 5GB of third_party/grpc etc.).
THIRD_PARTY = _env_path("RECSTORE_THIRD_PARTY_DIR", THIRD_PARTY)
BUILD_DIR = _env_path("RECSTORE_BUILD_DIR", BUILD_DIR)
BUILD_LIB = os.path.join(BUILD_DIR, "lib")


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
def _git(args, default="unknown"):
    try:
        out = subprocess.check_output(
            ["git"] + args, cwd=BASEDIR, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() or default
    except Exception:
        return default


def _git_show(fmt: str) -> str:
    return _git(["show", "-s", f"--format={fmt}", "HEAD"])


def get_version() -> str:
    """Dynamic version: base + optional +cuXYZ suffix."""
    base = "0.1.0"
    cuda_version = os.getenv("CUDA_VERSION")
    if cuda_version:
        parts = cuda_version.split(".")
        return f"{base}+cu{parts[0]}{parts[1]}"
    return base


# ---------------------------------------------------------------------------
# Native extension sources and link line
# ---------------------------------------------------------------------------
# Mirrors src/framework/pytorch/CMakeLists.txt -> add_library(recstore_torch_ops).
# Relative paths (relative to this setup.py at src/python/) satisfy
# setuptools 82's strict no-absolute-path check on setup() args.
# RecStoreBuildExt.run() converts them to absolute before ninja runs,
# so pip's temp build dir doesn't break (ninja flattens absolute paths
# under the temp dir, but relative ``../../`` paths escape it).
EXT_SOURCES = [
    "../../src/framework/pytorch/op_torch.cc",
    "../../src/framework/op.cc",
    "../../src/framework/common/local_shm_kv_client_op.cc",
    "../../src/framework/common/hierkv_local_runtime.cc",
    "../../src/framework/common/op_runtime_support.cc",
    "../../src/framework/common/local_shm_op_component.cc",
    "../../src/framework/common/ps_client_config_adapter.cc",
    "../../src/ps/client_factory.cc",
    "../../src/ps/rdma/rc_options.cc",
    "../../src/ps/rdma/rdma_ps_client_adapter.cc",
    # Boost regex instantiation that the CMake target compiles as a sibling .o
    # so the resulting .so does not depend on a system libboost_regex.
    "../../src/framework/pytorch/boost_regex_cxx11_inst.cc",
]

EXT_COMPILE_DEFS = [
    "-DCPPTRACE_STATIC_DEFINE",
    "-DGFLAGS_IS_A_DLL=0",
    "-DGLOG_USE_GLOG_EXPORT",
    "-DUSE_RPC",
    "-DUSE_DISTRIBUTED",
    "-DUSE_TENSORPIPE",
    "-DUSE_C10D_GLOO",
    "-DUSE_C10D_NCCL",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-Drecstore_torch_ops_EXPORTS",
]
# Optional: GPU cache.  When CMake built with RECSTORE_ENABLE_GPU_CACHE ON
# (the default in this repo), we must define it too so the op_torch.cc
# GPU code paths are compiled in.  Set RECSTORE_NO_GPU_CACHE=1 to drop it.
if os.environ.get("RECSTORE_NO_GPU_CACHE", "") not in ("1", "true", "TRUE"):
    EXT_COMPILE_DEFS.append("-DRECSTORE_ENABLE_GPU_CACHE")


# Include dirs: project source + third-party headers + build (for
# recstore_config.h generated by CMake).  All paths must be absolute -
# ninja may run from a build subdir, breaking relative -I flags.
EXT_INCLUDE_DIRS = [
    os.path.join(REPO_ROOT, "src"),
    REPO_ROOT,                                            # for #include "build/..."
    BUILD_DIR,                                            # recstore_config.h
    os.path.join(THIRD_PARTY, "folly"),
    os.path.join(THIRD_PARTY, "folly", "_build"),
    os.path.join(THIRD_PARTY, "fmt", "include"),
    os.path.join(THIRD_PARTY, "json", "include"),
    os.path.join(THIRD_PARTY, "brpc-install", "include"),
    os.path.join(THIRD_PARTY, "grpc-install", "include"),
    os.path.join(THIRD_PARTY, "cpptrace-install", "include"),
    os.path.join(THIRD_PARTY, "deps", "usr", "local", "include"),
    os.path.join(THIRD_PARTY, "folly", "folly-install-fPIC", "usr", "local", "include"),
    os.path.join(THIRD_PARTY, "glog", "glog-install-fPIC", "usr", "local", "include"),
    os.path.join(THIRD_PARTY, "liburing", "src", "include"),
    os.path.join(THIRD_PARTY, "libtorch", "libtorch", "include"),
    os.path.join(THIRD_PARTY, "libtorch", "libtorch", "include", "torch", "csrc", "api", "include"),
    "/usr/local/cuda/include",
    "/usr/include/boost169",
    "/tmp/openssl_root/include",
    "/usr/include/libdwarf",
    # Generated protobuf headers (CMake emits these into build/src/ps/proto).
    os.path.join(BUILD_DIR, "src", "ps", "proto"),
    os.path.join(REPO_ROOT, "src", "ps", "proto"),
    os.path.join(REPO_ROOT, "src", "ps", "grpc"),
]


# Library search paths (mirrors -L flags in CMake link.txt).
EXT_LIBRARY_DIRS = [
    BUILD_LIB,
    os.path.join(THIRD_PARTY, "folly", "folly-install-fPIC", "usr", "local", "lib"),
    os.path.join(THIRD_PARTY, "libtorch", "libtorch", "lib"),
    os.path.join(THIRD_PARTY, "deps", "usr", "local", "lib"),
    os.path.join(THIRD_PARTY, "glog", "glog-install-fPIC", "usr", "local", "lib64"),
    os.path.join(THIRD_PARTY, "grpc-install", "lib"),
    os.path.join(THIRD_PARTY, "grpc-install", "lib64"),
    os.path.join(THIRD_PARTY, "cpptrace-install", "lib64"),
    os.path.join(THIRD_PARTY, "brpc-install", "lib"),
    os.path.join(THIRD_PARTY, "protobuf-install", "lib64"),
    os.path.join(THIRD_PARTY, "liburing", "src"),
    os.path.join(BUILD_DIR, "jemalloc-install", "lib"),
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/lib64",
    "/lib64",
    "/tmp/openssl_root/lib",
]


def _a(*parts):
    """Build an absolute .a path inside the CMake build/lib dir."""
    return os.path.join(BUILD_LIB, *parts)


def _so(*parts):
    """Build an absolute .so path inside the CMake build/lib dir."""
    return os.path.join(BUILD_LIB, *parts)


def _tp_a(*parts):
    """Build an absolute .a path inside third_party/."""
    return os.path.join(THIRD_PARTY, *parts)


def _tp_so(*parts):
    """Build an absolute .so path inside third_party/."""
    return os.path.join(THIRD_PARTY, *parts)


# ---------------------------------------------------------------------------
# Link line (extracted from
#   build/src/framework/pytorch/CMakeFiles/recstore_torch_ops.dir/link.txt)
# Order matters for static archives; we wrap them in --start-group/--end-group
# to defuse circular dependencies (absl <-> grpc, folly <-> glog, etc.).
# ---------------------------------------------------------------------------
_LINK_STATICS = [
    # RecStore's own static libraries (from build/lib/).
    _a("libwq_ps_client.a"),
    _a("libgrpc_ps_client.a"),
    _a("libdist_grpc_ps_client.a"),
    _a("libdist_brpc_ps_client.a"),
    _a("libps_common.a"),
    _a("libps_proto.a"),
    _a("libps_base.a"),
    _a("libbase.a"),
    _a("libbase_memory.a"),
    _a("libfmt.a"),
    _a("libextendible_hash.a"),
    _a("libpet_kv.a"),
    _a("libkv_engine.a"),
    _a("libsparse_tensor.a"),
    _a("libjemalloc_compat.a"),
    _a("libiouring_backend.a"),
    _a("libframework_gpu_cache.a"),
    # boost_regex_cxx11 is compiled from source (see EXT_SOURCES) so we do
    # not link the pre-built libboost_regex_cxx11.a here.
    # gRPC / protobuf / absl (vendored under third_party/grpc-install).
    _tp_a("grpc-install", "lib64", "libprotobuf.a"),
    _tp_a("grpc-install", "lib64", "libprotobuf-lite.a"),
    _tp_a("grpc-install", "lib", "libgrpc++.a"),
    _tp_a("grpc-install", "lib", "libgrpc.a"),
    _tp_a("grpc-install", "lib", "libgrpc++_reflection.a"),
    _tp_a("grpc-install", "lib", "libgpr.a"),
    _tp_a("grpc-install", "lib", "libaddress_sorting.a"),
    _tp_a("grpc-install", "lib", "libre2.a"),
    _tp_a("grpc-install", "lib", "libcares.a"),
    _tp_a("grpc-install", "lib", "libssl.a"),
    _tp_a("grpc-install", "lib", "libcrypto.a"),
    _tp_a("grpc-install", "lib", "libupb_base_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_json_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_textformat_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_lex_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_reflection_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_mini_descriptor_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_wire_lib.a"),
    _tp_a("grpc-install", "lib", "libutf8_range_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_message_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_mini_table_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_hash_lib.a"),
    _tp_a("grpc-install", "lib", "libupb_mem_lib.a"),
]
# All absl static libs (there are ~60 of them).  Glob rather than hardcode so
# new absl versions just work.
_LINK_STATICS += sorted(
    glob.glob(os.path.join(THIRD_PARTY, "grpc-install", "lib64", "libabsl_*.a"))
)
# Folly + cityhash (static).
_LINK_STATICS += [
    _tp_a("folly", "folly-install-fPIC", "usr", "local", "lib", "libfolly.a"),
    _tp_a("deps", "usr", "local", "lib", "libcityhash.a"),
]
# cpptrace (static, for backtraces).
_LINK_STATICS += [
    _tp_a("cpptrace-install", "lib64", "libcpptrace.a"),
]
# liburing (static).
_LINK_STATICS += [
    _tp_a("liburing", "src", "liburing.a"),
]
# protobuf from the standalone protobuf-install tree (used by some brpc paths).
_LINK_STATICS += [
    _tp_a("protobuf-install", "lib64", "libprotobuf.a"),
]

_LINK_SHARED = [
    # RecStore shared libs - ONLY the client-side ones that CMake's
    # lib_recstore_ops.so links against.  liblocal_shm_common.so and
    # liblocal_shm_ps_server_runtime.so are NOT direct deps (the runtime
    # pulls them in transitively; listing them here would force the
    # dynamic loader to dlopen them at import time and trigger
    # "cannot allocate memory in static TLS block" errors).
    _so("libbrpc_ps_client.so"),
    _so("libdist_brpc_ps_client.so"),
    _so("liblocal_shm_ps_client.so"),
    # GPU cache shared lib: the static ``libframework_gpu_cache.a`` only
    # references the constructor symbols (U, undefined); the actual
    # definitions live in ``libgpu_cache.so``.  CMake's link line pulls
    # this .so in too (and --as-needed later drops it from NEEDED once
    # the symbols are absorbed).  We need it for symbols like
    # ``gpu_cache::gpu_cache<...>::~gpu_cache``.
    _so("libgpu_cache.so"),
    # libtorch - only libtorch.so + libc10.so + libc10_cuda.so are direct
    # NEEDED entries of the CMake-built .so; libtorch_cpu / libtorch_cuda
    # / libtorch_python are pulled in transitively by libtorch.so.
    _tp_so("libtorch", "libtorch", "lib", "libtorch.so"),
    _tp_so("libtorch", "libtorch", "lib", "libc10.so"),
    _tp_so("libtorch", "libtorch", "lib", "libc10_cuda.so"),
    # glog / gflags.
    _tp_so("glog", "glog-install-fPIC", "usr", "local", "lib64", "libglog.so.0.8.0"),
    _tp_so("deps", "usr", "local", "lib", "libgflags.so.2.3.0"),
    # CUDA runtime.
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/local/cuda/lib64/libcudart.so",
    "/usr/local/cuda/lib64/libnvToolsExt.so",
    # brpc (static .a - the .so variant is not in the CMake link line).
    _tp_a("brpc-install", "lib", "libbrpc.a"),
    # System shared libs.
    "/usr/lib64/libevent.so",
    "/usr/lib64/libz.so",
    "/usr/lib64/libbz2.so",
    "/usr/lib64/liblzma.so",
    "/usr/lib64/liblz4.so",
    "/usr/lib64/libzstd.so",
    "/usr/lib64/libsnappy.so",
    "/usr/lib64/libdwarf.so",
    "/usr/lib64/libiberty.a",
    "/usr/lib64/libaio.so",
    "/usr/lib64/libunwind.so",
    "/usr/lib64/libboost_regex-mt.so",
    "/usr/lib64/libboost_program_options-mt.so",
    "/usr/lib64/libboost_context-mt.so",
    "/usr/lib64/libboost_filesystem-mt.so",
    "/usr/lib64/libssl.so",
    "/usr/lib64/libcrypto.so",
    "/tmp/openssl_root/lib/libssl.so",
    "/tmp/openssl_root/lib/libcrypto.so",
]

# -l style flags for system libs without absolute paths.
_LINK_L_FLAGS = [
    "-libverbs",      # RDMA verbs (libibverbs.so)
    "-lleveldb",      # leveldb (system)
    "-lpthread",
    "-ldl",
    "-lm",
    "-lrt",
    "-lcudadevrt",
    "-lcudart_static",
]

# rpath: search $ORIGIN/../native (vendored .so files) and the build dirs.
# Using $ORIGIN keeps the wheel relocatable - no LD_LIBRARY_PATH needed.
# NOTE: rpath paths are COLON-separated (Unix convention), not comma.
# A single -Wl,-rpath,A,B,C is parsed by ld as "-rpath A" + "B" + "C" as
# separate linker inputs, which is wrong (and breaks the build by passing
# a directory as if it were an input file).
_LINK_RPATH = ":".join([
    "$ORIGIN/../native",
    "$ORIGIN",  # so _recstore_ops.so can find sibling .so files in same dir
    BUILD_LIB,
    os.path.join(THIRD_PARTY, "libtorch", "libtorch", "lib"),
    os.path.join(THIRD_PARTY, "glog", "glog-install-fPIC", "usr", "local", "lib64"),
    os.path.join(THIRD_PARTY, "deps", "usr", "local", "lib"),
    os.path.join(THIRD_PARTY, "folly", "folly-install-fPIC", "usr", "local", "lib"),
    os.path.join(THIRD_PARTY, "grpc-install", "lib"),
    os.path.join(THIRD_PARTY, "grpc-install", "lib64"),
    "/usr/local/cuda/lib64",
])


def _build_extension():
    # Wrap static archives in --start-group / --end-group so the linker can
    # resolve circular dependencies (absl <-> grpc, folly <-> glog) without
    # us having to order them perfectly.
    extra_link_args = [
        "-Wl,--no-as-needed",
        "-Wl,--start-group",
        *_LINK_STATICS,
        *_LINK_SHARED,
        "-Wl,--end-group",
        *_LINK_L_FLAGS,
        "-pthread",
        "-fopenmp",
        f"-Wl,-rpath,{_LINK_RPATH}",
    ]

    return CppExtension(
        name="recstore._recstore_ops",
        # Sources are injected in RecStoreBuildExt.run() as ABSOLUTE paths.
        # Passing them here (even relative) makes setuptools 82 resolve them
        # to absolute during egg_info, triggering "setup script specifies an
        # absolute path" error.  Empty list passes the metadata check.
        sources=[],
        include_dirs=EXT_INCLUDE_DIRS,
        library_dirs=EXT_LIBRARY_DIRS,
        extra_compile_args={
            "cxx": [
                "-std=c++17",
                "-O3",
                "-g",
                "-fPIC",
                "-faligned-new",
                "-fopenmp",
                "-Wall",
                "-Wno-unused-function",
                "-Wno-sign-compare",
                "-Wno-unused-parameter",
                "-Wno-unused-variable",
                "-Wno-attributes",
                "-Wno-parentheses",
                "-Wno-unused-but-set-variable",
                *EXT_COMPILE_DEFS,
            ],
        },
        extra_link_args=extra_link_args,
    )


# ---------------------------------------------------------------------------
# Vendored native libs (copy libtorch / glog .so into recstore/native/ so the
# wheel is self-contained without LD_LIBRARY_PATH).  Mirrors quantarec's
# _prepare_vendored_native_libs_for_wheel.
# ---------------------------------------------------------------------------
def _prepare_vendored_native_libs_for_wheel() -> None:
    """Copy runtime-shared libs into recstore/native/ for wheel distribution.

    Only run during wheel build (sdist/bdist_wheel).  Development
    ``pip install -e .`` skips this because rpath already points at
    ``BUILD_LIB`` and the third_party install dirs.
    """
    native_dir = os.path.join(BASEDIR, "pytorch", "recstore", "native")
    os.makedirs(native_dir, exist_ok=True)

    # libtorch runtime libs.
    torch_lib_dir = os.path.join(THIRD_PARTY, "libtorch", "libtorch", "lib")
    for name in ("libtorch.so", "libc10.so", "libc10_cuda.so",
                 "libtorch_cpu.so", "libtorch_cuda.so"):
        src = os.path.join(torch_lib_dir, name)
        if os.path.isfile(src):
            dst = os.path.join(native_dir, name)
            if not os.path.lexists(dst):
                shutil.copy2(src, dst)
                print(f"[setup] vendored native lib: {src} -> {dst}")


# ---------------------------------------------------------------------------
# Build-time version_info.py generation
# ---------------------------------------------------------------------------
_VERSION_INFO_TEMPLATE = '''"""Auto-generated by setup.py. Do not edit.

Build metadata for the recstore native extension.  Inspect from Python via::

    import recstore
    print(recstore.__build_info__)
"""
import os

BUILD_INFO = {{
    "version": "{version}",
    "git": {{
        "branch": "{git_branch}",
        "commit_hash": "{git_hash}",
        "commit_hash_full": "{git_hash_full}",
        "commit_time": "{git_time}",
        "commit_author": "{git_author}",
        "commit_message": "{git_message}",
        "tag": "{git_tag}",
    }},
    "build": {{
        "build_time": "{build_time}",
        "build_timestamp": {build_timestamp},
        "python_version": "{python_version}",
        "platform": "{platform}",
        "hostname": "{hostname}",
        "build_user": "{build_user}",
        "cuda_version": "{cuda_version}",
        "cxx_abi": 1,
    }},
}}


def get_version_info():
    return BUILD_INFO


def is_internal_enabled():
    return 0
'''


class BuildPyCommand(build_py):
    """Render version_info.py from the template before copying .py files."""

    def run(self):
        self._render_version_info()
        super().run()

    def _render_version_info(self) -> None:
        ctx = {
            "version": get_version(),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_hash": _git_show("%h"),
            "git_hash_full": _git_show("%H"),
            "git_time": _git_show("%ci"),
            "git_author": _git_show("%an"),
            "git_message": _git_show("%s").replace('"', '\\"'),
            "git_tag": _git(["describe", "--tags", "--exact-match"], default=""),
            "build_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_timestamp": int(time.time()),
            "python_version": "{}.{}.{}".format(*sys.version_info[:3]),
            "platform": sys.platform,
            "hostname": socket.gethostname(),
            "build_user": os.environ.get("USER", "unknown"),
            "cuda_version": os.environ.get("CUDA_VERSION", ""),
        }
        content = _VERSION_INFO_TEMPLATE.format(**ctx)

        # Write into both the source tree (for --inplace dev builds) and
        # build/lib.<tag>/recstore/ (for wheel builds).
        src_pkg = os.path.join(BASEDIR, "pytorch", "recstore")
        out_src = os.path.join(src_pkg, "version_info.py")
        with open(out_src, "w") as f:
            f.write(content)
        print(f"[setup] wrote version_info.py -> {out_src}")

        # Also write into build/lib.*/recstore/version_info.py so bdist_wheel
        # picks it up even if super().run() copies from src before this hook.
        build_lib = self.build_lib
        if build_lib:
            dst_dir = os.path.join(build_lib, "recstore")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, "version_info.py")
            with open(dst, "w") as f:
                f.write(content)
            print(f"[setup] wrote version_info.py -> {dst}")


# Custom build_ext that:
# 1. Injects source files as ABSOLUTE paths at build time (setuptools 82
#    blocks absolute paths in setup() args during egg_info; we pass
#    sources=[] to CppExtension and set them here, after the metadata
#    check).  Absolute paths are also needed so pip's temp build dir
#    doesn't break ninja (which flattens absolute paths under the temp
#    dir but lets relative ``../../`` paths escape it).
# 2. Copies vendored .so files for wheel builds.
class RecStoreBuildExt(BuildExtension):
    """BuildExtension subclass with source injection and vendoring."""

    def run(self):
        # Inject ABSOLUTE source paths (bypasses setuptools 82's
        # no-absolute-path check which only runs during egg_info, not
        # during build_ext).
        for ext in self.extensions:
            if not ext.sources:
                ext.sources = [
                    os.path.abspath(os.path.join(BASEDIR, s))
                    if not os.path.isabs(s)
                    else s
                    for s in EXT_SOURCES
                ]
        # Only prepare vendored libs when building a wheel (not in-place dev).
        if not self.inplace:
            _prepare_vendored_native_libs_for_wheel()
        super().run()


# ---------------------------------------------------------------------------
# Final setup() call - guarded so setuptools can ``import setup`` to read
# metadata (e.g. ``get_version`` for ``[tool.setuptools.dynamic]``) without
# triggering the ``setup()`` side effect.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _prepare_vendored_native_libs_for_wheel()

    setup(
        name="recstore",
        version=get_version(),
        description="Distributed parameter server and embedding store for recommendation models",
        packages=find_packages(where="pytorch"),
        package_dir={"": "pytorch"},
        package_data={
            "recstore": [
                "native/*.so",
                "native/*.a",
            ],
        },
        include_package_data=False,  # True triggers filesystem walk outside setup.py dir
        ext_modules=[_build_extension()],
        cmdclass={
            "build_py": BuildPyCommand,
            "build_ext": RecStoreBuildExt.with_options(
                no_python_abi_suffix=True
            ),
        },
        python_requires=">=3.8",
        install_requires=[
            "torch",
            "numpy",
        ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        zip_safe=False,
    )
