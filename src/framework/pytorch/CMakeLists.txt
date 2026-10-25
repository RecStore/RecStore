find_package(Torch REQUIRED)

# recstore_ops is split by op family; op_torch.cc is the single TORCH_LIBRARY
# registration unit and calls the Register*Ops helpers defined in each family.
add_library(recstore_torch_ops SHARED
    op_torch.cc
    op_base.cc
    op_flat.cc
    op_prefetch.cc
    op_bagpipe.cc
    op_profile.cc
    ../op.cc
    ../common/local_shm_kv_client_op.cc
    ../common/hierkv_local_runtime.cc
    ../common/op_runtime_support.cc
    ../common/local_shm_op_component.cc
    ../common/ps_client_config_adapter.cc
    ../../ps/client_factory.cc
    # rc_options.cc is deliberately not compiled here: it also lives in
    # libwq_ps_client.so (which this target links), and a second copy makes
    # gflags abort with a duplicate-registration error at import time.
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
    boost_regex_cxx11_inst.cc
)

# CUDA-runtime host helpers live in a .cu so the family .cc files stay free of
# CUDA includes and __has_include guards. CPU-only builds use a stub instead.
if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(recstore_torch_ops PRIVATE op_cuda_runtime.cu)
    target_compile_options(recstore_torch_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    # The top-level set(CUDA_SEPARABLE_COMPILATION ON) is cleared by the
    # deprecated FindCUDA module's option() (CMP0077), which leaves the target
    # with -rdc=true but no device link step and therefore an unloadable .so.
    # Set it on the target so every configure, cache state included, links.
    set_target_properties(recstore_torch_ops PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
else()
    target_sources(recstore_torch_ops PRIVATE op_cuda_runtime_cpu.cc)
endif()

target_compile_definitions(recstore_torch_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_libraries(recstore_torch_ops PRIVATE boost_regex_cxx11)

if(USE_FAKE_KVCLIENT)
    target_compile_definitions(recstore_torch_ops PRIVATE USE_FAKE_KVCLIENT)
endif()

target_include_directories(recstore_torch_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/ps/grpc
)

option(USE_PERF_REPORT "Enable performance reporting in Debug mode" OFF)

if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_PERF_REPORT)
    target_compile_definitions(recstore_torch_ops PRIVATE ENABLE_PERF_REPORT)
    target_link_libraries(recstore_torch_ops PRIVATE report_client)
endif()

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
target_link_libraries(recstore_torch_ops PRIVATE
    wq_ps_client
    grpc_ps_client
    dist_grpc_ps_client
    brpc_ps_client
    dist_brpc_ps_client
    local_shm_ps_client
    protobuf::libprotobuf
    gRPC::grpc++
    glog::glog
    Folly::folly
    ${TORCH_LIBRARIES}
)

if(TARGET framework_gpu_cache)
    target_compile_definitions(recstore_torch_ops PRIVATE RECSTORE_ENABLE_GPU_CACHE)
    target_link_libraries(recstore_torch_ops PRIVATE framework_gpu_cache)
endif()

target_link_options(recstore_torch_ops PRIVATE -Wl,--no-as-needed)

# Set the required C++ standard.
set_target_properties(recstore_torch_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Set the output name of the library.
set_target_properties(recstore_torch_ops PROPERTIES
    OUTPUT_NAME "_recstore_ops"
)

# ---------------------------------------------------------------------------
# Python wheel packaging (scikit-build-core)
# ---------------------------------------------------------------------------
# scikit-build-core only packs install() rules tagged COMPONENT python, so the
# install() rules contributed by third_party/ subprojects (fmt, glog,
# googletest, json, ycsb) can never leak into the distribution.
#
# The extension lands in recstore/lib/, which is the first location the
# recstore/__init__.py loader looks at, and the wheel rpath stays $ORIGIN-based
# so no build-host paths are baked into the artifact.
install(TARGETS recstore_torch_ops
        LIBRARY DESTINATION recstore/lib
        COMPONENT python)

set_target_properties(recstore_torch_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../native"
    # Drop toolchain-injected rpath entries (e.g. a CMAKE_BUILD_RPATH provided by
    # the container environment) so the installed artifact only carries $ORIGIN
    # paths.
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)

# Everything the loader needs at load time that is NOT on the container's
# default library path has to be vendored next to the extension, otherwise a
# wheel installed outside the build tree fails to import.
get_filename_component(_recstore_wheel_dep_prefix
    "${gflags_DIR}/../../.." ABSOLUTE)
set(RECSTORE_WHEEL_DEP_PREFIX "${_recstore_wheel_dep_prefix}" CACHE PATH
    "Install prefix searched for runtime libraries vendored into the wheel")
set(RECSTORE_WHEEL_EXTRA_DEP_DIRS "" CACHE STRING
    "Extra directories to search for runtime libraries vendored into the wheel")

set(_recstore_wheel_dep_dirs
    "${CMAKE_BINARY_DIR}/lib"
    "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/lib"
    "${PROJECT_SOURCE_DIR}/third_party/glog/glog-install-fPIC/usr/local/lib64"
    "${PROJECT_SOURCE_DIR}/third_party/protobuf-install/lib64"
    "${RECSTORE_WHEEL_DEP_PREFIX}/lib"
    "${RECSTORE_WHEEL_DEP_PREFIX}/lib64"
    ${RECSTORE_WHEEL_EXTRA_DEP_DIRS}
)

# Shared libraries the extension links against at runtime.  Names are sonames;
# the GPU-cache-only entry is added conditionally so a CPU build does not emit
# a bogus "missing dependency" warning.
#
# libtorch.so / libc10.so / libtorch_cpu.so / libtorch_cuda.so / libc10_cuda.so
# are deliberately NOT vendored: the extension is compiled against the libtorch
# that ships inside the torch wheel (third_party/libtorch is an extraction of
# it -- identical build IDs), and ``import recstore`` imports torch before
# loading the extension, so the already-mapped torch/lib copies satisfy the
# DT_NEEDED entries.  Vendoring them again would add ~1.3 GB of duplicate
# binaries.  torch is therefore a hard runtime dependency, pinned in
# pyproject.toml.
#
# liblocal_shm_ps_server_runtime.so is also NOT vendored: it is the in-process
# server side of the local_shm backend, linked only by the standalone
# local_shm_ps_server executable and by C++ tests.  The extension needs just
# liblocal_shm_ps_client.so + liblocal_shm_common.so (see its DT_NEEDED), and
# nothing dlopen()s the runtime, so shipping it only adds ~35 MB.
set(_recstore_wheel_sonames
    libbrpc_ps_client.so
    libdist_brpc_ps_client.so
    liblocal_shm_ps_client.so
    liblocal_shm_common.so
    libglog.so.0
    libgflags.so.2.2
    libprotobuf.so.32
    libboost_context.so.1.75.0
    libboost_filesystem.so.1.75.0
    libboost_program_options.so.1.75.0
    libboost_regex.so.1.75.0
    libboost_system.so.1.75.0
    libboost_thread.so.1.75.0
    libssl.so.1.1
    libcrypto.so.1.1
    libtbb.so.12
)
if(TARGET framework_gpu_cache)
  list(APPEND _recstore_wheel_sonames libgpu_cache.so)
endif()

set(RECSTORE_WHEEL_NATIVE_DEPS "")
foreach(_soname IN LISTS _recstore_wheel_sonames)
  unset(_dep)
  foreach(_dir IN LISTS _recstore_wheel_dep_dirs)
    if(EXISTS "${_dir}/${_soname}")
      set(_dep "${_dir}/${_soname}")
      break()
    endif()
  endforeach()
  if(_dep)
    # Resolve symlink chains (e.g. libglog.so.0 -> libglog.so.0.5.0) and install
    # the real file under the soname the loader asks for.
    file(REAL_PATH "${_dep}" _dep_real)
    list(APPEND RECSTORE_WHEEL_NATIVE_DEPS "${_soname}|${_dep_real}")
  else()
    message(WARNING
      "recstore wheel: runtime dependency '${_soname}' was not found in "
      "${_recstore_wheel_dep_dirs}; it will not be vendored, so the wheel may "
      "fail to import outside this container.")
  endif()
endforeach()

foreach(_dep IN LISTS RECSTORE_WHEEL_NATIVE_DEPS)
  string(REPLACE "|" ";" _dep_pair "${_dep}")
  list(GET _dep_pair 0 _dep_soname)
  list(GET _dep_pair 1 _dep_real)
  install(FILES "${_dep_real}"
          DESTINATION recstore/native
          RENAME "${_dep_soname}"
          COMPONENT python)
endforeach()

# ---------------------------------------------------------------------------
# version_info.py (build metadata baked into the package)
# ---------------------------------------------------------------------------
# Replaces the former setup.py build_py hook: the same template is rendered
# into the build tree (installed into the wheel) and into the source package
# (for cmake --build + PYTHONPATH development runs).
find_package(Git QUIET)

function(_recstore_git out_var)
  if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE "git")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ${ARGN}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE _out
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
  )
  if(NOT _rc EQUAL 0)
    set(_out "")
  endif()
  # Keep the values safe to inline into a Python string literal.
  string(REPLACE "\\" "\\\\" _out "${_out}")
  string(REPLACE "\"" "\\\"" _out "${_out}")
  string(REPLACE "\n" "\\n" _out "${_out}")
  set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

_recstore_git(RECSTORE_GIT_BRANCH rev-parse --abbrev-ref HEAD)
_recstore_git(RECSTORE_GIT_HASH show -s --format=%h HEAD)
_recstore_git(RECSTORE_GIT_HASH_FULL show -s --format=%H HEAD)
_recstore_git(RECSTORE_GIT_TIME show -s --format=%ci HEAD)
_recstore_git(RECSTORE_GIT_AUTHOR show -s --format=%an HEAD)
_recstore_git(RECSTORE_GIT_MESSAGE show -s --format=%s HEAD)
_recstore_git(RECSTORE_GIT_TAG describe --tags --exact-match)

string(TIMESTAMP RECSTORE_BUILD_TIME "%Y-%m-%dT%H:%M:%SZ" UTC)
string(TIMESTAMP RECSTORE_BUILD_TIMESTAMP "%s" UTC)

execute_process(COMMAND hostname
                OUTPUT_VARIABLE RECSTORE_HOSTNAME
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

set(RECSTORE_BUILD_USER "$ENV{USER}")
set(RECSTORE_CUDA_VERSION "$ENV{CUDA_VERSION}")
set(RECSTORE_PLATFORM "${CMAKE_SYSTEM_NAME}")
if(RECSTORE_PLATFORM STREQUAL "Linux")
  set(RECSTORE_PLATFORM "linux")
elseif(RECSTORE_PLATFORM STREQUAL "Darwin")
  set(RECSTORE_PLATFORM "darwin")
elseif(RECSTORE_PLATFORM STREQUAL "Windows")
  set(RECSTORE_PLATFORM "win32")
endif()
if(Python_VERSION)
  set(RECSTORE_PYTHON_VERSION "${Python_VERSION}")
else()
  set(RECSTORE_PYTHON_VERSION "unknown")
endif()

set(_recstore_version_info_template
    "${PROJECT_SOURCE_DIR}/src/python/version_info.py.in")
configure_file("${_recstore_version_info_template}"
               "${CMAKE_BINARY_DIR}/version_info.py" @ONLY)
configure_file("${_recstore_version_info_template}"
               "${PROJECT_SOURCE_DIR}/src/python/pytorch/recstore/version_info.py"
               @ONLY)

install(FILES "${CMAKE_BINARY_DIR}/version_info.py"
        DESTINATION recstore
        COMPONENT python)
