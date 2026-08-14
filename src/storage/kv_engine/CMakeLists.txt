find_package(PkgConfig QUIET)

option(USE_PMKV "enable PMKV" OFF)
if(USE_PMKV MATCHES "ON")
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBPMEMKV REQUIRED libpmemkv)
  else()
    message(FATAL_ERROR "pkg-config not found")
  endif()

  include_directories(${LIBPMEMKV_INCLUDE_DIRS})
  link_directories(${LIBPMEMKV_LIBRARY_DIRS})
endif()

# KV engine factory
add_library(kv_engine
  kv_engine_register.cc
)
target_link_libraries(kv_engine PUBLIC base)
if(USE_PMKV MATCHES "ON")
  target_link_libraries(kv_engine PUBLIC ${LIBPMEMKV_LIBRARIES})
endif()

target_link_libraries(
  kv_engine
  PUBLIC pet_kv index io_backend
  # spdk_wrapper
)
  # INTERFACE pet_kv dash_hash PMHashPCLHT PMHashLevel PMHashClevel PMHashCCEHVM)

if(TARGET fasterkv_backend)
  target_sources(
    kv_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/src/storage/external/fasterkv/fasterkv_engine.cc)
  target_link_libraries(kv_engine PUBLIC fasterkv_backend)
  target_compile_definitions(kv_engine PRIVATE RECSTORE_ENABLE_FASTERKV_ENGINE)
endif()

if(TARGET detable_plugin_api)
  target_sources(
    kv_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/src/storage/external/detable/detable_engine.cc)
  target_link_libraries(kv_engine PUBLIC detable_plugin_api)
  target_compile_definitions(kv_engine PRIVATE RECSTORE_ENABLE_DETABLE_ENGINE)
endif()

if(TARGET hps_native_support)
  target_sources(
    kv_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/src/storage/external/hps/hps_engine.cc)
  target_link_libraries(kv_engine PUBLIC hps_native_support)
  target_compile_definitions(kv_engine PRIVATE RECSTORE_ENABLE_HPS_ENGINE)
endif()
# KV engine factory done

add_executable(benchmark_kv_engine benchmark_kv_engine.cc)
target_link_libraries(
  benchmark_kv_engine
  PUBLIC
    pthread
    base
    -Wl,--whole-archive
    kv_engine
    -Wl,--no-whole-archive
    ps_common)
target_link_libraries(benchmark_kv_engine PUBLIC OpenMP::OpenMP_CXX)
if(USE_GPERF_PROFILING)
  target_compile_options(benchmark_kv_engine PRIVATE -fno-omit-frame-pointer)
  target_compile_definitions(benchmark_kv_engine PRIVATE RECSTORE_USE_GPERF_PROFILING)
  target_link_libraries(benchmark_kv_engine PRIVATE profiler)
endif()
