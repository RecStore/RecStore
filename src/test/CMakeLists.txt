if(NOT TARGET Folly::folly)
    find_package(folly REQUIRED)
endif()
find_package(ZLIB REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(add_python_unittest _name)
    add_test(
        NAME ${_name}
        COMMAND ${Python3_EXECUTABLE} -m unittest ${ARGN}
    )
    set_tests_properties(
        ${_name}
        PROPERTIES
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endfunction()

add_subdirectory(small_tests)
add_subdirectory(server_mgr)

# add_executable(test_f14_slab
#     test_f14_slab.cpp 
# )

# target_link_libraries(test_f14_slab PUBLIC
#     ${FOLLY_LIBRARIES}
#     mimalloc-static
# )

# target_compile_options(test_f14_slab PUBLIC -msse -msse2 -msse3 -msse4)

# brpc client libraries, only available when bRPC backend is enabled
if(NOT RECSTORE_DISABLE_BRPC)
    set(BRPC_TEST_LIBS brpc_ps_client dist_brpc_ps_client)
endif()

add_executable(test_hash_map
    test_hash_map.cpp 
)

target_compile_options(test_hash_map PUBLIC -fopenmp)

target_link_libraries(test_hash_map PUBLIC -fopenmp Folly::folly)


add_executable(coroutine 
    coroutine.cpp
)

target_link_libraries(coroutine PUBLIC
    ${BOOST_LIBRARIES}
    boost_context
    boost_coroutine
)

add_executable(test_extendible_hash test_extendible_hash.cpp)
target_link_libraries(test_extendible_hash PUBLIC
    extendible_hash 
    base 
    gtest 
    gtest_main 
    pthread
    base_memory
)
add_test(NAME test_extendible_hash COMMAND test_extendible_hash)

add_executable(test_io_backend test_io_backend.cpp)
target_link_libraries(test_io_backend PUBLIC
    io_backend
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_io_backend COMMAND test_io_backend)

add_executable(test_ssd_allocators test_ssd_allocators.cpp)
target_link_libraries(test_ssd_allocators PUBLIC
    base
    gtest
    gtest_main
    boost_context
    boost_coroutine
    pthread
)
add_test(NAME test_ssd_allocators COMMAND test_ssd_allocators)

add_executable(test_r2_allocator test_r2_allocator.cpp)
target_link_libraries(test_r2_allocator PUBLIC
    base_memory
    gtest
    gtest_main
    pthread
)
add_test(NAME test_r2_allocator COMMAND test_r2_allocator)

add_executable(test_io_uring_test_utils test_io_uring_test_utils.cpp)
target_link_libraries(test_io_uring_test_utils PUBLIC
    gtest
    gtest_main
    uring
    pthread
)
add_test(NAME test_io_uring_test_utils COMMAND test_io_uring_test_utils)

add_executable(test_config test_config.cpp)
target_link_libraries(test_config PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_config COMMAND test_config)

add_executable(test_shm_file test_shm_file.cpp)
target_link_libraries(test_shm_file PUBLIC
    base
    base_memory
    gtest
    gtest_main
    pthread
)
add_test(NAME test_shm_file COMMAND test_shm_file)

add_executable(test_concurrent_slab_memory_pool test_concurrent_slab_memory_pool.cpp)
target_link_libraries(test_concurrent_slab_memory_pool PUBLIC
    base
    base_memory
    gtest
    gtest_main
    pthread
)
add_test(NAME test_concurrent_slab_memory_pool COMMAND test_concurrent_slab_memory_pool)

set(RECSTORE_KVENGINE_TEST_DEFS)
if(TARGET fasterkv_backend)
    list(APPEND RECSTORE_KVENGINE_TEST_DEFS RECSTORE_TEST_ENABLE_FASTERKV_ENGINE)
endif()
if(TARGET hps_backend_support)
    list(APPEND RECSTORE_KVENGINE_TEST_DEFS RECSTORE_TEST_ENABLE_HPS_ENGINE)
endif()

add_executable(test_kvengine test_kvengine.cpp)
target_compile_definitions(test_kvengine PRIVATE ${RECSTORE_KVENGINE_TEST_DEFS})
target_link_libraries(test_kvengine PUBLIC
    extendible_hash
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
    TBB::tbb
)
add_test(NAME test_kvengine COMMAND test_kvengine)

if(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE)
    add_executable(test_hps_recstore_backend
        test_hps_recstore_backend.cpp
    )
    target_link_libraries(test_hps_recstore_backend PUBLIC
        base_memory
        hps_backend_support
        gtest
        gtest_main
        pthread
    )
    add_test(NAME test_hps_recstore_backend COMMAND test_hps_recstore_backend)
else()
    message(STATUS
        "Skipping test_hps_recstore_backend: HugeCTR CUDA headers are unavailable."
    )
endif()

if(TARGET rocksdb-shared)
    add_executable(test_raw_rocksdb_backend
        test_raw_rocksdb_backend.cpp
        ${PROJECT_SOURCE_DIR}/src/storage/external/hps/raw_rocksdb.cc
    )
    target_include_directories(test_raw_rocksdb_backend PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/third_party/rocksdb/include
    )
    target_link_libraries(test_raw_rocksdb_backend PUBLIC
        rocksdb-shared
        gtest
        gtest_main
        pthread
    )
    add_test(NAME test_raw_rocksdb_backend COMMAND test_raw_rocksdb_backend)
else()
    message(STATUS "Skipping test_raw_rocksdb_backend: rocksdb-shared target is unavailable.")
endif()

if(TARGET fasterkv_backend)
    add_executable(test_fasterkv_backend
        test_fasterkv_backend.cpp
    )
    target_include_directories(test_fasterkv_backend PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
    )
    target_link_libraries(test_fasterkv_backend PUBLIC
        fasterkv_backend
        gtest
        gtest_main
        pthread
    )
    add_test(NAME test_fasterkv_backend COMMAND test_fasterkv_backend)
else()
    message(STATUS
        "Skipping test_fasterkv_backend: fasterkv_backend target is unavailable."
    )
endif()

set(RECSTORE_EXTERNAL_KV_ENGINE_TEST_ENABLED OFF)
set(RECSTORE_EXTERNAL_KV_ENGINE_TEST_LIBS
    kv_engine
    gtest
    gtest_main
    pthread
)
set(RECSTORE_EXTERNAL_KV_ENGINE_TEST_DEFS)
if(TARGET fasterkv_backend)
    set(RECSTORE_EXTERNAL_KV_ENGINE_TEST_ENABLED ON)
    list(APPEND RECSTORE_EXTERNAL_KV_ENGINE_TEST_DEFS
        RECSTORE_TEST_ENABLE_FASTERKV_ENGINE)
endif()
if(TARGET hps_backend_support)
    set(RECSTORE_EXTERNAL_KV_ENGINE_TEST_ENABLED ON)
    list(APPEND RECSTORE_EXTERNAL_KV_ENGINE_TEST_DEFS
        RECSTORE_TEST_ENABLE_HPS_ENGINE)
endif()
if(RECSTORE_EXTERNAL_KV_ENGINE_TEST_ENABLED)
    add_executable(test_external_kv_engine test_external_kv_engine.cpp)
    target_compile_definitions(test_external_kv_engine PRIVATE
        ${RECSTORE_EXTERNAL_KV_ENGINE_TEST_DEFS}
    )
    target_link_libraries(test_external_kv_engine PUBLIC
        ${RECSTORE_EXTERNAL_KV_ENGINE_TEST_LIBS}
    )
    add_test(NAME test_external_kv_engine COMMAND test_external_kv_engine)
endif()

add_executable(test_index test_index.cpp)
target_link_libraries(test_index PUBLIC
    extendible_hash 
    base 
    gtest 
    gtest_main 
    pthread
)
add_test(NAME test_index COMMAND test_index)

add_executable(test_ps_client_factory test_ps_client_factory.cpp)
target_sources(test_ps_client_factory PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(test_ps_client_factory PUBLIC
    base
    wq_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    grpc_ps_client
    dist_grpc_ps_client
    ${BRPC_TEST_LIBS}
    local_shm_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    gflags
    glog
    gtest
    gtest_main
    pthread
)
target_link_options(test_ps_client_factory PUBLIC -Wl,--no-as-needed)
add_test(NAME test_ps_client_factory COMMAND test_ps_client_factory)

add_executable(test_rdmaps_client_adapter test_rdmaps_client_adapter.cpp)
target_sources(test_rdmaps_client_adapter PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/allshards_ps_client.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/petps_client.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rc_transport.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rc_options.cc
)
target_link_libraries(test_rdmaps_client_adapter PUBLIC
    base
    wq_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    grpc_ps_client
    dist_grpc_ps_client
    ${BRPC_TEST_LIBS}
    local_shm_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    gflags
    glog
    gtest
    gtest_main
    pthread
)
target_link_options(test_rdmaps_client_adapter PUBLIC -Wl,--no-as-needed)
add_test(NAME test_rdmaps_client_adapter COMMAND test_rdmaps_client_adapter)

add_executable(test_local_shm_protocol test_local_shm_protocol.cpp)
target_link_libraries(test_local_shm_protocol PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_protocol COMMAND test_local_shm_protocol)

add_executable(test_local_shm_region test_local_shm_region.cpp)
target_link_libraries(test_local_shm_region PUBLIC
    base
    local_shm_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_region COMMAND test_local_shm_region)

add_executable(test_local_shm_futex test_local_shm_futex.cpp)
target_link_libraries(test_local_shm_futex PUBLIC
    local_shm_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_futex COMMAND test_local_shm_futex)

add_executable(test_local_shm_queue test_local_shm_queue.cpp)
target_link_libraries(test_local_shm_queue PUBLIC
    local_shm_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_queue COMMAND test_local_shm_queue)

add_executable(test_local_shm_ps_client test_local_shm_ps_client.cpp)
target_link_libraries(test_local_shm_ps_client PUBLIC
    base
    local_shm_ps_client
    local_shm_ps_server_runtime
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_ps_client COMMAND test_local_shm_ps_client)
set_tests_properties(test_local_shm_ps_client PROPERTIES
    TIMEOUT 30
    RUN_SERIAL TRUE
    RESOURCE_LOCK local_shm_tests
)

add_executable(test_op_runtime_support test_op_runtime_support.cpp)
target_sources(test_op_runtime_support PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/op_runtime_support.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(test_op_runtime_support PUBLIC
    base
    wq_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    grpc_ps_client
    dist_grpc_ps_client
    ${BRPC_TEST_LIBS}
    local_shm_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    gflags
    glog
    gtest
    gtest_main
    pthread
)
add_test(NAME test_op_runtime_support COMMAND test_op_runtime_support)

add_executable(test_op test_op.cpp)
target_sources(test_op PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/op.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/local_shm_kv_client_op.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/hierkv_local_runtime.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/local_shm_op_component.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/op_runtime_support.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(test_op PUBLIC
    base
    wq_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    grpc_ps_client
    dist_grpc_ps_client
    ${BRPC_TEST_LIBS}
    local_shm_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    gflags
    glog
    gtest
    gtest_main
    pthread
)
add_test(NAME test_op COMMAND test_op)

add_executable(test_local_shm_op_component test_local_shm_op_component.cpp)
target_sources(test_local_shm_op_component PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/local_shm_op_component.cc
)
target_link_libraries(test_local_shm_op_component PUBLIC
    base
    local_shm_ps_client
    local_shm_ps_server_runtime
    gtest
    gtest_main
    pthread
)
add_test(NAME test_local_shm_op_component COMMAND test_local_shm_op_component)
set_tests_properties(test_local_shm_op_component PROPERTIES
    TIMEOUT 30
    RUN_SERIAL TRUE
    RESOURCE_LOCK local_shm_tests
)

add_python_unittest(
    test_run_local_shm_mixed_benchmark_py
    src/test/scripts/test_run_local_shm_mixed_benchmark.py
)
set_tests_properties(
    test_run_local_shm_mixed_benchmark_py
    PROPERTIES
    LABELS "local_shm;python"
)

add_python_unittest(
    test_run_local_shm_multi_process_benchmark_py
    src/test/scripts/test_run_local_shm_multi_process_benchmark.py
)
set_tests_properties(
    test_run_local_shm_multi_process_benchmark_py
    PROPERTIES
    LABELS "local_shm;python"
)

add_python_unittest(
    test_run_hierkv_recstore_mixed_benchmark_py
    src/test/scripts/test_run_hierkv_recstore_mixed_benchmark.py
)
set_tests_properties(
    test_run_hierkv_recstore_mixed_benchmark_py
    PROPERTIES
    LABELS "hierkv;python"
)

add_python_unittest(
    test_recstore_config_schema_py
    src/test/scripts/test_recstore_config_schema.py
)
set_tests_properties(
    test_recstore_config_schema_py
    PROPERTIES
    LABELS "config;python"
)

add_python_unittest(
    test_hierkv_python_ops_py
    src.python.pytorch.recstore.unittest.test_kvclient_local_fast_path
    src.python.pytorch.recstore.unittest.test_embeddingbag_single_node_distributed
    src.python.pytorch.recstore.unittest.test_sparse_optimizer_single_node_distributed
    src.test.model_zoo.rs_demo.test_recstore_distributed
)
set_tests_properties(
    test_hierkv_python_ops_py
    PROPERTIES
    LABELS "hierkv;python;integration"
)

if(TARGET recstore_torch_ops)
    add_python_unittest(
        test_hierkv_python_ops_with_torch_ops_py
        src.python.pytorch.recstore.unittest.test_hierkv_integration
    )
    set_tests_properties(
        test_hierkv_python_ops_with_torch_ops_py
        PROPERTIES
        LABELS "hierkv;python;integration;torch_ops"
    )
else()
    message(STATUS "Skipping HierKV torch ops Python integration tests because recstore_torch_ops is unavailable")
endif()

add_python_unittest(
    test_hierkv_rs_demo_py
    src.test.model_zoo.rs_demo.test_hybrid_dlrm
    src.test.model_zoo.rs_demo.test_recstore_runner
)
set_tests_properties(
    test_hierkv_rs_demo_py
    PROPERTIES
    LABELS "hierkv;python;rs_demo"
)

set(HUGECTR_TENSOR2_HEADER
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include/tensor2.hpp
)
set(HUGECTR_CONFIG_TEMPLATE
    ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include/config.hpp.in
)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND
   RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE AND
   EXISTS ${HUGECTR_TENSOR2_HEADER} AND
   EXISTS ${HUGECTR_CONFIG_TEMPLATE})
    set(HUGECTR_TEST_GENERATED_INCLUDE_DIR
        ${PROJECT_BINARY_DIR}/third_party/HugeCTR/HugeCTR/include
    )
    file(MAKE_DIRECTORY ${HUGECTR_TEST_GENERATED_INCLUDE_DIR})
    configure_file(
        ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include/config.hpp.in
        ${HUGECTR_TEST_GENERATED_INCLUDE_DIR}/config.hpp
        @ONLY
    )

    add_executable(test_hugectr_emb
        test_hugectr_emb.cpp
        ${PROJECT_SOURCE_DIR}/src/framework/hugectr/op_hugectr.cc
        ${PROJECT_SOURCE_DIR}/src/framework/hugectr/hierkv_backend.cc
        ${PROJECT_SOURCE_DIR}/src/framework/op.cc
        ${PROJECT_SOURCE_DIR}/src/framework/common/local_shm_kv_client_op.cc
        ${PROJECT_SOURCE_DIR}/src/framework/common/hierkv_local_runtime.cc
        ${PROJECT_SOURCE_DIR}/src/framework/common/local_shm_op_component.cc
        ${PROJECT_SOURCE_DIR}/src/framework/common/op_runtime_support.cc
        ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
        ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
        ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
    )
    target_include_directories(test_hugectr_emb PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/ps/grpc
        ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR
        ${PROJECT_SOURCE_DIR}/third_party/HugeCTR/HugeCTR/include
        ${HUGECTR_TEST_GENERATED_INCLUDE_DIR}
        ${RECSTORE_CUDA_INCLUDE_DIRS}
    )
    target_link_libraries(test_hugectr_emb PUBLIC
        base
        wq_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
        grpc_ps_client
        dist_grpc_ps_client
        ${BRPC_TEST_LIBS}
        local_shm_ps_client
        Folly::folly
        boost_regex
        boost_regex_cxx11
        boost_system
        protobuf::libprotobuf
        gRPC::grpc++
        glog::glog
        ${CUDA_CUDART_LIBRARY}
    )
    target_link_options(test_hugectr_emb PUBLIC -Wl,--no-as-needed)
    add_test(NAME test_hugectr_emb COMMAND test_hugectr_emb)
    set_tests_properties(test_hugectr_emb PROPERTIES LABELS "hierkv;hugectr")
else()
    message(STATUS
        "Skipping HugeCTR runtime tests: CUDA or HugeCTR headers/templates are unavailable."
    )
endif()

if(NOT RECSTORE_DISABLE_BRPC)
add_executable(brpc_ps_server_unit_test
    test_brpc_ps_server.cpp
    ${PROJECT_SOURCE_DIR}/src/ps/brpc/brpc_ps_server.cpp
)
target_include_directories(brpc_ps_server_unit_test PUBLIC ${BRPC_INCLUDE_PATH})
target_include_directories(brpc_ps_server_unit_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/ps/brpc
)
target_compile_definitions(brpc_ps_server_unit_test PRIVATE RECSTORE_NO_SERVER_MAIN)
target_link_libraries(brpc_ps_server_unit_test PUBLIC
    base
    ps_base
    ps_proto
    ${BRPC_LIB}
    ${CUSTOM_PROTOBUF_LIB}
    ${SYSTEM_OPENSSL_SSL_LIB}
    ${SYSTEM_OPENSSL_CRYPTO_LIB}
    ZLIB::ZLIB
    Threads::Threads
    dl
    stdc++fs
    kv_engine
    sparse_tensor
    gflags
    leveldb
    TBB::tbb
    gtest
    gtest_main
)
add_test(NAME brpc_ps_server_unit_test COMMAND brpc_ps_server_unit_test)
set_tests_properties(brpc_ps_server_unit_test PROPERTIES LABELS "ps_unit")
endif()

add_executable(test_allshards_ps_client test_allshards_ps_client.cpp)
target_link_libraries(test_allshards_ps_client PUBLIC
    base
    wq_ps_client
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    gtest
    gtest_main
    pthread
)
add_test(NAME test_allshards_ps_client COMMAND test_allshards_ps_client)

add_executable(test_rdma_rc_protocol test_rdma_rc_protocol.cpp)
target_sources(test_rdma_rc_protocol PRIVATE
    ${PROJECT_SOURCE_DIR}/src/ps/base/parameters.cpp
)
target_link_libraries(test_rdma_rc_protocol PUBLIC
    base
    ps_common
    gtest
    gtest_main
    pthread
)
add_test(NAME test_rdma_rc_protocol COMMAND test_rdma_rc_protocol)

add_executable(test_rdma_control_plane
    test_rdma_control_plane.cpp
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/control_plane.cc
)
target_link_libraries(test_rdma_control_plane PUBLIC
    base
    ps_proto
    gRPC::grpc++
    Folly::folly
    boost_regex
    boost_regex_cxx11
    boost_system
    gtest
    gtest_main
    pthread
)
add_test(NAME test_rdma_control_plane COMMAND test_rdma_control_plane)

add_executable(test_ps_transport_benchmark test_ps_transport_benchmark.cpp)
target_link_libraries(test_ps_transport_benchmark PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_ps_transport_benchmark COMMAND test_ps_transport_benchmark)

add_executable(test_raw_verbs_allocator test_raw_verbs_allocator.cpp)
target_link_libraries(test_raw_verbs_allocator PUBLIC
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_raw_verbs_allocator COMMAND test_raw_verbs_allocator)
add_executable(test_storage_cache_lru test_storage_cache_lru.cpp)
target_link_libraries(test_storage_cache_lru PUBLIC
    gtest
    gtest_main
    pthread
    storage_cache_lru
)
add_test(NAME test_storage_cache_lru COMMAND test_storage_cache_lru)

# report 单元测试
if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_PERF_REPORT)
    add_executable(test_report test_report.cpp)
    target_link_libraries(test_report PUBLIC
        report_client
        gtest
        gtest_main
        pthread
    )
    add_test(NAME test_report COMMAND test_report)
endif()
