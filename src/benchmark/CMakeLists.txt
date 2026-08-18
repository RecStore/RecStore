
if(RECSTORE_HUGECTR_CUDA_HEADERS_AVAILABLE)
    add_executable(backend_benchmark storage/backend_benchmark.cc)
    target_link_libraries(backend_benchmark PUBLIC
        pthread
        base
        kv_engine
        ps_common
        TBB::tbb
        hps_backend_support
        fasterkv_backend
        rocksdb-shared
        OpenMP::OpenMP_CXX
    )
else()
    message(STATUS "Skipping backend_benchmark: HugeCTR CUDA headers are unavailable.")
endif()

add_executable(dataset_wordcount_helper dataset/dataset_wordcount.cc)
target_link_libraries(dataset_wordcount_helper PUBLIC pthread base Folly::folly)

add_executable(dataset_renumber_id_helper dataset/dataset_renumbering.cc)
target_link_libraries(dataset_renumber_id_helper PUBLIC pthread base Folly::folly)
target_link_libraries(dataset_renumber_id_helper PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(dataset_renumber_id_helper PUBLIC TBB::tbb)

add_executable(benchmark_client
    ps/benchmark_client.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(benchmark_client PUBLIC ${warp_wq_ps_client} base Folly::folly)
target_link_libraries(benchmark_client PUBLIC OpenMP::OpenMP_CXX)

add_executable(ps_transport_benchmark
    ps/ps_transport_benchmark.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(ps_transport_benchmark PUBLIC
    base
    grpc_ps_client
    dist_grpc_ps_client
    brpc_ps_client
    dist_brpc_ps_client
    local_shm_ps_client
    ${warp_wq_ps_client}
    Folly::folly
    OpenMP::OpenMP_CXX
)

add_executable(rdma_rc_transport_benchmark
    ps/rdma_rc_transport_benchmark.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(rdma_rc_transport_benchmark PUBLIC
    base
    ${warp_wq_ps_client}
    Folly::folly
    OpenMP::OpenMP_CXX
)

add_executable(recstore_mixed_benchmark
    mixed/recstore_mixed_benchmark.cc
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
target_link_libraries(recstore_mixed_benchmark PUBLIC
    base
    grpc_ps_client
    dist_grpc_ps_client
    brpc_ps_client
    dist_brpc_ps_client
    local_shm_ps_client
    ${warp_wq_ps_client}
    Folly::folly
    OpenMP::OpenMP_CXX
)
