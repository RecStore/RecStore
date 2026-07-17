find_package(Torch REQUIRED)

add_library(recstore_torch_ops SHARED
    op_torch.cc
    ../op.cc
    ../common/local_shm_kv_client_op.cc
    ../common/hierkv_local_runtime.cc
    ../common/op_runtime_support.cc
    ../common/local_shm_op_component.cc
    ../common/ps_client_config_adapter.cc
    ../../ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rc_options.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)

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
