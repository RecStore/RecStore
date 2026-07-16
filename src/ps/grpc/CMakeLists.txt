find_package(Python3 COMPONENTS Interpreter Development)

set(BUILD_SHARED_LIBS OFF)  # grpc_ps_client/dist_grpc_ps_client are STATIC; avoids duplicate gRPC static init when loaded by recstore_torch_ops

# gRPC PS Client Library
add_library(grpc_ps_client STATIC
    grpc_ps_client.cpp grpc_ps_client.h
)
set_target_properties(grpc_ps_client PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(grpc_ps_client PUBLIC
    ${PROJECT_BINARY_DIR}/src/ps/proto
)


target_link_libraries(
    grpc_ps_client PUBLIC
    base
    ps_base
    PRIVATE
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
)
if(USE_PERF_REPORT)
    target_compile_definitions(grpc_ps_client PUBLIC ENABLE_PERF_REPORT)
    target_link_libraries(grpc_ps_client PUBLIC report_client)
endif()

# gRPC PS Client Test
add_executable(grpc_ps_client_test
    ${PROJECT_SOURCE_DIR}/src/test/ps/grpc/grpc_ps_client_test.cpp
    ${PROJECT_SOURCE_DIR}/src/test/server_mgr/ps_server_launcher.cpp
)
target_link_libraries(grpc_ps_client_test PUBLIC
    grpc_ps_client
    pthread
    stdc++fs
)
add_test(NAME grpc_ps_client_test COMMAND $<TARGET_FILE:grpc_ps_client_test>)
set_tests_properties(grpc_ps_client_test PROPERTIES LABELS "ps_integration")

# gRPC PS Server
add_executable(grpc_ps_server
    grpc_ps_server.cpp
    grpc_ps_server.h
)

target_link_libraries(grpc_ps_server PUBLIC
    base
    ps_base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    stdc++fs
    kv_engine
    sparse_tensor
    gflags
    glog
    TBB::tbb
)
if(USE_PERF_REPORT)
    target_compile_definitions(grpc_ps_server PRIVATE ENABLE_PERF_REPORT)
    target_link_libraries(grpc_ps_server PUBLIC report_client)
endif()

# Distributed gRPC PS Client
add_library(dist_grpc_ps_client
    dist_grpc_ps_client.cpp dist_grpc_ps_client.h
)
set_target_properties(dist_grpc_ps_client PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Link gRPC only through grpc_ps_client to avoid duplicate grpc symbols in
# consumers (e.g. dist_grpc_ps_client_test), which can deadlock in grpc_init().
target_link_libraries(dist_grpc_ps_client PUBLIC
    base
    ps_base
    grpc_ps_client
)
if(USE_PERF_REPORT)
    target_compile_definitions(dist_grpc_ps_client PUBLIC ENABLE_PERF_REPORT)
    target_link_libraries(dist_grpc_ps_client PUBLIC report_client)
endif()

# Distributed gRPC PS Client Test
add_executable(dist_grpc_ps_client_test
    ${PROJECT_SOURCE_DIR}/src/test/ps/grpc/dist_grpc_ps_client_test.cpp
    ${PROJECT_SOURCE_DIR}/src/test/server_mgr/ps_server_launcher.cpp
)
target_link_libraries(dist_grpc_ps_client_test PUBLIC
    dist_grpc_ps_client
    pthread
    stdc++fs
)
add_test(NAME dist_grpc_ps_client_test COMMAND $<TARGET_FILE:dist_grpc_ps_client_test>)
set_tests_properties(dist_grpc_ps_client_test PROPERTIES LABELS "ps_integration")

# gRPC PS Network Benchmark
add_executable(grpc_ps_bench
    grpc_ps_bench.cpp
)
target_link_libraries(grpc_ps_bench PUBLIC
    grpc_ps_client
    gflags
)
