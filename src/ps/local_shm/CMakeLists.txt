add_library(local_shm_common SHARED
    local_shm_futex.cpp
    local_shm_queue.cpp
    local_shm_region.cpp
)

target_link_libraries(local_shm_common PUBLIC
    base
)

add_library(local_shm_ps_client SHARED
    local_shm_client.cpp
)

target_link_libraries(local_shm_ps_client PUBLIC
    local_shm_common
    base
    ps_base
    pthread
)

add_library(local_shm_ps_server_runtime SHARED
    local_shm_server.cpp
)

target_link_libraries(local_shm_ps_server_runtime PUBLIC
    local_shm_common
    base
    ps_base
)

target_link_libraries(local_shm_ps_server_runtime PRIVATE
    sparse_tensor
    gflags
    glog
    jemalloc_compat
    TBB::tbb
    pthread
)
target_compile_options(local_shm_ps_server_runtime PRIVATE -msse4.2)

if(TARGET report_client)
  target_compile_definitions(local_shm_ps_server_runtime PRIVATE ENABLE_PERF_REPORT)
  target_link_libraries(local_shm_ps_server_runtime PRIVATE report_client)
endif()

if(NOT BUILD_SHARED_LIBS AND TARGET spdk_backend)
  target_link_libraries(local_shm_ps_server_runtime PUBLIC spdk_backend)
endif()

add_executable(local_shm_ps_server
    local_shm_server_main.cpp
)

target_link_libraries(local_shm_ps_server PUBLIC
    local_shm_ps_server_runtime
    Folly::folly
    pthread
)
