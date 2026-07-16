find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

if(USE_BRPC_CPU_PROFILER)
    find_library(GPERFTOOLS_TCMALLOC_AND_PROFILER
        NAMES tcmalloc_and_profiler
    )
    if(NOT GPERFTOOLS_TCMALLOC_AND_PROFILER)
        message(FATAL_ERROR "USE_BRPC_CPU_PROFILER=ON but libtcmalloc_and_profiler was not found")
    endif()
endif()


add_subdirectory(base)
add_subdirectory(proto)
add_subdirectory(grpc)
add_subdirectory(brpc)
add_subdirectory(rdma)
add_subdirectory(local_shm)

if(brpc_FOUND)
    add_executable(ps_server
        ps_server.cpp
    )
    target_compile_definitions(ps_server PRIVATE RECSTORE_NO_SERVER_MAIN=1)
    target_include_directories(ps_server PUBLIC ${BRPC_INCLUDE_PATH})
    
    target_link_libraries(ps_server PUBLIC
        base
        ps_base
        ps_proto
        # gRPC dependencies
        gRPC::grpc++_reflection
        gRPC::grpc++
        protobuf::libprotobuf
        # bRPC dependencies
        ${BRPC_LIB}
        ${CUSTOM_PROTOBUF_LIB}
        ${SYSTEM_OPENSSL_SSL_LIB}
        ${SYSTEM_OPENSSL_CRYPTO_LIB}
        ZLIB::ZLIB
        Threads::Threads
        dl
        # Common dependencies
        stdc++fs
        kv_engine
        sparse_tensor
        gflags
        glog
        leveldb
        TBB::tbb
    )
    
    target_sources(ps_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/grpc/grpc_ps_server.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/brpc/brpc_ps_server.cpp
    )

    if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_GPERF_PROFILING)
        target_compile_definitions(ps_server PRIVATE ENABLE_GPERF_PROFILING)
        target_link_libraries(ps_server PRIVATE profiler)
    endif()

    if(USE_BRPC_CPU_PROFILER)
        target_compile_definitions(ps_server PRIVATE BRPC_ENABLE_CPU_PROFILER)
        target_compile_options(ps_server PRIVATE -fno-omit-frame-pointer)
        target_link_libraries(ps_server PRIVATE ${GPERFTOOLS_TCMALLOC_AND_PROFILER})
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "Debug" AND USE_PERF_REPORT)
        target_compile_definitions(ps_server PRIVATE ENABLE_PERF_REPORT)
        target_link_libraries(ps_server PRIVATE report_client)
    endif()
endif()
