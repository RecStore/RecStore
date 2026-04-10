find_package(folly REQUIRED)

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

add_executable(test_hash_map
    test_hash_map.cpp 
)

target_link_libraries(test_hash_map PUBLIC
    ${FOLLY_LIBRARIES}
)

target_compile_options(test_hash_map PUBLIC -fopenmp)

target_link_libraries(test_hash_map PUBLIC -fopenmp)


add_executable(corotine 
    corotine.cpp
)

target_link_libraries(corotine PUBLIC
    ${BOOST_LIBRARIES}
    ${FOLLY_LIBRARIES}
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

add_executable(test_engine_extendible_hash test_engine_extendible_hash.cpp)
target_link_libraries(test_engine_extendible_hash PUBLIC
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
add_test(NAME test_engine_extendible_hash COMMAND test_engine_extendible_hash)

add_executable(test_cceh test_cceh.cpp)
target_link_libraries(test_cceh PUBLIC
    cceh
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_cceh COMMAND test_cceh)

add_executable(test_engine_cceh test_engine_cceh.cpp)
target_link_libraries(test_engine_cceh PUBLIC
    cceh
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
)
add_test(NAME test_engine_cceh COMMAND test_engine_cceh)

add_executable(test_io_backend test_io_backend.cpp)
target_link_libraries(test_io_backend PUBLIC
    cceh
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_io_backend COMMAND test_io_backend)

add_executable(test_factory test_factory.cpp)
target_link_libraries(test_factory PUBLIC
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
add_test(NAME test_factory COMMAND test_factory)

add_executable(test_index test_index.cpp)
target_link_libraries(test_index PUBLIC
    extendible_hash 
    base 
    gtest 
    gtest_main 
    pthread
)
add_test(NAME test_index COMMAND test_index)

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
