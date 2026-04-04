find_package(folly REQUIRED)

add_subdirectory(small_tests)

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
    dram_hash 
    base 
    gtest 
    gtest_main 
    pthread
    base_memory
)
add_test(NAME test_extendible_hash COMMAND test_extendible_hash)

add_executable(test_dram_index_ssd_value test_dram_index_ssd_value.cpp)
target_link_libraries(test_dram_index_ssd_value PUBLIC
    dram_hash
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
    TBB::tbb
)
add_test(NAME test_dram_index_ssd_value COMMAND test_dram_index_ssd_value)

add_executable(test_ssdcceh test_ssdcceh.cpp)
target_link_libraries(test_ssdcceh PUBLIC
    ssd_cceh
    base
    gtest
    gtest_main
    pthread
)
add_test(NAME test_ssdcceh COMMAND test_ssdcceh)

add_executable(test_ssd_index_ssd_value test_ssd_index_ssd_value.cpp)
target_link_libraries(test_ssd_index_ssd_value PUBLIC
    ssd_cceh
    base
    base_memory
    ps_common
    kv_engine
    gtest
    gtest_main
    pthread
)
add_test(NAME test_ssd_index_ssd_value COMMAND test_ssd_index_ssd_value)

add_executable(test_factory test_factory.cpp)
target_link_libraries(test_factory PUBLIC
    dram_hash
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
    dram_hash 
    base 
    gtest 
    gtest_main 
    pthread
)
add_test(NAME test_index COMMAND test_index)

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