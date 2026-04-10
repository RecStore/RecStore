find_package(PkgConfig)


#set (RECSTORE_SPDK_ENABLED OFF)
set (RECSTORE_SPDK_ENABLED ON)
if(RECSTORE_SPDK_ENABLED )


pkg_search_module(SPDK REQUIRED IMPORTED_TARGET GLOBAL spdk_nvme spdk_env_dpdk)
message(STATUS SPDK_LIBRARIES ${SPDK_LIBRARIES})
message(STATUS SPDK_INCLUDE_DIR ${SPDK_INCLUDE_DIR})

set(SPDK_POS "${PROJECT_SOURCE_DIR}/third_party/spdk")

set(SPDK_LIB "-L/usr/local/lib -L${SPDK_POS}/build/lib \ 
 -lspdk_nvme -lspdk_keyring -lspdk_sock -lspdk_sock_posix -lspdk_thread -lspdk_trace \ 
 -lspdk_rpc -lspdk_jsonrpc -lspdk_json -lspdk_vfio_user -lspdk_vmd -lspdk_env_dpdk \ 
 -lspdk_dma -lspdk_util -lspdk_log \
 -lssl -lfuse3
")

set(DPDK_LIB "-L${SPDK_POS}/dpdk/build/lib \
    -lrte_eal -lrte_eal -lrte_mempool -lrte_ring -lrte_mbuf \ 
    -lrte_bus_pci -lrte_pci -lrte_mempool_ring -lrte_telemetry -lrte_kvargs \ 
    -lrte_power -lrte_ethdev -lrte_net -lrte_vhost -lrte_net -lrte_dmadev -lrte_cryptodev -lrte_hash -lrte_rcu \ "
)

separate_arguments(SPDK_LIB UNIX_COMMAND "${SPDK_LIB}")
separate_arguments(DPDK_LIB UNIX_COMMAND "${DPDK_LIB}")

add_executable(hello_spdk hello_world.cpp)
target_link_options(hello_spdk PUBLIC -Wl,--whole-archive -Wl,--no-as-needed)
target_link_options(hello_spdk PUBLIC ${SPDK_LIB} ${DPDK_LIB})
target_link_options(hello_spdk PUBLIC -Wl,--no-whole-archive)
target_link_options(hello_spdk PUBLIC -lnuma -ldl  -L${PROJECT_SOURCE_DIR}/third_party/spdk/isa-l/.libs -lisal -pthread -lrt -luuid -lcrypto -lm -laio)

add_library(spdk_wrapper spdk_wrapper.cc)
target_link_options(spdk_wrapper INTERFACE -Wl,--whole-archive -Wl,--no-as-needed)
target_link_options(spdk_wrapper INTERFACE ${SPDK_LIB} ${DPDK_LIB})
target_link_options(spdk_wrapper INTERFACE -Wl,--no-whole-archive)
target_link_options(spdk_wrapper INTERFACE -lnuma -ldl  -L${PROJECT_SOURCE_DIR}/third_party/spdk/isa-l/.libs -lisal -pthread -lrt -luuid -lcrypto -lm -laio)
target_link_libraries(spdk_wrapper Folly::folly)
target_link_libraries(spdk_wrapper protobuf::libprotobuf)

add_executable(perf_spdk perf_spdk.cc)
target_link_libraries(perf_spdk spdk_wrapper)

add_executable(test_spdk_wrapper test_spdk_wrapper.cc)
target_link_libraries(test_spdk_wrapper spdk_wrapper gtest)

add_executable(test_naivekvell test_naiveKvell.cc)
target_link_libraries(test_naivekvell  spdk_wrapper gtest)


# add_executable(perf_spdkkvell_from_gpu perf_spdkkvell_from_gpu.cu)
# target_link_libraries(perf_spdkkvell_from_gpu spdk_wrapper Folly::folly)
# set_target_properties(perf_spdkkvell_from_gpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_target_properties(perf_spdkkvell_from_gpu PROPERTIES CUDA_ARCHITECTURES OFF)
# set_property(TARGET perf_spdkkvell_from_gpu PROPERTY CUDA_STANDARD 17)

# find_package(RocksDB)

# add_library(ssd_ps_lib INTERFACE)
# target_link_libraries(ssd_ps_lib INTERFACE RocksDB::rocksdb-shared spdk_wrapper)

endif(RECSTORE_SPDK_ENABLED )
