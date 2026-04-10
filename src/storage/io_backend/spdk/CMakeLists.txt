find_package(PkgConfig)

set(RECSTORE_SPDK_ENABLED ON)
if(RECSTORE_SPDK_ENABLED)

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
 -lrte_power -lrte_ethdev -lrte_net -lrte_vhost -lrte_net -lrte_dmadev -lrte_cryptodev -lrte_hash -lrte_rcu ")

separate_arguments(SPDK_LIB UNIX_COMMAND "${SPDK_LIB}")
separate_arguments(DPDK_LIB UNIX_COMMAND "${DPDK_LIB}")

add_library(spdk_backend spdk_backend.cpp)

target_link_options(spdk_backend INTERFACE -Wl,--whole-archive -Wl,--no-as-needed)
target_link_options(spdk_backend INTERFACE ${SPDK_LIB} ${DPDK_LIB})
target_link_options(spdk_backend INTERFACE -Wl,--no-whole-archive)
target_link_options(spdk_backend INTERFACE -lnuma -ldl -L${PROJECT_SOURCE_DIR}/third_party/spdk/isa-l/.libs -lisal -pthread -lrt -luuid -lcrypto -lm -laio)

target_include_directories(spdk_backend PUBLIC ${SPDK_INCLUDE_DIRS})
target_link_libraries(spdk_backend PUBLIC base)

endif(RECSTORE_SPDK_ENABLED)
