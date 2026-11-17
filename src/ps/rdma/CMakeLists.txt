option(ENABLE_RDMA_INTEGRATION_TESTS "Enable RDMA integration tests" OFF)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_library(wq_ps_client
    control_plane.cc
    rdma_deployment.cc
    rdma_distributed_executor.cc
    petps_client.cc
    rc_options.cc
    rc_transport.cc
    raw_verbs_transport.cc
    ${PROJECT_SOURCE_DIR}/src/ps/base/parameters.cpp
)

target_link_libraries(wq_ps_client PUBLIC
    base
    ps_common
    ps_proto
    gRPC::grpc++
    Folly::folly
    pthread
    rt
    ibverbs
)
set(warp_wq_ps_client -Wl,--whole-archive wq_ps_client -Wl,--no-whole-archive CACHE INTERNAL "wq_ps_client")

add_executable(petps_server
    control_plane.cc
    rdma_deployment.cc
    petps_server.cc
    rc_options.cc
    rc_transport.cc
    raw_verbs_transport.cc
)
target_link_libraries(petps_server PUBLIC
    base
    ps_base
    kv_engine
    sparse_tensor
    ps_common
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    TBB::tbb
    Folly::folly
    boost_context
    boost_coroutine
    boost_regex
    boost_system
    pthread
    rt
    ibverbs
)

add_executable(
    petps_integration_test
    ${PROJECT_SOURCE_DIR}/src/test/ps/rdma/petps_integration_test.cpp
)
target_sources(petps_integration_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/framework/common/ps_client_config_adapter.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)
add_dependencies(petps_integration_test petps_server)
target_link_libraries(petps_integration_test PUBLIC
    wq_ps_client
    base
    gtest
    Folly::folly
    boost_regex
    boost_system
    pthread
)

add_test(
    NAME petps_hashed_value_transfer_test
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
            --server-count 1
            --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
            --test-binary $<TARGET_FILE:petps_integration_test>
            --gtest-filter=PetPSIntegrationTest.HashedValueBatchGetTransferSingleShard
            --value-size 512
            --max-kv-num-per-request 500
            --client-timeout 45
            --cluster-timeout 45
)
set_tests_properties(
    petps_hashed_value_transfer_test
    PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 90
        LABELS "rdma_integration"
)

add_test(
    NAME petps_lane_rebuild_test
    COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
            --server-count 1
            --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
            --test-binary $<TARGET_FILE:petps_integration_test>
            --gtest-filter=PetPSIntegrationTest.LaneRebuildResumesServiceSingleShard:PetPSIntegrationTest.RebuildDropsInflightAsyncRpcRepro
            --rdma-lane-rebuild
            --client-timeout 60
            --cluster-timeout 60
)
set_tests_properties(
    petps_lane_rebuild_test
    PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 120
        LABELS "rdma_integration"
)

# The merged-gradient (v4 owner) path needs two shards and city_hash, and every
# case must run in its own process: the QP rendezvous protocol carries no
# instance identifier, so a process may only host one adapter.
function(add_petps_owner_test name case)
    add_test(
        NAME ${name}
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 2
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_multishard_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.${case}
                --value-size 512
                --max-kv-num-per-request 500
                --client-timeout 90
                --cluster-timeout 90
                ${ARGN}
    )
    set_tests_properties(
        ${name}
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 180
            LABELS "rdma_integration"
    )
endfunction()

add_petps_owner_test(
    petps_owner_exact_bits_test
    OwnerExactBitsAndEmptyOwnerMultiShard
)
add_petps_owner_test(
    petps_owner_wrong_rank_or_shard_test
    OwnerWrongRankOrShardMultiShard
)
add_petps_owner_test(
    petps_owner_late_invalid_sticky_test
    OwnerLateInvalidStickyMultiShard
)
add_petps_owner_test(
    petps_owner_retry_and_version_test
    OwnerRetryAndVersionMultiShard
)
# One OS client exposes two logical lanes with one 1-slot QP each, which is what
# the adapter-abort window needs to stay open.
add_petps_owner_test(
    petps_owner_adapter_abort_test
    OwnerAdapterAbortReachesAllShardsWindowOne
    --logical-clients-per-process 2
    --rdma-qps-per-client-per-shard 1
)

if(ENABLE_RDMA_INTEGRATION_TESTS)
add_test(
        NAME petps_single_shard_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 1
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots
    )
    set_tests_properties(
        petps_single_shard_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
            LABELS "rdma_integration"
    )

    # Lane rebuild racing the get-payload worker threads. Needs the offload
    # workers so it is opt-in: the sustained collision takes ~30s.
    add_test(
        NAME petps_lane_rebuild_get_workers_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 1
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.RebuildRacesGetPayloadWorkersRepro
                --rdma-lane-rebuild
                --rdma-rc-server-get-workers=4
                --client-timeout 120
                --cluster-timeout 60
    )
    set_tests_properties(
        petps_lane_rebuild_get_workers_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 180
            LABELS "rdma_integration"
    )

    # Deterministic fence proof: stall the GET payload workers so a task
    # provably straddles a lane rebuild. The driver asserts shard-0 reported
    # fence events (fence_rejected + dropped_stale > 0); the test itself only
    # guards that no wrong response leaks through.
    add_test(
        NAME petps_lane_rebuild_fence_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 1
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.RebuildRacesGetPayloadWorkersRepro
                --rdma-lane-rebuild
                --rdma-rc-server-get-workers=4
                --rdma-rc-server-get-worker-hold-us=2000000
                --client-timeout 120
                --cluster-timeout 60
                --expect-fence-events
    )
    set_tests_properties(
        petps_lane_rebuild_fence_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 180
            LABELS "rdma_integration"
    )

    add_test(
        NAME petps_adapter_split_get_multi_shard_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 2
                --client-count 1
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_multishard_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.AdapterSplitGetRoundTripMultiShard
                --max-kv-num-per-request 2
    )
    set_tests_properties(
        petps_adapter_split_get_multi_shard_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
            LABELS "rdma_integration"
    )

    add_test(
        NAME petps_multiclient_stress_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_multiclient_stress.py
                --server-count 1
                --client-count 2
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.RepeatedPutGetStressSingleShard:PetPSIntegrationTest.AsyncGetPrefetchStressSingleShard
    )
    set_tests_properties(
        petps_multiclient_stress_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 90
            LABELS "rdma_integration"
    )

    add_test(
        NAME petps_multishard_multiclient_stress_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_multiclient_stress.py
                --server-count 2
                --client-count 2
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_multishard_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.RepeatedPutGetStressMultiShard
    )
    set_tests_properties(
        petps_multishard_multiclient_stress_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 120
            LABELS "rdma_integration"
    )

    # Fault injection in shard 1 while the test drives the client. The driver
    # owns the per-scenario flags; it only needs the scenario name.
    foreach(fault_scenario fast_fail rebuild killed)
        add_test(
            NAME petps_fault_injection_${fault_scenario}_test
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_fault_injection.py
                    --scenario ${fault_scenario}
                    --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_multishard_test.json
                    --test-binary $<TARGET_FILE:petps_integration_test>
        )
        set_tests_properties(
            petps_fault_injection_${fault_scenario}_test
            PROPERTIES
                SKIP_RETURN_CODE 77
                TIMEOUT 300
                LABELS "rdma_integration"
        )
    endforeach()
endif()
