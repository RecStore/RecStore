option(USE_FAKE_KVCLIENT "Use FakeKVClientOp for testing" OFF)
option(RECSTORE_ENABLE_TENSORFLOW "Build TensorFlow framework ops" OFF)
option(RECSTORE_ENABLE_GPU_CACHE "Build RecStore framework GPU cache" ON)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER AND RECSTORE_ENABLE_GPU_CACHE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
else()
    message(STATUS "Skipping framework/gpu because CUDA is disabled, nvcc is unavailable, or RECSTORE_ENABLE_GPU_CACHE is OFF")
endif()

set(_recstore_libtorch_lib_dir "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/lib")
set(_recstore_libtorch_requires_cuda OFF)
if(EXISTS "${_recstore_libtorch_lib_dir}/libtorch_cuda.so" OR
   EXISTS "${_recstore_libtorch_lib_dir}/libc10_cuda.so")
    set(_recstore_libtorch_requires_cuda ON)
endif()

if(_recstore_libtorch_requires_cuda AND NOT (ENABLE_CUDA AND CMAKE_CUDA_COMPILER))
    message(STATUS "Skipping framework/pytorch because CUDA libtorch is present but CUDA is disabled or nvcc not found")
else()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)
endif()

if(RECSTORE_ENABLE_TENSORFLOW)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
else()
    message(STATUS "Skipping framework/tensorflow because RECSTORE_ENABLE_TENSORFLOW is OFF")
endif()

if(USE_FAKE_KVCLIENT)
    message(STATUS "Using FakeKVClientOp for testing!")
    add_definitions(-DUSE_FAKE_KVCLIENT)
endif()

if(TARGET recstore_torch_ops)
    add_test(
        NAME pytorch_client_test
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
                "$<TARGET_FILE:recstore_torch_ops>"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
    )

    add_test(
        NAME pytorch_client_test_rdma_basic
        COMMAND ${CMAKE_COMMAND} -E env
                RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
                RECSTORE_CLIENT_TEST_PHASE=basic
                RECSTORE_USE_LOCAL_MEMCACHED=auto
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
                "$<TARGET_FILE:recstore_torch_ops>"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
    )

    add_test(
        NAME pytorch_client_test_rdma
        COMMAND ${CMAKE_COMMAND} -E env
                RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
                RECSTORE_CLIENT_TEST_PHASE=full
                RECSTORE_USE_LOCAL_MEMCACHED=auto
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
                "$<TARGET_FILE:recstore_torch_ops>"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
    )

    add_test(
        NAME pytorch_client_test_rdma_auto
        COMMAND ${CMAKE_COMMAND} -E env
                RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
                RECSTORE_CLIENT_TEST_PHASE=full
                RECSTORE_USE_LOCAL_MEMCACHED=auto
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
                "$<TARGET_FILE:recstore_torch_ops>"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
    )

    add_test(
        NAME pytorch_client_test_rdma_descriptor_basic
        COMMAND ${CMAKE_COMMAND} -E env
                RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
                RECSTORE_CLIENT_TEST_PHASE=basic
                RECSTORE_USE_LOCAL_MEMCACHED=auto
                RECSTORE_RDMA_TRANSPORT_MODE=descriptor_doorbell
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
                "$<TARGET_FILE:recstore_torch_ops>"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
    )

    set_tests_properties(
        pytorch_client_test_rdma_basic
        pytorch_client_test_rdma
        pytorch_client_test_rdma_auto
        pytorch_client_test_rdma_descriptor_basic
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 90
            LABELS "rdma_integration"
    )
endif()

if(TARGET recstore_torch_ops)
    add_test(
        NAME dist_emb_unittest
        COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb
    )

    set_tests_properties(dist_emb_unittest PROPERTIES
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
    )
else()
    message(STATUS "Skipping dist_emb_unittest because recstore_torch_ops is unavailable")
endif()

add_test(
    NAME ebc_precision_test
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_ebc_precision_wrapper
)

set_tests_properties(ebc_precision_test PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)

add_test(
    NAME ebc_multiprocess_precision_test
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_ebc_precision_wrapper.TestEBCPrecision.test_multiprocess_precision
)

set_tests_properties(ebc_multiprocess_precision_test PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)
