add_library(sparse_tensor
	sparse_tensor.cpp
	optimizer.cpp
)

target_link_libraries(sparse_tensor PRIVATE
    kv_engine
)

if(USE_PERF_REPORT)
    target_compile_definitions(sparse_tensor PRIVATE ENABLE_PERF_REPORT)
    target_link_libraries(sparse_tensor PRIVATE report_client)
endif()
