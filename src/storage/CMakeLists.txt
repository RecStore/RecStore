if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
	add_subdirectory(gpu_cache)
else()
	message(STATUS "Skipping storage/gpu_cache because CUDA is disabled or nvcc not found")
endif()
add_subdirectory(cache)
add_subdirectory(io_backend)
add_subdirectory(index)
add_subdirectory(nvm)
add_subdirectory(hybrid)

# 保持最后
add_subdirectory(kv_engine)
