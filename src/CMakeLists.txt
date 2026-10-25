add_subdirectory(base)
add_subdirectory(ps)
add_subdirectory(memory)
add_subdirectory(storage)
add_subdirectory(python/bindings)
add_subdirectory(optimizer)
add_subdirectory(benchmark)

add_subdirectory(test)

add_subdirectory(framework)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  add_subdirectory(executable/uvm)
else()
  message(STATUS "Skipping executable/uvm because CUDA is disabled or nvcc not found")
endif()
