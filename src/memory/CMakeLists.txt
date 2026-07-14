# src/memory/CMakeLists.txt

add_library(base_memory
  allocators/persist_loop_slab_allocator.cc
  shm_file.cc
)

target_include_directories(base_memory PRIVATE
  ${CMAKE_SOURCE_DIR}/src/memory
  ${JEMALLOC_INSTALL_DIR}/include
)

add_dependencies(base_memory jemalloc)
target_link_libraries(base_memory PUBLIC base jemalloc ${CMAKE_DL_LIBS})

add_executable(pm_command pm_command.cc)
target_link_libraries(pm_command PUBLIC base_memory)

add_executable(benchmark_malloc allocators/benchmark_malloc.cc)
target_link_libraries(benchmark_malloc PUBLIC base_memory pthread)
