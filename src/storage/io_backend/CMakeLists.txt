add_library(io_backend INTERFACE)

add_subdirectory(iouring)

option(RECSTORE_ENABLE_SPDK "Build SPDK backend" OFF)
if(RECSTORE_ENABLE_SPDK)
  add_subdirectory(spdk)
  target_link_libraries(io_backend INTERFACE iouring_backend spdk_backend)
  target_compile_definitions(io_backend INTERFACE RECSTORE_ENABLE_SPDK)
else()
  target_link_libraries(io_backend INTERFACE iouring_backend)
endif()
