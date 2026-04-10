add_library(io_backend INTERFACE)

add_subdirectory(iouring)
add_subdirectory(spdk)

target_link_libraries(io_backend
  INTERFACE
    iouring_backend
    spdk_backend
)
