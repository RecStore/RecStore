find_library(LIBURING_LIBRARY uring REQUIRED)

add_library(iouring_backend io_uring_backend.cpp)

target_link_libraries(iouring_backend
  PUBLIC
    base
    ${LIBURING_LIBRARY}
)
