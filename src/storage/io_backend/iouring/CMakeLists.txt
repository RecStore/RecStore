find_library(LIBURING_LIBRARY uring REQUIRED)

add_library(iouring_backend io_uring_backend.cpp)

target_include_directories(iouring_backend
  PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)

target_link_libraries(iouring_backend
  PUBLIC
    base
    boost_context
    ${LIBURING_LIBRARY}
)
