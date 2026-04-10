add_library(cceh CCEH.cpp)

target_include_directories(cceh
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cceh
  PUBLIC
    base
    io_backend
)
