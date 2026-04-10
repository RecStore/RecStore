add_library(storage_cache INTERFACE)

target_include_directories(storage_cache
  INTERFACE
    ${PROJECT_SOURCE_DIR}/src
)

add_subdirectory(lru)
