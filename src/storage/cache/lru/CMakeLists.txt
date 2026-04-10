add_library(storage_cache_lru INTERFACE)

target_include_directories(storage_cache_lru
  INTERFACE
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(storage_cache_lru
  INTERFACE
    storage_cache
)
