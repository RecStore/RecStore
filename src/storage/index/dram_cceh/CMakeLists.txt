add_library(extendible_hash)

target_sources(extendible_hash
  PRIVATE 
   extendible_hash.cpp
)

target_include_directories(extendible_hash
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(extendible_hash
  PUBLIC
    base
)
