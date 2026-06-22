cmake_minimum_required(VERSION 3.10)

add_library(recstore_quantarec_adapter STATIC
  quantarec_ps_store.cc
)

target_include_directories(recstore_quantarec_adapter
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(recstore_quantarec_adapter PUBLIC cxx_std_17)
