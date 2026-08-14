add_library(detable_plugin_api INTERFACE)
target_include_directories(detable_plugin_api INTERFACE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(detable_plugin_api INTERFACE ${CMAKE_DL_LIBS})
