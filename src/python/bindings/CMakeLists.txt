pybind11_add_module(_recstore_kv MODULE
  recstore_kv_pybind.cc
  recstore_kv_bindings.cc
  local_kv_binding.cc
  local_indexed_kv_binding.cc
  local_indexed_kv_core.cc
  local_indexed_kv_lookup.cc
  local_indexed_kv_checkpoint.cc
  local_indexed_kv_optimizer.cc
)

target_link_libraries(_recstore_kv PRIVATE
  -Wl,--whole-archive
  kv_engine
  -Wl,--no-whole-archive
  OpenMP::OpenMP_CXX
)

target_compile_definitions(_recstore_kv PRIVATE
  RECSTORE_VERSION="${RECSTORE_VERSION}"
  RECSTORE_KV_API_VERSION=1
  RECSTORE_KV_ABI_VERSION=1
)

set_target_properties(_recstore_kv PROPERTIES
  INSTALL_RPATH "$ORIGIN/native"
  INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
)

install(TARGETS _recstore_kv
  LIBRARY DESTINATION recstore
  COMPONENT python
)

add_test(NAME recstore_kv_binding_metadata
  COMMAND ${CMAKE_COMMAND} -E env
    "LD_PRELOAD=$<TARGET_FILE:_recstore_kv>"
    "PYTHONPATH=$<TARGET_FILE_DIR:_recstore_kv>"
    "${PYTHON_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_recstore_kv_metadata.py"
)
