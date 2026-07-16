# FindTBB.cmake - locate system TBB and define TBB::tbb target
find_path(TBB_INCLUDE_DIR NAMES tbb/tbb.h PATHS /usr/include)
find_library(TBB_LIBRARY NAMES tbb PATHS /usr/lib64 /usr/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG TBB_LIBRARY TBB_INCLUDE_DIR)
if(TBB_FOUND AND NOT TARGET TBB::tbb)
  add_library(TBB::tbb SHARED IMPORTED)
  set_target_properties(TBB::tbb PROPERTIES
    IMPORTED_LOCATION "${TBB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INCLUDE_DIR}")
endif()
mark_as_advanced(TBB_INCLUDE_DIR TBB_LIBRARY)
