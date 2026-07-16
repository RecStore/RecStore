set(CITYHASH_ROOT_DIR ${PROJECT_SOURCE_DIR}/third_party/cityhash)
include(${PROJECT_SOURCE_DIR}/cmake/FindCityhash.cmake)

if(NOT RECSTORE_DISABLE_CPPTRACE)
  find_package(cpptrace REQUIRED)
endif()
find_package(CURL REQUIRED)

if(USE_PERF_REPORT)
  add_library(report_client report/report_client.cpp)
  target_link_libraries(report_client PUBLIC curl nlohmann_json::nlohmann_json glog)

  add_library(report SHARED report/report_client.cpp)
  target_link_libraries(report PUBLIC curl nlohmann_json::nlohmann_json glog)
endif()

add_library(base base.cc pprint.cc shm_helper.cc bind_core.cc counter.cc)
target_link_libraries(base PUBLIC ${CITYHASH_LIBRARIES} gflags glog fmt)
target_include_directories(base PUBLIC ${CITYHASH_INCLUDE_DIRS})
target_link_libraries(base PUBLIC protobuf::libprotobuf protobuf::libprotobuf-lite nlohmann_json::nlohmann_json)
if(NOT RECSTORE_DISABLE_CPPTRACE)
  target_link_libraries(base PUBLIC cpptrace::cpptrace)
endif()

# boost regex CXX11 ABI instantiation library.  Defined here (early) so both
# the recstore_torch_ops .so and server/test targets can link it.  folly and
# recstore are built with _GLIBCXX_USE_CXX11_ABI=1, but the system boost 1.69
# libraries only provide the old-ABI symbols for transform()/transform_primary().
add_library(boost_regex_cxx11 STATIC
    ${PROJECT_SOURCE_DIR}/src/framework/pytorch/boost_regex_cxx11_inst.cc)
target_compile_definitions(boost_regex_cxx11 PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(boost_regex_cxx11 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(boost_regex_cxx11 PRIVATE /usr/include/boost169)
set_source_files_properties(${PROJECT_SOURCE_DIR}/src/framework/pytorch/boost_regex_cxx11_inst.cc PROPERTIES
    COMPILE_FLAGS "-include ${PROJECT_SOURCE_DIR}/third_party/deps/dwarf5_compat.h -include cstring -include cstdio -include cstdlib -include string -include algorithm -include utility -include locale")


add_executable(pq_test test_pq.cc)
target_link_libraries(pq_test PUBLIC base)



find_package(pybind11 REQUIRED)
pybind11_add_module(timer_module MODULE pybind_timer.cc)
target_link_libraries(timer_module PRIVATE base)
