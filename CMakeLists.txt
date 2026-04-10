cmake_minimum_required(VERSION 3.10)

option(ENABLE_CUDA "Build with CUDA support" ON)
option(USE_BRPC_CPU_PROFILER "Enable bRPC builtin CPU profiler" OFF)

if(ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(_recstore_nvcc_candidates
    /usr/local/cuda/bin/nvcc
    /usr/bin/nvcc
  )
  file(GLOB _recstore_nvcc_glob_candidates "/usr/local/cuda-*/bin/nvcc")
  list(APPEND _recstore_nvcc_candidates ${_recstore_nvcc_glob_candidates})

  foreach(_recstore_nvcc IN LISTS _recstore_nvcc_candidates)
    if(EXISTS "${_recstore_nvcc}")
      set(CMAKE_CUDA_COMPILER "${_recstore_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
      get_filename_component(_recstore_cuda_bin "${_recstore_nvcc}" DIRECTORY)
      get_filename_component(_recstore_cuda_root "${_recstore_cuda_bin}" DIRECTORY)
      set(CUDAToolkit_ROOT "${_recstore_cuda_root}" CACHE PATH "CUDA toolkit root" FORCE)
      set(CUDA_TOOLKIT_ROOT_DIR "${_recstore_cuda_root}" CACHE PATH "CUDA toolkit root (FindCUDA compatibility)" FORCE)
      message(STATUS "Auto-detected nvcc: ${CMAKE_CUDA_COMPILER}")
      break()
    endif()
  endforeach()
endif()

if(ENABLE_CUDA)
  project(RecStore LANGUAGES CXX CUDA C)
else()
  project(RecStore LANGUAGES CXX C)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
if(ENABLE_CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") # works

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
endif()
find_package(OpenMP REQUIRED)
find_package(Threads)

include(ExternalProject)




set(JEMALLOC_INSTALL_DIR ${CMAKE_BINARY_DIR}/jemalloc-install)

ExternalProject_Add(jemalloc_ext
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/third_party/jemalloc   # 你的 jemalloc 源码目录
  CONFIGURE_COMMAND autoconf
  BUILD_IN_SOURCE   1
  BUILD_COMMAND     ./configure
                      --with-jemalloc-prefix=je
                      --prefix=${JEMALLOC_INSTALL_DIR}
                      CC=${CMAKE_C_COMPILER}
                      CXX=${CMAKE_CXX_COMPILER}
                    && $(MAKE) -j
  INSTALL_COMMAND   $(MAKE) install
)


set(JEMALLOC_LIB_PATH ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.a)


file(MAKE_DIRECTORY ${JEMALLOC_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${JEMALLOC_INSTALL_DIR}/lib)

add_library(jemalloc STATIC IMPORTED GLOBAL)   
set_target_properties(jemalloc PROPERTIES
  IMPORTED_LOCATION             ${JEMALLOC_LIB_PATH}
  INTERFACE_INCLUDE_DIRECTORIES ${JEMALLOC_INSTALL_DIR}/include
)
add_dependencies(jemalloc jemalloc_ext)


list(APPEND CMAKE_BUILD_RPATH   "${JEMALLOC_INSTALL_DIR}/lib")
list(APPEND CMAKE_INSTALL_RPATH "${JEMALLOC_INSTALL_DIR}/lib")


if (OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fuse-ld=mold -faligned-new -Wno-unused-function")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -faligned-new -Wno-unused-function")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")


# setting compiler flags

if(ENABLE_CUDA)
  set(SM "80;86")
  foreach(arch_name ${SM})
    if (arch_name STREQUAL 86 OR 
      arch_name STREQUAL 80 OR 
      arch_name STREQUAL 75 OR 
      arch_name STREQUAL 70 OR 
      arch_name STREQUAL 61 OR 
      arch_name STREQUAL 60)
      list(APPEND cuda_arch_list ${arch_name})
      message(STATUS "-- Assign GPU architecture (sm=${arch_name})")
    else()
      message(FATAL_ERROR "-- Unknown or unsupported GPU architecture (set sm=70)")
    endif()
  endforeach()

  list(LENGTH cuda_arch_list cuda_arch_list_length)
  if(${cuda_arch_list_length} EQUAL 0)
    list(APPEND cuda_arch_list "80")
  endif()
  list(REMOVE_DUPLICATES cuda_arch_list)

  foreach(arch_name ${cuda_arch_list})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch_name},code=sm_${arch_name}")
  endforeach()
endif()

set(CC_WARNING_FLAG "-Wall -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-value -Wno-unused-variable -Wno-attributes -Wno-parentheses -Wno-unused-but-set-variable")
string(REPLACE " " "," NVCC_WARNING_FLAG ${CC_WARNING_FLAG})

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -fPIC -g -ggdb3 -Wall")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -g -ggdb3 ${CC_WARNING_FLAG} --std=c++17")
if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++ -rdc=true --std=c++17 -Xcompiler ${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC -lineinfo --expt-extended-lambda --expt-relaxed-constexpr")
endif()

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG} -fPIC -g -ggdb3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -g -ggdb3 -O0 ${CC_WARNING_FLAG}")
if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG}  -g -G -Xptxas -O0 -lineinfo -O0 -Xcompiler -O0,${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC")
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fPIC -g -ggdb3 -O3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -fPIC -g -ggdb3 -O3 ${CC_WARNING_FLAG}")
if(ENABLE_CUDA)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}  -g -Xptxas -O3 -lineinfo -O3 -Xcompiler -O3,${NVCC_WARNING_FLAG},-g,-ggdb3,-fPIC")
endif()


# setting output folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

# building

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/third_party/grpc-install")

include("${CMAKE_CURRENT_LIST_DIR}/cmake/FindGRPC.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake")



list(APPEND CMAKE_PREFIX_PATH "/home/xieminhui/.local/share/cmake/")
list(APPEND CMAKE_PREFIX_PATH "/home/pod/.local/share/cmake/")

list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch")
set(Torch_DIR "${PROJECT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch")

# list(APPEND CMAKE_PREFIX_PATH "/home/xieminhui/.local/lib/python3.10/site-packages/torch/share/cmake/Torch/")


# if (EXISTS "/home/pod/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
#   list(APPEND CMAKE_PREFIX_PATH "/home/pod/.local/lib/python3.10/site-packages/torch/share/cmake/Torch")
# endif()

# if (EXISTS "/home/xieminhui/miniconda3/lib/python3.11/site-packages/torch/share/cmake/Torch")
#   set(Torch_DIR "/home/xieminhui/miniconda3/lib/python3.11/site-packages/torch/share/cmake/Torch")
# endif()

# if (EXISTS "/home/xieminhui/anaconda3/lib/python3.1/site-packages/torch/share/cmake/Torch")
#   set(Torch_DIR "/home/xieminhui/anaconda3/lib/python3.1/site-packages/torch/share/cmake/Torch")
# endif()







set(folly_DIR "${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/lib/cmake/folly")
set(FOLLY_ROOT_DIR "${folly_DIR}")

set(FMT_DEBUG_POSTFIX "" CACHE STRING "Debug library postfix." FORCE)
add_subdirectory(third_party/fmt)
find_package(folly REQUIRED)
find_package(gflags REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/include)
link_directories(${PROJECT_SOURCE_DIR}/third_party/folly/folly-install-fPIC/usr/local/lib)

set(glog_DIR "${PROJECT_SOURCE_DIR}/third_party/glog/glog-install-fPIC/usr/local/lib/cmake/glog")
find_package(glog CONFIG REQUIRED)


if(OPENMP_FOUND)
  if(ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()
  message(STATUS "add -fopenmp to compiler")
else()
  message(FATAL_ERROR "without openmp the multi-node all2all will fail")
endif()

add_subdirectory(third_party/googletest)
add_subdirectory(third_party/Mayfly-main)
add_subdirectory(third_party/json)
# add_subdirectory(third_party/pybind11)
# add_subdirectory(third_party/mimalloc)

set(PYBIND11_CMAKE_DIR "/usr/local/lib/python3.10/dist-packages/pybind11/share/cmake/pybind11")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

set(CTEST_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/Testing")
enable_testing()

include_directories(
  ${PROJECT_SOURCE_DIR}/src
)
include_directories(
  ${PROJECT_SOURCE_DIR}
)


set(ONEAPI_ROOT /opt/intel/oneapi)
set(TBB_ROOT ${ONEAPI_ROOT}/tbb/latest/lib)
list(APPEND CMAKE_PREFIX_PATH ${TBB_ROOT}/cmake)
find_package(TBB REQUIRED)


message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

set(YCSB_SRC_DIR ${CMAKE_SOURCE_DIR}/third_party/ycsb)
set(YCSB_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ycsb-build)
option(RECSTORE_ENABLE_HDRHISTOGRAM_PROGRAMS "Build HdrHistogram programs/tests under third_party/ycsb" OFF)
if(NOT RECSTORE_ENABLE_HDRHISTOGRAM_PROGRAMS)
  # Keep hdr_histogram library build, but skip third-party tests/examples registration in CTest.
  set(HDR_HISTOGRAM_BUILD_PROGRAMS OFF CACHE BOOL "Build tests and examples" FORCE)
endif()
if (EXISTS "${YCSB_SRC_DIR}/CMakeLists.txt")
  add_subdirectory(${YCSB_SRC_DIR})
endif()
