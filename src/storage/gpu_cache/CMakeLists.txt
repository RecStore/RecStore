#
# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.8)
file(GLOB gpu_cache_src
  nv_gpu_cache.cu
  static_table.cu
  static_hash_table.cu
  uvm_table.cu
  gpu_cache_nohash.cu
)


add_library(gpu_cache SHARED ${gpu_cache_src})
# uvm_table.cu uses OpenMP; CUDA compilation does not inherit CMAKE_CXX_FLAGS,
# so the runtime has to be linked at the target.
target_link_libraries(gpu_cache PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(gpu_cache PUBLIC  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>) 
# target_compile_options(gpu_cache PUBLIC  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++11>) 

set_target_properties(gpu_cache PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(gpu_cache PROPERTIES CUDA_ARCHITECTURES OFF)

