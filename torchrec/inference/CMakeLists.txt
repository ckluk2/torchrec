# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(inference)


# dependencies
find_package(Torch REQUIRED)

include_directories(${Torch_INCLUDE_DIRS})
add_library( fbgemm SHARED IMPORTED GLOBAL )

set_target_properties(fbgemm PROPERTIES IMPORTED_LOCATION /home/paulzhan/miniconda3/envs/inference/lib/python3.9/site-packages/fbgemm_gpu/fbgemm_gpu_py.so)

set(CMAKE_CXX_STANDARD 17)


# inference library

# for our own header files
include_directories(include/)
include_directories(gen/)


# server
add_executable(server2 server2.cpp)
target_link_libraries(server2
  "${TORCH_LIBRARIES}"
  fbgemm)