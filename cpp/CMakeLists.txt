cmake_minimum_required(VERSION 3.18)
project(adsrank_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Point at an extracted onnxruntime-linux-x64-<ver> release tarball.
if(NOT DEFINED ORT_ROOT)
  set(ORT_ROOT "$ENV{ORT_ROOT}")
endif()
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "set ORT_ROOT to an onnxruntime release directory")
endif()

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${ORT_ROOT}/include)
target_link_libraries(bench PRIVATE ${ORT_ROOT}/lib/libonnxruntime.so)
target_compile_options(bench PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(bench PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib)
