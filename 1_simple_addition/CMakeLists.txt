cmake_minimum_required(VERSION 3.14)

project(addition
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD			 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS		OFF)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(exec
  src/main.cu
)
