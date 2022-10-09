cmake_minimum_required(VERSION 3.18.0)
# project(dltensor)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # set -fPIC

option(ARM_ENABLE "Enable Arm" OFF)
if(ARM_ENABLE)
  set(CMAKE_BUILD_PLATFORM MACARM)
  add_definitions(-DEX_PLATFORM_ARM)
  # include(${CMAKE_SOURCE_DIR}/cmake/arm_defs.cmake)
endif()

include_directories(include/)
set(SRCS include/dltensor/builtin_fp16.h include/dltensor/tensor.h src/tensor.cc)

add_library(dltensor STATIC ${SRCS})
