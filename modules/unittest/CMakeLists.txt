

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
  set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
      CACHE INTERNAL "Top-level directory of this project")
endif()

include(${MODELSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
if(GTEST_FOUND)
  message(STATUS "Using installed GTest for modules unittest")
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  if(NOT has_built_gtest)
    add_subdirectory(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest)
    set(has_built_gtest TRUE)
  endif()
  include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest/googletest/include)
endif()


# 必须包含框架头文件
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include/private)
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include/util)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../util/include)  # modules/util 头文件

# unittest 头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(test_srcs "")
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# util 单元测试文件
file(GLOB test_util_srcs ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp)
list(APPEND test_srcs ${test_util_srcs})

# util 单元测试文件 [平台特定的单元测试文件]
if(NVIDIA_PLATFORM)
  file(GLOB test_util_cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/util/cuda/*.cpp)
  list(APPEND test_srcs ${test_util_cuda_srcs})
  message(STATUS "Unittest: NVIDIA platform test sources added")
endif()

include(${MODELSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${MODELSTREAM_ROOT_PATH})
have_modules_target(${MODELSTREAM_ROOT_PATH})

if(HAVE_FRAMEWORK_TARGET AND HAVE_MODULES_TARGET)
  foreach(module ${module_list})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../${module}/include)  # modules/$module_name/include
    file(GLOB_RECURSE test_${module}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${module}/*.cpp)
    list(APPEND test_srcs ${test_${module}_srcs})
  endforeach()
else()
  message(FATAL_ERROR "modules target not found")
endif()

add_executable(modelstream_modules_test ${test_srcs})
add_dependencies(modelstream_modules_test modelstream_va modelstream_core)

set(test_link_libs modelstream_va modelstream_core)
if(NVIDIA_PLATFORM)
  list(APPEND test_link_libs ${CUDA_LIBRARIES})
endif()

if(GTEST_FOUND)
  target_link_libraries(modelstream_modules_test ${test_link_libs} ${GTEST_LIBRARIES} glog gflags rt dl pthread)
else()
  target_link_libraries(modelstream_modules_test ${test_link_libs} gtest glog gflags rt dl pthread)
endif()
