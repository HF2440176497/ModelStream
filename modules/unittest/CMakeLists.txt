

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

# 检查
if(NOT build_gtest_already)
  add_subdirectory(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest)
  set(build_gtest_already ON)
endif()
include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest/googletest/include)

# 必须包含框架头文件
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include/private)
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include/util)

# 包含CUDA头文件
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# unittest 头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(test_srcs "")
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# util 单元测试文件
file(GLOB test_util_srcs ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp)
list(APPEND test_srcs ${test_util_srcs})

if(NVIDIA_PLATFORM)
  file(GLOB test_util_cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp)
  list(APPEND test_srcs ${test_util_cuda_srcs})
endif()


if(build_source)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/include)
  file(GLOB_RECURSE test_source_srcs ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
  list(APPEND test_srcs ${test_source_srcs})
endif()

# 检查 modelstream_va 
have_modules_target(${MODELSTREAM_ROOT_PATH})

add_executable(modelstream_modules_test ${test_srcs})
add_dependencies(modelstream_modules_test modelstream_va)
target_link_libraries(modelstream_modules_test modelstream_va gflags glog gtest rt dl pthread ${CUDA_LIBRARIES})
