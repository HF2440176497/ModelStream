

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
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/private)
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/util)

set(test_srcs "")
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

if(build_source)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../source/include)
  file(GLOB_RECURSE test_source_srcs ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
  list(APPEND test_srcs ${test_source_srcs})
endif()

if(TARGET modelstream_va AND TARGET gtest)
    add_executable(modelstream_modules_test ${test_srcs})
    target_link_libraries(modelstream_modules_test 
        modelstream_va
        gflags
        glog
        gtest
        gtest
        dl 
        pthread
        rt
    )
else()
    if(NOT TARGET modelstream_va)
        message(STATUS "modelstream_va target not found - skipping modelstream_modules_test")
    endif()
    if(NOT TARGET gtest)
        message(STATUS "gtest target not found - skipping modelstream_modules_test")
    endif()
endif()