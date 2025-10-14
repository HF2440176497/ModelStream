

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

# include source in framework/core
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core/src)

# ---[ gflags
include(${MODELSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
include_directories(${GFLAGS_INCLUDE_DIRS})

# ---[ glog
include(${MODELSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
include_directories(${GLOG_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
# unitest/core/ and test_main.cpp
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core core_test_srcs)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp)

add_executable(cnstream_core_test ${core_test_srcs})
add_dependencies(cnstream_core_test cnstream_core gtest)

