

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

include(${MODELSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)

if(GTEST_FOUND)
  message(STATUS "Using installed GTest for framework unittest")
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  if(NOT build_gtest_already)
    add_subdirectory(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest)
    set(build_gtest_already ON)
  endif()
  include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest/googletest/include)
endif()

# ---[ source: framework unittest
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core core_test_srcs)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

if(TARGET modelstream_core AND (TARGET gtest OR GTEST_FOUND))
    add_executable(modelstream_core_test ${core_test_srcs})
    if(GTEST_FOUND)
      # 使用系统安装的 gtest 库
      target_link_libraries(modelstream_core_test 
          modelstream_core
          gflags
          glog
          ${GTEST_LIBRARIES}
      )
    else()
      target_link_libraries(modelstream_core_test 
          modelstream_core
          gflags
          glog
          gtest
      )
    endif()
else()
    if(NOT TARGET modelstream_core)
        message(STATUS "modelstream_core target not found - skipping modelstream_core_test")
    endif()
    if(NOT TARGET gtest AND NOT GTEST_FOUND)
        message(STATUS "gtest not found - skipping modelstream_core_test")
    endif()
endif()
