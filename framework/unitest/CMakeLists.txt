

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


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
# unitest/core/, base.cpp, main.cpp
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core core_test_srcs)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

if(TARGET modelstream_core AND TARGET gtest)
    add_executable(modelstream_core_test ${core_test_srcs})
    target_link_libraries(modelstream_core_test 
        modelstream_core
        gflags
        glog
        gtest
    )
else()
    if(NOT TARGET modelstream_core)
        message(STATUS "modelstream_core target not found - skipping modelstream_core_test")
    endif()
    if(NOT TARGET gtest)
        message(STATUS "gtest target not found - skipping modelstream_core_test")
    endif()
endif()
