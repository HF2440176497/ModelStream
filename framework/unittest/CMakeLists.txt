

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

include(${MODELSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
if(GTEST_FOUND)
  message(STATUS "Using installed GTest for framework unittest")
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  if(NOT has_built_gtest)
    add_subdirectory(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest)
    set(has_built_gtest TRUE)
  endif()
  include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/googletest/googletest/include)
endif()

# ---[ source: framework unittest
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core core_test_srcs)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

include(${MODELSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${MODELSTREAM_ROOT_PATH})

if (GTEST_FOUND and HAVE_FRAMEWORK_TARGET)
  add_executable(modelstream_core_test ${core_test_srcs})
  target_include_directories(modelstream_core_test PRIVATE
    ${MODELSTREAM_ROOT_PATH}/framework/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(modelstream_core_test 
      modelstream_core
      ${GTEST_LIBRARIES}
      ${GLOG_LIBRARIES}
      ${GFLAGS_LIBRARIES}
  )
else()

