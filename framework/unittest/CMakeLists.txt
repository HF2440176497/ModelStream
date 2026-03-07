

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

include(${VSTREAM_ROOT_PATH}/cmake/FindGtest.cmake)
if(GTEST_FOUND)
  message(STATUS "Using installed GTest for modules unittest")
  include_directories(${GTEST_INCLUDE_DIRS})
else()
  if(NOT has_built_gtest)
    add_subdirectory(${VSTREAM_ROOT_PATH}/3rdparty/googletest)
    set(has_built_gtest TRUE)
  endif()
  include_directories(${VSTREAM_ROOT_PATH}/3rdparty/googletest/googletest/include)
endif()

# ---[ source: framework unittest
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core core_test_srcs)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/base.cpp)
list(APPEND core_test_srcs ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})

if (HAVE_FRAMEWORK_TARGET)
  add_executable(vstream_core_test ${core_test_srcs})
  target_include_directories(vstream_core_test PRIVATE
    ${VSTREAM_ROOT_PATH}/framework/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(GTEST_FOUND)
    target_link_libraries(vstream_core_test 
        vstream_core
        ${GTEST_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
    )
  elseif(has_built_gtest)
        target_link_libraries(vstream_core_test 
        vstream_core
        gtest
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
    )
  endif()

  if(DEFINED VSTREAM_LIB_PATH)
    set_target_properties(vstream_core_test PROPERTIES
        BUILD_RPATH ${VSTREAM_LIB_PATH}
    )
  endif()
endif()

