

if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Please create a separate directory for build files.")
endif()

cmake_minimum_required(VERSION 3.13)

if(WIN32)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupLibraryPaths.cmake)
endif()


project(ModelStream LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MODELSTREAM_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Top-level directory of this project")

option(build_modules "build modules" OFF)
option(build_modules_contrib "build extra modules" OFF)
option(build_samples "build sample programs" OFF)
option(build_tests "build unit tests" ON)
option(build_tools "build tools" OFF)
option(build_python_api "build python api" OFF)


# 静态库 and 动态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${MODELSTREAM_ROOT_PATH}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MODELSTREAM_ROOT_PATH}/lib)

# 可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${MODELSTREAM_ROOT_PATH}/bin)

# add macro definitions
if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()


if(build_tests)
  # ---[ gtest
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGtest.cmake)
  
  if(GTEST_FOUND)
    message(STATUS "Found installed GTest (include: ${GTEST_INCLUDE_DIRS}, library: ${GTEST_LIBRARIES})")
    set(build_gtest_already ON)
  else()
    message(STATUS "GTest not found, building from 3rdparty/googletest")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest)
    set(build_gtest_already ON)
  endif()
endif()

add_subdirectory(framework)
if(build_modules)
  add_subdirectory(modules)
endif()
if(build_modules_contrib)
  add_subdirectory(modules_contrib)
endif()
if(build_samples)
  add_subdirectory(samples)
endif()
if(build_tools)
  add_subdirectory(tools)
endif()

