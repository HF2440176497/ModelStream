

if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Please create a separate directory for build files.")
endif()

cmake_minimum_required(VERSION 3.13)

project(ModelStream LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_LIBYUV "Build libyuv library" ON)

option(NVIDIA_PLATFORM "NVIDIA platform (Enable CUDA)" OFF)
option(RKNN_PLATFORM "RKNN platform (Enable RKNN)" OFF)

if(NVIDIA_PLATFORM)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_compile_definitions(NVIDIA_PLATFORM)
  message(STATUS "CUDA support enabled - CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
elseif(RKNN_PLATFORM)
  # ...

  add_compile_definitions(RKNN_PLATFORM)
  message(STATUS "RKNN support enabled")
endif()

set(MODELSTREAM_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Top-level directory of this project")

option(build_modules "build modules" ON)
option(build_modules_contrib "build extra modules" OFF)
option(build_samples "build sample programs" OFF)
option(build_tests "build unit tests" ON)
option(build_test_coverage  "Test code coverage" OFF)
option(build_tools "build tools" ON)
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
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/googletest)

  # get_target_property(GTEST_LOCATION gtest LOCATION)
  # message(STATUS "GTest library location: ${GTEST_LOCATION}")
  
  # get_target_property(GTEST_TYPE gtest TYPE)
  # message(STATUS "GTest target type: ${GTEST_TYPE}")

  set(build_gtest_already ON)  # used by subdirectories
endif()

# ---[ libyuv
if(BUILD_LIBYUV)
  set(LIBYUV_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})  # 与项目保持一致
  set(LIBYUV_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(LIBYUV_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libyuv)
  message(STATUS "libyuv library built and integrated")
endif()

# To use sanitizers, the version of GCC is required to be no less than 4.9
option(SANITIZE_MEMORY "Enable MemorySanitizer for sanitized targets." OFF)
option(SANITIZE_ADDRESS "Enable AddressSanitizer for sanitized targets." OFF)
option(SANITIZE_THREAD "Enable ThreadSanitizer for sanitized targets." OFF)
option(SANITIZE_UNDEFINED "Enable UndefinedBehaviorSanitizer for sanitized targets." OFF)

# ---[ Build Flags
if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
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



