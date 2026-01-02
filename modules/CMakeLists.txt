cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

# ---[ gflags
include(${MODELSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
include_directories(${GFLAGS_INCLUDE_DIRS})

# ---[ glog
include(${MODELSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
include_directories(${GLOG_INCLUDE_DIRS})

# ---[ rapidjson
include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/rapidjson/include)

# ---[ framework
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)

option(build_source "build module source" ON)

if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

# ---[ 3rdparty
set(3RDPARTY_LIBS "")
set(DEPENDENCIES "")

include(${MODELSTREAM_ROOT_PATH}/cmake/have_cnstream_target.cmake)
# ---[ framework
have_framework_target(${MODELSTREAM_ROOT_PATH})
list(APPEND 3RDPARTY_LIBS modelstream_core)
if(HAVE_FRAMEWORK_TARGET)
  list(APPEND DEPENDENCIES modelstream_core)
endif()

# ---[ opencv
find_package(OpenCV REQUIRED COMPONENTS 
    core imgproc highgui features2d imgcodecs videoio)
if(OpenCV_FOUND)
  # 使用手动指定的方式添加
  list(APPEND 3RDPARTY_LIBS ${OpenCV_LIBS})
  include_directories(${OpenCV_INCLUDE_DIRS})
  message(STATUS "OpenCV version: ${OpenCV_VERSION}; Include dirs: ${OpenCV_INCLUDE_DIRS}; Libraries: ${OpenCV_LIBS}")
else()
  message(FATAL_ERROR "OpenCV not found! Please ensure OpenCV is installed and set in CMAKE_PREFIX_PATH")
endif()

# ---[ yuv
# 检查是否从根目录构建了libyuv
if(BUILD_LIBYUV)
  if(TARGET yuv or TARGET yuv_shared)  # 根据 libyuv 内的 cmake 文件确定
    include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
    message(STATUS "Using libyuv library from root build - target yuv")
    if(TARGET yuv)  # 优先使用静态库
      list(APPEND 3RDPARTY_LIBS yuv)
      list(APPEND DEPENDENCIES yuv)
      message(STATUS "Using libyuv static library: yuv")
    elseif(TARGET yuv_shared)
      list(APPEND 3RDPARTY_LIBS yuv_shared)
      list(APPEND DEPENDENCIES yuv_shared)
      message(STATUS "Using libyuv shared library: yuv_shared")
    endif()
  endif()
else()
  message(FATAL_ERROR "libyuv not found. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

set(module_list "")
if(build_source)
  list(APPEND module_list source)
  # install(DIRECTORY source/include/ DESTINATION include)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util/include)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB util_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/*.cpp)
list(APPEND srcs ${util_src})

# 根据NVIDIA_PLATFORM选项决定是否包含CUDA源码
if(NVIDIA_PLATFORM)
  file(GLOB cuda_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cpp)
  list(APPEND srcs ${cuda_src})
  message(STATUS "CUDA support enabled - adding CUDA source files")
endif()

# 递归添加模块的源文件
# 要求每个 module 目录都是 include src 目录结构
foreach(module ${module_list})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${module}/include)
  file(GLOB_RECURSE module_src ${CMAKE_CURRENT_SOURCE_DIR}/${module}/*.cpp)
  list(APPEND srcs ${module_src})
endforeach()

# 项目根目录的 cmake 已经定义了 OUTPUT_DIRECTORY

add_library(modelstream_va SHARED ${srcs})
add_dependencies(modelstream_va ${DEPENDENCIES})
set_target_properties(modelstream_va PROPERTIES LINK_FLAGS_RELEASE -s)

# install(TARGETS modelstream_va LIBRARY DESTINATION lib64)
target_link_libraries(modelstream_va ${3RDPARTY_LIBS} rt dl pthread)

if(build_tests)
  add_subdirectory(unittest)
endif()