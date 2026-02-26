cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

# ---[ gflags
include(${MODELSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for modules unittest")
  include_directories(${GFLAGS_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${MODELSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for modules unittest")
  include_directories(${GLOG_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ nlohmann/json
include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ framework
include(${MODELSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${MODELSTREAM_ROOT_PATH})
if(HAVE_FRAMEWORK_TARGET)
  include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)
  list(APPEND DEPENDENCIES modelstream_core)
else()
  message(FATAL_ERROR "framework target not found")
endif()

if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

option(build_util "build module util" ON)
option(build_source "build module source" ON)
option(build_inference "build module inference" OFF)
option(build_va "build modulestream_va" ON)

# ---[ 3rdparty
set(MODULES_3RDPARTY_LIBS "")
set(MODULES_DEPENDENCIES "")

list(APPEND MODULES_3RDPARTY_LIBS ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES})

# ---[ opencv
include(${MODELSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND MODULES_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ FFmpeg
include(${MODELSTREAM_ROOT_PATH}/cmake/FindFFmpeg.cmake)
if(FFMPEG_FOUND)
  list(APPEND MODULES_3RDPARTY_LIBS ${FFMPEG_LIBRARIES})
  include_directories(${FFMPEG_INCLUDE_DIR})
  if(FFMPEG_ROOT_DIR)
    link_directories(${FFMPEG_ROOT_DIR}/lib)
  endif()
  message(STATUS "----- FFmpeg found Includes: ${FFMPEG_INCLUDE_DIR} Libraries: ${FFMPEG_LIBRARIES}")
else()
  message(FATAL_ERROR "FFmpeg not found! Please ensure FFmpeg is installed and set FFMPEG_ROOT_DIR")
endif()

# ---[ libyuv
if(has_built_libyuv)
  if(TARGET yuv)  # 根据 libyuv 内的 cmake 文件确定
    list(APPEND MODULES_3RDPARTY_LIBS yuv)
    list(APPEND MODULES_DEPENDENCIES yuv)
    # message(STATUS "----- Using libyuv static library: yuv")
  elseif(TARGET yuv_shared)
    list(APPEND MODULES_3RDPARTY_LIBS yuv_shared)
    list(APPEND MODULES_DEPENDENCIES yuv_shared)
    # message(STATUS "----- Using libyuv shared library: yuv_shared")
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

set(module_list "")
if(build_source)
  list(APPEND module_list source)
endif()

if(build_inference)
  list(APPEND module_list inference)
endif()

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB util_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/*.cpp)

if(NVIDIA_PLATFORM)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  file(GLOB util_cuda_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cu)
  list(APPEND util_src ${util_cuda_src})
  
  list(APPEND MODULES_3RDPARTY_LIBS 
    CUDA::cudart
    CUDA::nvml
  )
  list(APPEND MODULES_3RDPARTY_LIBS         
    CUDA::npps
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppidei
    CUDA::nppif
    CUDA::nppig
    CUDA::nppim
    CUDA::nppist
    CUDA::nppisu
    CUDA::nppitc
  )
    
  message(STATUS "NVIDIA platform enabled - adding CUDA sources")
endif()

list(APPEND srcs ${util_src})

# 递归添加模块的源文件: 要求每个 module 目录都是 include src 目录结构
foreach(module ${module_list})
  file(GLOB_RECURSE module_src ${CMAKE_CURRENT_SOURCE_DIR}/${module}/*.cpp)
  list(APPEND srcs ${module_src})
endforeach()


if(build_util)
  set(util_obj_srcs "")
  file(GLOB util_common_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/*.cpp)
  list(APPEND util_obj_srcs ${util_common_src})

  if(NVIDIA_PLATFORM)
    file(GLOB util_cuda_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cpp 
          ${CMAKE_CURRENT_SOURCE_DIR}/util/src/cuda/*.cu)
    list(APPEND util_obj_srcs ${util_cuda_src})
    message(STATUS "Util module: NVIDIA platform sources added")
  endif()

  add_library(modules_util OBJECT ${util_obj_srcs})
  target_include_directories(modules_util PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/util/include
  )
  add_dependencies(modules_util ${MODULES_DEPENDENCIES})
  target_link_libraries(modules_util ${MODULES_DEPENDENCIES} ${MODULES_3RDPARTY_LIBS} rt dl pthread)
  add_custom_target(util DEPENDS modules_util)
  message(STATUS "Util module enabled - targets: modules_util, util")
endif()

if(build_va)
  add_library(modelstream_va SHARED ${srcs})
  target_include_directories(modelstream_va PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/util/include
  )

  foreach(module ${module_list})
    target_include_directories(modelstream_va PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${module}/include
    )
  endforeach()

  add_dependencies(modelstream_va ${MODULES_DEPENDENCIES})
  target_link_libraries(modelstream_va ${MODULES_DEPENDENCIES} ${MODULES_3RDPARTY_LIBS} rt dl pthread)
  add_custom_target(va DEPENDS modelstream_va)
  # set_target_properties(modelstream_va PROPERTIES LINK_FLAGS_RELEASE -s)
  message(STATUS "modelstream_va enabled - targets: modelstream_va")
endif()

if(build_tests)
  add_subdirectory(unittest)
endif()