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
include(${MODELSTREAM_ROOT_PATH}/cmake/have_cnstream_target.cmake)
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)


if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

option(build_source "build module source" OFF)
option(build_inference "build module inference" OFF)
option(build_util "build module util" ON)
option(build_va "build modulestream_va" OFF)

# ---[ 3rdparty
set(3RDPARTY_LIBS "")
set(DEPENDENCIES "")

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

set(module_list "")
if(build_source)
  list(APPEND module_list source)
endif()

if(build_inference)
  list(APPEND module_list inference)
endif()

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB util_src ${CMAKE_CURRENT_SOURCE_DIR}/util/src/*.cpp)
list(APPEND srcs ${util_src})

# 递归添加模块的源文件: 要求每个 module 目录都是 include src 目录结构
foreach(module ${module_list})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${module}/include)
  file(GLOB_RECURSE module_src ${CMAKE_CURRENT_SOURCE_DIR}/${module}/*.cpp)
  list(APPEND srcs ${module_src})
endforeach()


if(build_util)
  add_library(modules_util OBJECT ${util_src})
  target_include_directories(modules_util PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/util
  )    
  add_custom_target(util DEPENDS modules_util)
  message(STATUS "Util module enabled - targets: modules_util, util")
endif()

if(build_va)
  add_library(modelstream_va SHARED ${srcs})
  target_include_directories(modelstream_va PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/util
  )
  add_dependencies(modelstream_va ${DEPENDENCIES})
  target_link_libraries(modelstream_va ${3RDPARTY_LIBS} rt dl pthread)
  set_target_properties(modelstream_va PROPERTIES LINK_FLAGS_RELEASE -s)
  message(STATUS "modelstream_va enabled - targets: modelstream_va")
endif()

if(build_tests)
  add_subdirectory(unittest)
endif()