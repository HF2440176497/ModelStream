cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(TOOLS_3RDPARTY_LIBS "")
set(TOOLS_DEPENDENCIES "")

if (build_va)
  include(${MODELSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
  have_modules_target(${MODELSTREAM_ROOT_PATH})
  include_directories(
    ${MODELSTREAM_ROOT_PATH}/modules
    ${MODELSTREAM_ROOT_PATH}/modules/util/include
  )
  if (module_list)
    foreach(module ${module_list})
      include_directories(
        ${MODELSTREAM_ROOT_PATH}/modules/${module}/include
      )
    endforeach()
  endif()
  if(HAVE_MODULES_TARGET)
    list(APPEND TOOLS_DEPENDENCIES modelstream_va)
  endif()
endif()

# ---[ framework
include(${MODELSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${MODELSTREAM_ROOT_PATH})
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)
if(HAVE_FRAMEWORK_TARGET)
  list(APPEND TOOLS_DEPENDENCIES modelstream_core)
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

if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

list(APPEND TOOLS_3RDPARTY_LIBS ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})

# ---[ opencv
include(${MODELSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND TOOLS_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ libyuv
if(NOT has_built_libyuv)
  add_subdirectory(${MODELSTREAM_ROOT_PATH}/3rdparty/libyuv ${CMAKE_CURRENT_BINARY_DIR}/lib/)
  set(has_built_libyuv TRUE)
endif()

if(TARGET yuv)
  list(APPEND TOOLS_3RDPARTY_LIBS yuv)
  list(APPEND TOOLS_DEPENDENCIES yuv)
  # message(STATUS "----- Using libyuv static library: yuv")
elseif(TARGET yuv_shared)
  list(APPEND TOOLS_3RDPARTY_LIBS yuv_shared)
  list(APPEND TOOLS_DEPENDENCIES yuv_shared)
  # message(STATUS "----- Using libyuv shared library: yuv_shared")
else()
  message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
endif()

include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/libyuv/include)

# ---[ Options
option(build_inspect "build cnstream inspect" ON)
option(build_demo "build demo executable file" ON)

if(build_inspect)
  add_subdirectory(inspect)
endif()

if(build_demo)
  add_subdirectory(demo)
endif()

