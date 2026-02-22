
# 单独编译 demo/cuda 下的源文件，临时验证

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..
        CACHE INTERNAL "Top-level directory of this project")
endif()

project(tools_cuda_demo LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "89")

set(3RDPARTY_LIBS "")
set(DEPENDENCIES "")

# ---[ CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND 3RDPARTY_LIBS ${CUDA_LIBRARIES})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

# ---[ NPP (NVIDIA Performance Primitives)
find_path(NPP_INCLUDE_DIR npp.h 
    PATHS ${CUDA_INCLUDE_DIRS}
    PATH_SUFFIXES npp)
if(NPP_INCLUDE_DIR)
    include_directories(${NPP_INCLUDE_DIR})
    message(STATUS "----- NPP include: ${NPP_INCLUDE_DIR}")
endif()

# ---[ opencv
include(${MODELSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND 3RDPARTY_LIBS ${OpenCV_LIBS})
endif()

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

# ---[ libyuv
include_directories(${MODELSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
add_subdirectory(${MODELSTREAM_ROOT_PATH}/3rdparty/libyuv
  ${CMAKE_BINARY_DIR}/bin)
if(TARGET yuv)
  list(APPEND 3RDPARTY_LIBS yuv)
  list(APPEND DEPENDENCIES yuv)
  message(STATUS "----- Using libyuv static library: yuv")
elseif(TARGET yuv_shared)
  list(APPEND 3RDPARTY_LIBS yuv_shared)
  list(APPEND DEPENDENCIES yuv_shared)
  message(STATUS "----- Using libyuv shared library: yuv_shared")
else()
  message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
endif()

# Source files
set(DEMO_SOURCES
    transfmt_cuda_demo.cu
    transfmt_cuda_npp_demo.cu
)

foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    cuda_add_executable(${demo_name} ${demo_source})
    
    target_link_libraries(${demo_name} 
        ${3RDPARTY_LIBS}
        npps
        nppc
        nppial
        nppicc
        nppidei
        nppif
        nppig
        nppim
        nppist
        nppisu
        nppitc
        rt
        dl
        pthread
    )
endforeach()
