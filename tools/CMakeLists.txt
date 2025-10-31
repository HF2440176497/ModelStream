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

# ---[ source: framework
include_directories(${MODELSTREAM_ROOT_PATH}/framework/core/include)

# ---[ Options
option(build_inspect "build cnstream inspect" ON)

if(build_inspect)
  add_subdirectory(inspect)
endif()
