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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src core_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src/private private_srcs)


add_library(modelstream_core SHARED ${core_srcs} ${private_srcs})
target_link_libraries(modelstream_core dl pthread ${GFLAGS_LIBRARIES})
set_target_properties(modelstream_core PROPERTIES LINK_FLAGS_RELEASE -s)


# install



# unitest
if(build_tests)
  add_subdirectory(unitest)
endif()
