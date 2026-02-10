cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED MODELSTREAM_ROOT_PATH)
    set(MODELSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..
        CACHE INTERNAL "Top-level directory of this project")
endif()

file(GLOB DEMO_SOURCES "*.cpp")

foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    add_executable(${demo_name} ${demo_source})
    add_dependencies(${demo_name} ${DEPENDENCIES})
    target_link_libraries(${demo_name} ${DEPENDENCIES} ${3RDPARTY_LIBS} rt dl pthread)
endforeach()
