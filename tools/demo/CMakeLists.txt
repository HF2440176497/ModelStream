cmake_minimum_required(VERSION 3.13)

set(DEMO_SOURCES
    json_parse_demo.cpp
    libyuv_demo.cpp
)

if(NVIDIA_PLATFORM)
    list(APPEND DEMO_SOURCES cuda/transfmt_cuda_demo.cpp)
endif()

foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    add_executable(${demo_name} ${demo_source})
    add_dependencies(${demo_name} ${DEPENDENCIES})
    target_link_libraries(${demo_name} ${DEPENDENCIES} ${3RDPARTY_LIBS} rt dl pthread)
endforeach()
