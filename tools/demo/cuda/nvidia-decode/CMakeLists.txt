cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mno-avx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-avx2")


find_package(SDL2 REQUIRED)

add_executable(main
    src/main.cpp
    src/decode.cpp
    src/decode_server.cpp
    src/frame_server.cpp
    src/utils.cpp
)


# 包含FFmpeg头文件目录
target_include_directories(main PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
	/usr/local/ffmpeg/include
    /usr/local/include/opencv4
	${SDL2_INCLUDE_DIRS}
)

# 链接库目录
target_link_directories(main PRIVATE
    /usr/local/lib
	/usr/local/ffmpeg/lib
	${SDL2_LIBRARY_DIRS}
)


target_link_libraries(main
    avformat
    avcodec
    avdevice
    avfilter
    swscale
    swresample
    avutil
    SDL2::SDL2  # 使用 SDL2 的目标
    
    # opencv 
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs

    # basic libs
    bz2
    asound
    z
    m
    pthread
    dl
)
