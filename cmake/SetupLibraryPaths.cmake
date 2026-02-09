
# 用于在 Windows 平台下方便地设置第三方库的路径
# 使用方法：在顶层 CMakeLists.txt 中 include 此文件
# 或者通过命令行参数：-DCMAKE_TOOLCHAIN_FILE=cmake/SetupLibraryPaths.cmake

if(WIN32)
    message(STATUS "=== Setting up library paths for Windows ===")
    
    # OpenCV
    if(NOT DEFINED OpenCV_ROOT_DIR)
        set(OpenCV_ROOT_DIR "D:/source/opencv/build/install" CACHE PATH "Folder contains OpenCV")
        message(STATUS "OpenCV_ROOT_DIR: ${OpenCV_ROOT_DIR}")
    else()
        message(STATUS "OpenCV_ROOT_DIR (user defined): ${OpenCV_ROOT_DIR}")
    endif()
    
    # GTest
    if(NOT DEFINED GTEST_ROOT_DIR)
        set(GTEST_ROOT_DIR "D:/source/googletest-1.15.2/install" CACHE PATH "Folder contains Google Test")
        message(STATUS "GTEST_ROOT_DIR: ${GTEST_ROOT_DIR}")
    else()
        message(STATUS "GTEST_ROOT_DIR (user defined): ${GTEST_ROOT_DIR}")
    endif()
    
    # GFlags
    if(NOT DEFINED GFLAGS_ROOT_DIR)
        set(GFLAGS_ROOT_DIR "D:/source/gflags-2.3.0/install" CACHE PATH "Folder contains Gflags")
        message(STATUS "GFLAGS_ROOT_DIR: ${GFLAGS_ROOT_DIR}")
    else()
        message(STATUS "GFLAGS_ROOT_DIR (user defined): ${GFLAGS_ROOT_DIR}")
    endif()
    
    # Glog
    if(NOT DEFINED GLOG_ROOT_DIR)
        set(GLOG_ROOT_DIR "D:/source/glog-0.7.1/install" CACHE PATH "Folder contains Google glog")
        message(STATUS "GLOG_ROOT_DIR: ${GLOG_ROOT_DIR}")
    else()
        message(STATUS "GLOG_ROOT_DIR (user defined): ${GLOG_ROOT_DIR}")
    endif()
    
    message(STATUS "=== Library paths setup complete ===")
else()
    message(STATUS "SetupLibraryPaths.cmake: Not on Windows, skipping automatic path setup")
endif()
