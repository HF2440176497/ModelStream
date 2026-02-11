# - Try to find GTest
#
# The following variables are optionally searched for defaults
#  GTEST_ROOT_DIR:            Base directory where all GTEST components are found
#
# The following are set after configuration is done:
#  GTEST_FOUND
#  GTEST_INCLUDE_DIRS
#  GTEST_LIBRARIES
#  GTEST_LIBRARY_DIRS
#  GMOCK_FOUND
#  GMOCK_INCLUDE_DIRS
#  GMOCK_LIBRARIES

include(FindPackageHandleStandardArgs)

set(GTEST_ROOT_DIR "" CACHE PATH "Folder contains Google Test")

if(GTEST_ROOT_DIR)
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES include)
else()
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
endif()

if(GTEST_ROOT_DIR)
    find_path(GMOCK_INCLUDE_DIR gmock/gmock.h
        PATHS ${GTEST_ROOT_DIR}
        PATH_SUFFIXES include)
else()
    find_path(GMOCK_INCLUDE_DIR gmock/gmock.h)
endif()

if(WIN32)
    if(GTEST_ROOT_DIR)
        find_library(GTEST_LIBRARY
            NAMES gtest gtest_static libgtest libgtest_static
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)

        find_library(GTEST_MAIN_LIBRARY
            NAMES gtest_main gtest_main_static libgtest_main libgtest_main_static
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)

        find_library(GMOCK_LIBRARY
            NAMES gmock gmock_static libgmock libgmock_static
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)

        find_library(GMOCK_MAIN_LIBRARY
            NAMES gmock_main gmock_main_static libgmock_main libgmock_main_static
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)
    else()
        find_library(GTEST_LIBRARY
            NAMES gtest gtest_static libgtest libgtest_static)

        find_library(GTEST_MAIN_LIBRARY
            NAMES gtest_main gtest_main_static libgtest_main libgtest_main_static)

        find_library(GMOCK_LIBRARY
            NAMES gmock gmock_static libgmock libgmock_static)

        find_library(GMOCK_MAIN_LIBRARY
            NAMES gmock_main gmock_main_static libgmock_main libgmock_main_static)
    endif()
else()
    if(GTEST_ROOT_DIR)
        find_library(GTEST_LIBRARY
            NAMES gtest
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)

        find_library(GTEST_MAIN_LIBRARY
            NAMES gtest_main
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)

        find_library(GMOCK_LIBRARY
            NAMES gmock
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)

        find_library(GMOCK_MAIN_LIBRARY
            NAMES gmock_main
            PATHS ${GTEST_ROOT_DIR}
            PATH_SUFFIXES lib lib64)
    else()
        find_library(GTEST_LIBRARY gtest)
        find_library(GTEST_MAIN_LIBRARY gtest_main)
        find_library(GMOCK_LIBRARY gmock)
        find_library(GMOCK_MAIN_LIBRARY gmock_main)
    endif()
endif()

find_package_handle_standard_args(GTest DEFAULT_MSG GTEST_INCLUDE_DIR GTEST_LIBRARY)

# just link to gtest，not gtest_main
if(GTEST_FOUND)
    set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
    set(GTEST_LIBRARIES ${GTEST_LIBRARY})  
    message(STATUS "Found gtest  (include: ${GTEST_INCLUDE_DIR}, library: ${GTEST_LIBRARY})")
    mark_as_advanced(GTEST_LIBRARY GTEST_MAIN_LIBRARY GTEST_INCLUDE_DIR GTEST_ROOT_DIR)
endif()

if(GMOCK_INCLUDE_DIR AND GMOCK_LIBRARY)
    set(GMOCK_FOUND TRUE)
    set(GMOCK_INCLUDE_DIRS ${GMOCK_INCLUDE_DIR})
    set(GMOCK_LIBRARIES ${GMOCK_LIBRARY})
    message(STATUS "Found gmock  (include: ${GMOCK_INCLUDE_DIR}, library: ${GMOCK_LIBRARY})")
    mark_as_advanced(GMOCK_LIBRARY GMOCK_MAIN_LIBRARY GMOCK_INCLUDE_DIR)
endif()
