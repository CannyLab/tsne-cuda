# - Try to find GTEST
#
# The following variables are optionally searched for defaults
#  GTEST_ROOT_DIR:            Base directory where all GTEST components are found
#
# The following are set after configuration is done:
#  GTEST_FOUND
#  GTEST_INCLUDE_DIRS
#  GTEST_LIBRARIES
#  GTEST_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(GTEST_ROOT_DIR "" CACHE PATH "Folder contains GTest")

# We are testing only a couple of files in the include directories
find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS "${GTEST_ROOT_DIR}/include")
find_library(GTEST_LIBRARY gtest PATHS "${GTEST_ROOT_DIR}/lib")

find_package_handle_standard_args(GTest DEFAULT_MSG GTEST_INCLUDE_DIR GTEST_LIBRARY)


if(GTEST_FOUND)
    set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
    set(GTEST_LIBRARIES ${GTEST_LIBRARY})
    message(STATUS "Found gtest  (include: ${GTEST_INCLUDE_DIR}, library: ${GTEST_LIBRARY})")
    mark_as_advanced(GTEST_LIBRARY_DEBUG GTEST_LIBRARY_RELEASE
                     GTEST_LIBRARY GTEST_INCLUDE_DIR GTEST_ROOT_DIR)
endif()
