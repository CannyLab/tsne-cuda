# - Try to find FAISS
#
# The following variables are optionally searched for defaults
#  FAISS_ROOT_DIR:            Base directory where all FAISS components are found
#
# The following are set after configuration is done:
#  FAISS_FOUND
#  FAISS_INCLUDE_DIRS
#  FAISS_LIBRARIES
#  FAISS_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

if(NOT DEFINED FAISS_ROOT_DIR)
    set(FAISS_ROOT_DIR "" CACHE PATH "Folder contains FAISS")
endif()

# We are testing only a couple of files in the include directories
find_path(FAISS_INCLUDE_DIR faiss/Index.h PATHS "${FAISS_ROOT_DIR}/include")
find_path(FAISS_GPU_INCLUDE_DIR faiss/gpu/GpuIndex.h PATHS "${FAISS_ROOT_DIR}/include")
find_library(FAISS_LIBRARY faiss PATHS "${FAISS_ROOT_DIR}/lib")

find_package_handle_standard_args(FAISS DEFAULT_MSG FAISS_INCLUDE_DIR FAISS_GPU_INCLUDE_DIR FAISS_LIBRARY)

if(FAISS_FOUND)
    set(FAISS_INCLUDE_DIRS ${FAISS_INCLUDE_DIR})
    set(FAISS_LIBRARIES ${FAISS_LIBRARY})
    message(STATUS "Found FAISS  (include: ${FAISS_INCLUDE_DIR}, library: ${FAISS_LIBRARY})")
    mark_as_advanced(FAISS_LIBRARY_DEBUG FAISS_LIBRARY_RELEASE
                     FAISS_LIBRARY FAISS_INCLUDE_DIR FAISS_ROOT_DIR)
endif()
