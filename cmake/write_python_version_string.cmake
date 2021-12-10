
# Write the python version string to __init__.py
#set(PYTHON_VERSION "\n\n__version__ = '${VERSION_STRING}.dev${BUILD_NUMBER}'\n")
set(PYTHON_VERSION "\n\n__version__ = '${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}'\n")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/python/tsnecuda/__init__.py" ${PYTHON_VERSION})
