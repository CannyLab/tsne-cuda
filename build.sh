case ${cuda_version} in
    cuda90) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0
        ;;
    cuda91) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.1
        ;;
    cuda92) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.2
        ;;
    cuda100) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
        ;;
    cuda101) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1
        ;;
    cuda102) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2
        ;;
    cuda110) CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.0
        ;;
    *) echo "cuda version not recognized: {cuda_version}"; exit 1
        ;;
esac

# Add the CUDA executable to the beginning of the path
export PATH=$CUDA_TOOLKIT_ROOT_DIR/bin:$PATH

# Note, for CUDA > 10.x, we have to worry about cublas not being found - we can fix this by linking cublas
# sudo ln -s /usr/lib/x86_64-linux-gnu/libcublas.so.10.2.2.89 /usr/local/cuda-10.2/libcublas.so

# Export the linux toolchain file
CMAKE_PLATFORM_FLAGS+={-DCMAKE_TOOLCHAIN_FILE="${RECIPE_DIR}/cross-linux.cmake"}

# Initialize the repository
git submodule init
git submodule update

# Configure the library
cd ./build
/snap/bin/cmake --version
/snap/bin/cmake .. -DBUILD_PYTHON=TRUE -DWITH_MKL=FALSE -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS[@]} ${SRC_DIR} -DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR} -DCMAKE_CUDA_HOST_COMPILER=${CXX}

# Build the library
make -j12
cd python/

# Install the python files
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cp -r ../lib/* $PREFIX/lib/
