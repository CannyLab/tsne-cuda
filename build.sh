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

CMAKE_PLATFORM_FLAGS+={-DCMAKE_TOOLCHAIN_FILE="${RECIPE_DIR}/cross-linux.cmake"}
git submodule init
git submodule update
cd ./build
cmake  .. -DBUILD_PYTHON=TRUE -DWITH_MKL=FALSE -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS[@]} ${SRC_DIR} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} -DCMAKE_C_COMPILER=gcc-4.9 -DCMAKE_CXX_COMPILER=g++-4.9
pwd
make
make
cd python/
pwd
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cp -r ../lib/* $PREFIX/lib/
