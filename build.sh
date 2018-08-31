git submodule init
git submodule update 
cd ./build
cmake  .. -DBUILD_PYTHON=TRUE -DWITH_MKL=FALSE -DCMAKE_C_COMPILER=gcc-4.9 -DCMAKE_CXX_COMPILER=g++-4.9
make -j4
cd python/
pip install -e .
