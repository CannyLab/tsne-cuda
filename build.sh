export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
git submodule init
git submodule update 
cd ./build
cmake  .. -DBUILD_PYTHON=TRUE -DWITH_MKL=FALSE -DCMAKE_C_COMPILER=gcc-4.9 -DCMAKE_CXX_COMPILER=g++-4.9
pwd
make -j10
make
cd python/
pwd
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cp -r ../lib/* $PREFIX/lib/
