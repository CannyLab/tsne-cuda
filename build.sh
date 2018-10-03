source ~/.bashrc
git submodule init
git submodule update 
cd ./build
cmake  .. -DBUILD_PYTHON=TRUE -DWITH_MKL=FALSE -DCMAKE_C_COMPILER=gcc-4.9 -DCMAKE_CXX_COMPILER=g++-4.9
pwd
make -j5 
make
cd python/
pwd
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cp -r ../lib/* $PREFIX/lib/
