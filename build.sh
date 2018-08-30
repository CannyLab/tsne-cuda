git submodule init
git submodule update 
cd ./build
cmake  .. -DBUILD_PYTHON=TRUE
make -j4
make -j4
make
make
cd python/
pip install -e .
