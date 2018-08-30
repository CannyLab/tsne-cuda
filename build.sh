git submodule init
git submodule update 
cd ./build
cmake  .. -DBUILD_PYTHON=TRUE
cd python/
pip install -e .
