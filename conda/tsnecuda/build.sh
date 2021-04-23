# Copyright (c) 2021 Regents of the University of California
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

set -e
set -x

CMAKE_PLATFORM_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE="${RECIPE_DIR}/cross-linux.cmake")
cmake -B _build . -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS[@]} ${SRC_DIR}
make -C _build -j $CPU_COUNT

cd _build/python/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX
