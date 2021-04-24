# Copyright (c) 2021 Regents of the University of California
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

set -e
set -x

cmake -B _build . -DCMAKE_INSTALL_PREFIX=${PREFIX}
make -C _build -j $CPU_COUNT

cd _build/python/
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX
