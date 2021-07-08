#!/bin/bash
set -e

## Build the project
cmake ..
make -j8

# Copy the built source files to the mounted build directory
cp -r python/ /artifacts/build_$1


# Setup version infomation NOTE: This doesn't work on pypi
# cat python/VERSION.txt | awk 'NF{print $0 "+cu10.1"}' > tmp && mv tmp python/VERSION.txt

# Deploy the package to the PyPI
# cd python/ && python3 setup.py bdist_wheel
# python3 -m twine upload --repository testpypi dist/*
