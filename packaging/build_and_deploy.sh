#!/bin/bash
set -e

## Build the project
cmake ..
make -j8

# Copy the built source files to the mounted build directory
cp -r python/ /artifacts/build_$1
