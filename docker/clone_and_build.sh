#!/bin/bash

# Clone the repository
git clone https://github.com/cannylab/tsne-cuda.git

# CD into the directory and build
cd tsne-cuda && git checkout docker-build
conda build . -c serge-sans-paille
