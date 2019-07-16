#!/bin/bash

# Build the conda package
echo "Building version 2.1"
conda build . -c serge-sans-paille

# Upload the conda package to anaconda cloud
conda install anaconda-client
anaconda login --username $CONDA_USERNAME --password $CONDA_PASSWORD
anaconda upload --user $CONDA_UPLOAD_USERNAME /home/root/miniconda/conda-bld/noarch/tsnecuda*.tar.bz2 --force

