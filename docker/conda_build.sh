#!/bin/bash

# Build the conda package
conda build . -c serge-sans-paille

# Upload the conda package to anaconda cloud
conda install anaconda-client
anaconda login --username $CONDA_USERNAME --password $CONDA_PASSWORD
anaconda upload /home/root/miniconda3/miniconda/conda-bld/linux-64/tsnecuda*.tar.bz2

