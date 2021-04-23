<!--
 Copyright (c) 2021 Regents of the University of California

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Building/Packaging for Anaconda

## Step 1 - Build the docker images

```
docker build . -f ./conda/Dockerfile.cuda11.0 -t tsnecuda-docker-11.0
docker run -i -t tsnecuda-docker-11.0:latest
```

## Step 2 - Build the conda package

```
conda-build ./tsnecuda/ --variants '{ "cudatoolkit": "11.0","python":"3.8" }' -c pytorch -c conda-forge
```

```
conda debug ./tsnecuda/ --variants '{ "cudatoolkit": "11.0","python":"3.8" }' -c pytorch -c conda-forge
```
