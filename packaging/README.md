<!--
 Copyright (c) 2021 Regents of the University of California

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Building

To build the packages for both pip and for anaconda, we first need to compile the sources. For each CUDA version,
we have to compile the sources independently with:

```bash
docker build . -f ./packaging/Dockerfile.cuda10.2 -t tsnecuda-docker-10.2
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-10.2:latest
```

# Anaconda
First, build all of the independent CUDA source files with the above. We can then deploy the conda packages with:
```
conda-build -c pytorch ./packaging/conda/
anaconda upload $PACKAGE
```

# PyPi
First, build all of the independent CUDA source files with the above. We can then deploy the pypi package with:
```
cd build/build_10.2/ && python3 setup.py bdist_wheel
python3 -m twine upload dist/*
```

For the other files, the mirror servers at tsnecuda.isx.ai need to be updated.
