#!/bin/bash

set -e
set -x

# Build all of the docker files
docker build . -f ./packaging/Dockerfile.cuda10.1 -t tsnecuda-docker-10.1
docker build . -f ./packaging/Dockerfile.cuda10.2 -t tsnecuda-docker-10.2
docker build . -f ./packaging/Dockerfile.cuda11.0 -t tsnecuda-docker-11.0
docker build . -f ./packaging/Dockerfile.cuda11.1 -t tsnecuda-docker-11.1
docker build . -f ./packaging/Dockerfile.cuda11.2 -t tsnecuda-docker-11.2
docker build . -f ./packaging/Dockerfile.cuda11.3 -t tsnecuda-docker-11.3

# Run all of the docker files to get outputs
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-10.1:latest
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-10.2:latest
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-11.0:latest
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-11.1:latest
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-11.2:latest
docker run -it --mount src=$(realpath ./build),target='/artifacts',type=bind tsnecuda-docker-11.3:latest

# Own all of the files
sudo chown -R $(whoami):$(whoami) ./build

# Run the anaconda packaging and upload
conda-build -c pytorch ./packaging/conda/

# Run the local pip packaging and upload
cd ./build/build_10.1
cat VERSION.txt | awk 'NF{print $0 "+cu101"}' > tmp && mv tmp VERSION.txt
python3 setup.py bdist_wheel
mv dist/*.whl ../../packaging/pypi_extra_hosts/dist/
cd ../../

cd ./build/build_10.2
python3 setup.py bdist_wheel
python3 -m twine upload dist/*
mv dist/*.whl ../../packaging/pypi_extra_hosts/dist/
cd ../../

cd ./build/build_11.0
cat VERSION.txt | awk 'NF{print $0 "+cu110"}' > tmp && mv tmp VERSION.txt
python3 setup.py bdist_wheel
mv dist/*.whl ../../packaging/pypi_extra_hosts/dist/
cd ../../

cd ./build/build_11.1
cat VERSION.txt | awk 'NF{print $0 "+cu111"}' > tmp && mv tmp VERSION.txt
python3 setup.py bdist_wheel
mv dist/*.whl ../../packaging/pypi_extra_hosts/dist/
cd ../../

cd ./build/build_11.2
cat VERSION.txt | awk 'NF{print $0 "+cu112"}' > tmp && mv tmp VERSION.txt
python3 setup.py bdist_wheel
mv dist/*.whl ../../packaging/pypi_extra_hosts/dist/
cd ../../

cd ./build/build_11.3
cat VERSION.txt | awk 'NF{print $0 "+cu113"}' > tmp && mv tmp VERSION.txt
python3 setup.py bdist_wheel
mv dist/*.whl ../../packaging/pypi_extra_hosts/dist/
cd ../../
