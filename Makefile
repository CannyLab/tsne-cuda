# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_61 -code=sm_61 -ccbin g++ -std=c++11 -g
NVCCFLAGS = -O3 -arch=compute_61 -code=sm_61 -ccbin g++ -std=c++11 -g
LIBS = -lcublas
LDFLAGS = -Wno-deprecated-gpu-targets

TARGETS = tsne_utils

all:	$(TARGETS)

tsne_utils: tsne_utils.o Utilities.o
	$(CC) -o $@ $(NVCCLIBS) $(LIBS) $(LDFLAGS) tsne_utils.o Utilities.o

tsne_utils.o: src/tsne_utils.cu src/Utilities.cuh src/Utilities.cu
	$(CC) -c $(NVCCFLAGS) src/tsne_utils.cu src/Utilities.cu

clean:
	rm -f *.o $(TARGETS)
