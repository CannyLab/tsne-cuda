# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_61 -code=sm_61 -std=c++11
NVCCFLAGS = -O3 -arch=compute_61 -code=sm_61 -std=c++11
NVCCLIBS = 
LDFLAGS = -Wno-deprecated-gpu-targets

TARGETS = dist
all:	$(TARGETS)
dist: dist.o
	$(CC) -o $@ $(NVCCLIBS) $(LDFLAGS) dist.o
dist.o: dist.cu
	$(CC) -c $(NVCCFLAGS) dist.cu
clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
