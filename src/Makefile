APPS := bh
FLAGS := -O3 -arch=sm_61 -g -Xptxas -v
NVCC := nvcc
GCC := g++
CC := $(GCC)

.PHONY: all

all: bh sparse

bh : bh.cu
	$(NVCC) $(FLAGS) -DVARIANT=0 -o $@ $<

sparse : sparse.cu
	$(NVCC) $(FLAGS) -lcusparse -o $@ $<
