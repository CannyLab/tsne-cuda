APPS := bh
FLAGS := -O3 -arch=sm_61 -g -Xptxas -v
NVCC := nvcc
GCC := g++
CC := $(GCC)

.PHONY: all

all: bh

bh : bh.cu
	$(NVCC) $(FLAGS) -DVARIANT=0 -o $@ $<
