NVCC=nvcc
CUDAFLAGS=
all:
	$(NVCC) $(CUDAFLAGS) cuda_quicksort.cu