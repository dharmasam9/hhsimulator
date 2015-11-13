CC=nvcc
all:
	$(CC) main.cu hsolve_kernels.cu -o runfile
clean:
	rm runfile
