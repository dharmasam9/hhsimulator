# compiler

CC = nvcc

# comiler flags
CFLAGS = -w

# include directories
INCLUDES = -I./

# libraries
LIBS = -lcusparse

# target
FILE = gpu_hsolve


all:
	$(CC) $(CFLAGS) $(FILE).cu $(INCLUDES) $(LIBS)


