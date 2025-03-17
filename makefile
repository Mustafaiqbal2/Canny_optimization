# Compiler settings
NVCC = nvcc
CC = gcc
NVCC_FLAGS = -O3 -arch=sm_70 -Xcompiler -Wall
CC_FLAGS = -O3 -Wall

# If you don't know your GPU architecture, comment out the line above and uncomment this one:
# NVCC_FLAGS = -O3 -arch=compute_50 -code=sm_50 -Xcompiler -Wall

# Libraries
CUDA_LIBS = -lcudart

# Source files
CUDA_SRCS = optimized_canny.cu
C_SRCS = 
OBJ_FILES = optimized_pgm_io.o

# Output executable
TARGET = canny_cuda

# Default image for testing
# PIC=pics/pic_small.pgm
# PIC=pics/pic_medium.pgm
PIC=pics/pic_large.pgm

# Build rules
all: $(TARGET)

$(TARGET): $(CUDA_SRCS) $(OBJ_FILES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(CUDA_LIBS)
optimized_pgm_io.o: optimized_pgm_io.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
run: $(TARGET)
	./$(TARGET) $(PIC) 2.5 0.25 0.5
# 			      sigma tlow thigh

benchmark: $(TARGET)
	@echo "==============================================="
	@echo "Running benchmark tests..."
	@echo "==============================================="
	@for i in 1 2 3; do \
	    echo "Run $$i:"; \
	    ./$(TARGET) $(PIC) 2.5 0.25 0.5; \
	done
	@echo "==============================================="
	@echo "Benchmark complete."
	@echo "==============================================="

profile: $(TARGET)
	nvprof --metrics achieved_occupancy,sm_efficiency ./$(TARGET) $(PIC) 2.5 0.25 0.5

memory-profile: $(TARGET)
	nvprof --print-gpu-trace ./$(TARGET) $(PIC) 2.5 0.25 0.5

visual-profile: $(TARGET)
	nvvp ./$(TARGET) $(PIC) 2.5 0.25 0.5

clean:
	rm -f $(TARGET) *.o *.pgm *.fim *.nvvp

.PHONY: all run benchmark profile memory-profile visual-profile clean