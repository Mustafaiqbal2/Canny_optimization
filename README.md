# Canny Edge Detection Optimization

This project implements an optimized version of the Canny edge detection algorithm using CUDA for GPU acceleration. The implementation focuses on minimizing memory transfers, utilizing parallelism effectively, and optimizing kernel execution to achieve significant performance gains.

## Project Overview

Canny edge detection is a popular image processing algorithm that involves several steps:
1. Gaussian smoothing to reduce noise
2. Gradient computation (first derivatives in x and y directions)
3. Non-maximum suppression to thin edges
4. Hysteresis thresholding to identify and connect edges

The original CPU implementation processes each pixel sequentially, making it a perfect candidate for GPU parallelization.

## Optimization Techniques

Several optimization techniques were applied to achieve maximum performance:

### 1. Kernel Fusion
- Combined derivative and magnitude calculations into a single kernel
- Integrated edge initialization and histogram building
- Combined expansion and cleanup phases in hysteresis

### 2. Memory Optimization
- Used pinned memory for input image transfers
- Created a single data pipeline that minimizes host-device transfers
- Allocated all device memory at once rather than in separate function calls

### 3. Parallel Processing
- Utilized shared memory for Gaussian kernels
- Implemented efficient thread block organization
- Used atomic operations for histogram construction

### 4. Hybrid CPU-GPU Processing
- Kept threshold computation on CPU where sequential processing is more efficient
- Performed histogram accumulation on GPU where parallelism helps

## Implementation Phases

1. **Initial Phase**: Converted Gaussian smoothing, Derivative and Magnitude calculations to CUDA (5x speedup)
2. **Second Phase**: Parallelized hysteresis and non-maximal suppression (7x speedup)
3. **Third Phase**: Combined independent kernel functions to reduce overhead
4. **Fourth Phase**: Used pinned memory for host-device transfers
5. **Final Phase**: Created single execution pipeline for all operations
6. **Post-Final Phase**: Moved appropriate operations back to CPU (threshold computations)

## Performance Results

The optimizations led to dramatic performance improvements:

| Image Size | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------|---------------|---------------|---------|
| Large      | 361.83        | 7.54          | 47.97x  |
| Medium     | 82.13         | 3.29          | 25.03x  |
| Small      | 3.63          | 1.06          | 3.45x   |

### Detailed Timing Breakdown

```
Memory: Device allocation: 0.707 ms
Transfer: Image to device: 0.872 ms
Kernel: Gaussian X direction: 0.695 ms
Kernel: Gaussian Y direction: 0.877 ms
Kernel: Derivatives and magnitude: 0.176 ms
Kernel: Non-maximum suppression: 0.302 ms
Kernel: Edge initialization: 0.198 ms
Transfer: Histogram to host: 0.111 ms
CPU: Threshold computation: 0.009 ms
Kernel: Threshold application: 0.140 ms
Kernel: Hysteresis expansion (6 iterations): 0.786 ms
Transfer: Result to host: 2.932 ms
Memory: Device cleanup: 2.750 ms
```

## Usage

To build and run the optimized Canny edge detector:

```bash
# Build the application
make

# Run on a test image
./canny_cuda pics/pic_large.pgm 2.5 0.25 0.5
#              image_file     sigma tlow thigh
```

## Hardware Requirements

- CUDA-compatible GPU (code optimized for SM 7.0 architecture)
- CUDA toolkit installed

## Key Findings

1. Memory transfers between host and device were a significant bottleneck - minimizing these transfers was the most significant single optimization 
2. Using pinned memory for the input image provided was critical for performance
3. Some operations (like threshold computation) were more efficient on the CPU despite the overhead of transferring the histogram
4. The combined pipeline approach reduced overall execution time by eliminating redundant memory operations

The final implementation achieves nearly 50x speedup for large images compared to the CPU version, demonstrating the effectiveness of the applied optimization techniques.