/*******************************************************************************
* FILE: hysteresis.c
* This code was re-written by Mike Heath from original code obtained indirectly
* from Michigan State University. heath@csee.usf.edu (Re-written in 1996).
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define VERBOSE 0

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

// Device constants for edge definitions
__device__ const unsigned char D_NOEDGE = 255;
__device__ const unsigned char D_POSSIBLE_EDGE = 128;
__device__ const unsigned char D_EDGE = 0;

// Step 1: Initialize edge map and compute histogram
__global__ void init_edges_kernel(unsigned char *d_nms, unsigned char *d_edge, 
                                 short *d_mag, int *d_hist, int rows, int cols) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = r * cols + c;
    
    if (r < rows && c < cols) {
        // Initialize edge map
        if (d_nms[pos] == D_POSSIBLE_EDGE) {
            d_edge[pos] = D_POSSIBLE_EDGE;
            // Contribute to histogram using atomic add
            if (d_mag[pos] < 32768) {
                atomicAdd(&d_hist[d_mag[pos]], 1);
            }
        } else {
            d_edge[pos] = D_NOEDGE;
        }
        
        // Mark borders as NOEDGE
        if (r == 0 || r == rows-1 || c == 0 || c == cols-1) {
            d_edge[pos] = D_NOEDGE;
        }
    }
}

// Step 2: Apply threshold to mark strong edges
__global__ void apply_threshold_kernel(unsigned char *d_edge, short *d_mag, 
                                     int highthreshold, int rows, int cols) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = r * cols + c;
    
    if (r < rows && c < cols) {
        if ((d_edge[pos] == D_POSSIBLE_EDGE) && (d_mag[pos] >= highthreshold)) {
            d_edge[pos] = D_EDGE;  // Mark strong edges
        }
    }
}

// Step 3: Propagate edges - replaces recursive follow_edges function
__global__ void hysteresis_expansion_kernel(unsigned char *d_edge, short *d_mag, 
                                         int lowthreshold, int rows, int cols,
                                         int *d_changes)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Skip borders
    if (r < 1 || r >= rows-1 || c < 1 || c >= cols-1) {
        return;
    }
    
    int pos = r * cols + c;
    
    // Look at neighboring pixels only if this is a confirmed edge
    if (d_edge[pos] == D_EDGE) {
        // Check all 8 neighbors
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue; // Skip self
                
                int npos = (r+i) * cols + (c+j);
                if (d_edge[npos] == D_POSSIBLE_EDGE && d_mag[npos] >= lowthreshold) {
                    d_edge[npos] = D_EDGE;
                    *d_changes = 1; // Indicates we made a change
                }
            }
        }
    }
}

// Step 4: Clean up by converting remaining POSSIBLE_EDGE to NOEDGE
__global__ void cleanup_edges_kernel(unsigned char *d_edge, int rows, int cols)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = r * cols + c;
    
    if (r < rows && c < cols) {
        if (d_edge[pos] == D_POSSIBLE_EDGE) {
            d_edge[pos] = D_NOEDGE;
        }
    }
}

// Host function
void apply_hysteresis_cuda(short int *h_mag, unsigned char *h_nms, int rows, int cols,
                          float tlow, float thigh, unsigned char *h_edge)
{
    // Allocate device memory
    short *d_mag;
    unsigned char *d_nms, *d_edge;
    int *d_hist, *d_changes;
    int h_hist[32768] = {0};
    
    cudaMalloc((void**)&d_mag, rows * cols * sizeof(short));
    cudaMalloc((void**)&d_nms, rows * cols * sizeof(unsigned char));
    cudaMalloc((void**)&d_edge, rows * cols * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist, 32768 * sizeof(int));
    cudaMalloc((void**)&d_changes, sizeof(int));
    
    // Initialize device memory
    cudaMemset(d_hist, 0, 32768 * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_mag, h_mag, rows * cols * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nms, h_nms, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                 (rows + blockSize.y - 1) / blockSize.y);
    
    // Step 1: Initialize edges and compute histogram
    init_edges_kernel<<<gridSize, blockSize>>>(d_nms, d_edge, d_mag, d_hist, rows, cols);
    
    // Copy histogram back to host for threshold calculation
    cudaMemcpy(h_hist, d_hist, 32768 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Compute thresholds on host (similar to original code)
    int numedges = 0, maximum_mag = 0;
    for (int r = 1; r < 32768; r++) {
        if (h_hist[r] != 0) maximum_mag = r;
        numedges += h_hist[r];
    }
    
    int highcount = (int)(numedges * thigh + 0.5);
    int r = 1;
    numedges = h_hist[1];
    while ((r < (maximum_mag-1)) && (numedges < highcount)) {
        r++;
        numedges += h_hist[r];
    }
    
    int highthreshold = r;
    int lowthreshold = (int)(highthreshold * tlow + 0.5);
    
    if (VERBOSE) {
        printf("The input low and high fractions of %f and %f computed to\n", tlow, thigh);
        printf("magnitude of the gradient threshold values of: %d %d\n", lowthreshold, highthreshold);
    }
    
    // Step 2: Apply threshold to mark strong edges
    apply_threshold_kernel<<<gridSize, blockSize>>>(d_edge, d_mag, highthreshold, rows, cols);
    
    // Step 3: Iterative hysteresis expansion (replaces recursive follow_edges)
    int h_changes = 1;
    while (h_changes) {
        h_changes = 0;
        cudaMemcpy(d_changes, &h_changes, sizeof(int), cudaMemcpyHostToDevice);
        
        hysteresis_expansion_kernel<<<gridSize, blockSize>>>(d_edge, d_mag, lowthreshold, rows, cols, d_changes);
        
        // Check if we made any changes
        cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Step 4: Clean up - convert remaining possible edges to no edges
    cleanup_edges_kernel<<<gridSize, blockSize>>>(d_edge, rows, cols);
    
    // Copy result back to host
    cudaMemcpy(h_edge, d_edge, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_mag);
    cudaFree(d_nms);
    cudaFree(d_edge);
    cudaFree(d_hist);
    cudaFree(d_changes);
}


__global__ void non_max_supp_kernel(short *d_mag, short *d_gradx, short *d_grady, 
                                  int nrows, int ncols, unsigned char *d_result)
{
    // Calculate global position for this thread
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = r * ncols + c;
    
    // Skip border pixels and out-of-bounds pixels
    if (r < 1 || r >= nrows-1 || c < 1 || c >= ncols-1) {
        if (r < nrows && c < ncols) {
            d_result[pos] = (unsigned char)0;  // Zero the border
        }
        return;
    }
    
    // Get the magnitude value at this pixel
    short m00 = d_mag[pos];
    
    // If magnitude is zero, this is not an edge
    if (m00 == 0) {
        d_result[pos] = (unsigned char)NOEDGE;
        return;
    }
    
    // Calculate perpendicular direction to gradient
    short gx = d_gradx[pos];
    short gy = d_grady[pos];
    float xperp = -((float)gx) / ((float)m00);
    float yperp = ((float)gy) / ((float)m00);
    
    float mag1, mag2;
    short z1, z2;
    
    // Determine gradient direction sector and compute interpolated values
    if (gx >= 0) {
        if (gy >= 0) {
            if (gx >= gy) { // Sector 111
                // Left point
                z1 = d_mag[pos-1];
                z2 = d_mag[pos-ncols-1];
                mag1 = (m00 - z1)*xperp + (z2 - z1)*yperp;
                
                // Right point
                z1 = d_mag[pos+1];
                z2 = d_mag[pos+ncols+1];
                mag2 = (m00 - z1)*xperp + (z2 - z1)*yperp;
            } 
            else { // Sector 110
                // Left point
                z1 = d_mag[pos-ncols];
                z2 = d_mag[pos-ncols-1];
                mag1 = (z1 - z2)*xperp + (z1 - m00)*yperp;
                
                // Right point
                z1 = d_mag[pos+ncols];
                z2 = d_mag[pos+ncols+1];
                mag2 = (z1 - z2)*xperp + (z1 - m00)*yperp;
            }
        }
        else {
            if (gx >= -gy) { // Sector 101
                // Left point
                z1 = d_mag[pos-1];
                z2 = d_mag[pos+ncols-1];
                mag1 = (m00 - z1)*xperp + (z1 - z2)*yperp;
                
                // Right point
                z1 = d_mag[pos+1];
                z2 = d_mag[pos-ncols+1];
                mag2 = (m00 - z1)*xperp + (z1 - z2)*yperp;
            }
            else { // Sector 100
                // Left point
                z1 = d_mag[pos+ncols];
                z2 = d_mag[pos+ncols-1];
                mag1 = (z1 - z2)*xperp + (m00 - z1)*yperp;
                
                // Right point
                z1 = d_mag[pos-ncols];
                z2 = d_mag[pos-ncols+1];
                mag2 = (z1 - z2)*xperp + (m00 - z1)*yperp;
            }
        }
    }
    else {
        if (gy >= 0) {
            if (-gx >= gy) { // Sector 011
                // Left point
                z1 = d_mag[pos+1];
                z2 = d_mag[pos-ncols+1];
                mag1 = (z1 - m00)*xperp + (z2 - z1)*yperp;
                
                // Right point
                z1 = d_mag[pos-1];
                z2 = d_mag[pos+ncols-1];
                mag2 = (z1 - m00)*xperp + (z2 - z1)*yperp;
            }
            else { // Sector 010
                // Left point
                z1 = d_mag[pos-ncols];
                z2 = d_mag[pos-ncols+1];
                mag1 = (z2 - z1)*xperp + (z1 - m00)*yperp;
                
                // Right point
                z1 = d_mag[pos+ncols];
                z2 = d_mag[pos+ncols-1];
                mag2 = (z2 - z1)*xperp + (z1 - m00)*yperp;
            }
        }
        else {
            if (-gx > -gy) { // Sector 001
                // Left point
                z1 = d_mag[pos+1];
                z2 = d_mag[pos+ncols+1];
                mag1 = (z1 - m00)*xperp + (z1 - z2)*yperp;
                
                // Right point
                z1 = d_mag[pos-1];
                z2 = d_mag[pos-ncols-1];
                mag2 = (z1 - m00)*xperp + (z1 - z2)*yperp;
            }
            else { // Sector 000
                // Left point
                z1 = d_mag[pos+ncols];
                z2 = d_mag[pos+ncols+1];
                mag1 = (z2 - z1)*xperp + (m00 - z1)*yperp;
                
                // Right point
                z1 = d_mag[pos-ncols];
                z2 = d_mag[pos-ncols-1];
                mag2 = (z2 - z1)*xperp + (m00 - z1)*yperp;
            }
        }
    }
    
    // Now determine if the current point is a maximum point
    if ((mag1 > 0.0f) || (mag2 > 0.0f)) {
        d_result[pos] = (unsigned char)NOEDGE;
    }
    else {
        if (mag2 == 0.0f)
            d_result[pos] = (unsigned char)NOEDGE;
        else
            d_result[pos] = (unsigned char)POSSIBLE_EDGE;
    }
}
void non_max_supp_cuda(short *h_mag, short *h_gradx, short *h_grady, 
                     int nrows, int ncols, unsigned char *h_result)
{
    // Allocate device memory
    short *d_mag, *d_gradx, *d_grady;
    unsigned char *d_result;
    
    cudaMalloc((void**)&d_mag, nrows * ncols * sizeof(short));
    cudaMalloc((void**)&d_gradx, nrows * ncols * sizeof(short));
    cudaMalloc((void**)&d_grady, nrows * ncols * sizeof(short));
    cudaMalloc((void**)&d_result, nrows * ncols * sizeof(unsigned char));
    
    // Copy data to device
    cudaMemcpy(d_mag, h_mag, nrows * ncols * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradx, h_gradx, nrows * ncols * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady, h_grady, nrows * ncols * sizeof(short), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((ncols + blockSize.x - 1) / blockSize.x, 
                 (nrows + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    non_max_supp_kernel<<<gridSize, blockSize>>>(d_mag, d_gradx, d_grady, nrows, ncols, d_result);
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, nrows * ncols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_mag);
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_result);
}
