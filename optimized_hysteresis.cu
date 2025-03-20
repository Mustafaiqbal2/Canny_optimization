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

// Combined kernel for final hysteresis expansion and cleanup
__global__ void expansion_and_cleanup_kernel(unsigned char *d_edge, short *d_mag, 
      int lowthreshold, int rows, int cols,
      int *d_changes, bool final_pass)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        int pos = r * cols + c;

        // Skip borders for expansion
        if (r >= 1 && r < rows-1 && c >= 1 && c < cols-1) {
            // Look at neighboring pixels only if this is a confirmed edge
            if (d_edge[pos] == D_EDGE) 
            {
                // Check all 8 neighbors
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (i == 0 && j == 0) continue; // Skip self

                        int npos = (r+i) * cols + (c+j);
                        if (d_edge[npos] == D_POSSIBLE_EDGE && d_mag[npos] >= lowthreshold) {
                            d_edge[npos] = D_EDGE;
                            *d_changes = 1; 
                        }
                    }
                }
            }
        }

        // Cleanup phase - only execute on final pass
        if (final_pass && d_edge[pos] == D_POSSIBLE_EDGE) {
        d_edge[pos] = D_NOEDGE;
        }
    }
}
