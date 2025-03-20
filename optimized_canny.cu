typedef long long fixed;
#define fixeddot 16

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "optimized_hysteresis.cu"
#include "optimized_pgm_io.c"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void canny_cuda(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname);

int main(int argc, char *argv[])
{
   char *infilename = NULL;  /* Name of the input image */
   char *dirfilename = NULL; /* Name of the output gradient direction image */
   char outfilename[128];    /* Name of the output "edge" image */
   char composedfname[128];  /* Name of the output "direction" image */
   unsigned char *image;     /* The input image */
   unsigned char *edge;      /* The output edge image */
   int rows, cols;           /* The dimensions of the image. */
   float sigma,              /* Standard deviation of the gaussian kernel. */
	 tlow,               /* Fraction of the high threshold in hysteresis. */
	 thigh;              /* High hysteresis threshold control. The actual
			        threshold is the (100 * thigh) percentage point
			        in the histogram of the magnitude of the
			        gradient image that passes non-maximal
			        suppression. */

   /****************************************************************************
   * Get the command line arguments.
   ****************************************************************************/
   if(argc < 5){
   fprintf(stderr,"\n<USAGE> %s image sigma tlow thigh [writedirim]\n",argv[0]);
      fprintf(stderr,"\n      image:      An image to process. Must be in ");
      fprintf(stderr,"PGM format.\n");
      fprintf(stderr,"      sigma:      Standard deviation of the gaussian");
      fprintf(stderr," blur kernel.\n");
      fprintf(stderr,"      tlow:       Fraction (0.0-1.0) of the high ");
      fprintf(stderr,"edge strength threshold.\n");
      fprintf(stderr,"      thigh:      Fraction (0.0-1.0) of the distribution");
      fprintf(stderr," of non-zero edge\n                  strengths for ");
      fprintf(stderr,"hysteresis. The fraction is used to compute\n");
      fprintf(stderr,"                  the high edge strength threshold.\n");
      fprintf(stderr,"      writedirim: Optional argument to output ");
      fprintf(stderr,"a floating point");
      fprintf(stderr," direction image.\n\n");
      exit(1);
   }

   infilename = argv[1];
   sigma = atof(argv[2]);
   tlow = atof(argv[3]);
   thigh = atof(argv[4]);

   if(argc == 6) dirfilename = infilename;
   else dirfilename = NULL;

   /****************************************************************************
   * Read in the image. This read function allocates memory for the image.
   ****************************************************************************/
  unsigned char *temp_image;
  if(read_pgm_image(infilename, &temp_image, &rows, &cols) == 0){
     fprintf(stderr, "Error reading the input image, %s.\n", infilename);
     exit(1);
  }
  
  // Transfer to pinned memory for better CUDA performance
  cudaMallocHost((void**)&image, rows * cols * sizeof(unsigned char));
  memcpy(image, temp_image, rows * cols * sizeof(unsigned char));
  free(temp_image);
   /****************************************************************************
   * Perform the edge detection. All of the work takes place here.
   ****************************************************************************/
   if(dirfilename != NULL){
      sprintf(composedfname, "%s_s_%3.2f_l_%3.2f_h_%3.2f.fim", infilename,
      sigma, tlow, thigh);
      dirfilename = composedfname;
   }

   ///////  
   // Create CUDA timing events
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   // Record start time
   cudaEventRecord(start, 0);
   
   canny_cuda(image, rows, cols, sigma, tlow, thigh, &edge, dirfilename);
   
   // Record end time
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Canny edge detection processing time: %f ms\n", milliseconds);
   
   // Clean up timing events
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   ///////

   /****************************************************************************
   * Write out the edge image to a file.
   ****************************************************************************/
   sprintf(outfilename, "%s_s_%3.2f_l_%3.2f_h_%3.2f_cuda.pgm", infilename, sigma, tlow, thigh);
   if(write_pgm_image(outfilename, edge, rows, cols, "", 255) == 0){
      fprintf(stderr, "Error writing the edge image, %s.\n", outfilename);
      exit(1);
   }

   cudaFreeHost(image);
   return 0;
}

__global__ void derivative_magnitude_kernel(short int *d_smoothedim, int rows, int cols, 
                                           short int *d_magnitude, short int *d_delta_x, short int *d_delta_y)
{
   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;
   int pos = r * cols + c;

   if (r < rows && c < cols) {
      // X derivative
      short int dx;
      if (c == 0)
         dx = d_smoothedim[pos+1] - d_smoothedim[pos];
      else if (c == cols-1)
         dx = d_smoothedim[pos] - d_smoothedim[pos-1];
      else
         dx = d_smoothedim[pos+1] - d_smoothedim[pos-1];
      
      // Y derivative
      short int dy;
      if (r == 0)
         dy = d_smoothedim[pos+cols] - d_smoothedim[pos];
      else if (r == rows-1)
         dy = d_smoothedim[pos] - d_smoothedim[pos-cols];
      else
         dy = d_smoothedim[pos+cols] - d_smoothedim[pos-cols];
      
      // Store derivatives for later use in non-max suppression
      d_delta_x[pos] = dx;
      d_delta_y[pos] = dy;
      
      // Calculate magnitude in the same kernel
      int sq1 = (int)dx * (int)dx;
      int sq2 = (int)dy * (int)dy;
      d_magnitude[pos] = (short)(0.5f + sqrtf((float)sq1 + (float)sq2));
   }
}


void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum=0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if((*kernel = (float *) malloc((*windowsize)* sizeof(float))) == NULL){
      fprintf(stderr, "Error callocing the gaussian kernel array.\n");
      exit(1);
   }

   for(i=0;i<(*windowsize);i++){
      x = (float)(i - center);
      fx = pow(2.71828, -0.5*x*x/(sigma*sigma)) / (sigma * sqrt(6.2831853));
      (*kernel)[i] = fx;
      sum += fx;
   }

   for(i=0;i<(*windowsize);i++) (*kernel)[i] /= sum;
}


__global__ void opt_gaussian_smooth_x_kernel(unsigned char *d_image, int rows, int cols, 
float *d_kernel, int windowsize,
float *d_tempim)
{
   // Get the half size of the kernel (radius)
   int center = windowsize / 2;

   // Shared memory for the Gaussian kernel
   extern __shared__ float s_kernel[];

   // Load kernel into shared memory (cooperatively by all threads in the block)
   int tid = threadIdx.x + threadIdx.y * blockDim.x;
   int blockSize = blockDim.x * blockDim.y;

   for (int i = tid; i < windowsize; i += blockSize) {
      if (i < windowsize) {
         s_kernel[i] = d_kernel[i];
      }
   }

   // Ensure all threads have loaded the kernel
   __syncthreads();
   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;

   if (r < rows && c < cols) {
      float dot = 0.0f;
      float sum = 0.0f;

      // Perform convolution in X direction
      for(int cc = -center; cc <= center; cc++) {
         if(((c+cc) >= 0) && ((c+cc) < cols)) {
            dot += (float)d_image[r*cols+(c+cc)] * s_kernel[center+cc];
            sum += s_kernel[center+cc];
         }
   }

   // Store result in temporary buffer
   d_tempim[r*cols+c] = dot/sum;
   }
}

__global__ void opt_gaussian_smooth_y_kernel(float *d_tempim, int rows, int cols, 
float *d_kernel, int windowsize,
short int *d_smoothedim)
{
   // Get the half size of the kernel (radius)
   int center = windowsize / 2;

   // Shared memory for the Gaussian kernel
   extern __shared__ float s_kernel[];

   // Load kernel into shared memory (cooperatively by all threads in the block)
   int tid = threadIdx.x + threadIdx.y * blockDim.x;
   int blockSize = blockDim.x * blockDim.y;

   for (int i = tid; i < windowsize; i += blockSize) {
      if (i < windowsize) {
         s_kernel[i] = d_kernel[i];
      }
   }

   // Ensure all threads have loaded the kernel
   __syncthreads();

   // Calculate global position for this thread
   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;

   // Only process pixels within the image boundaries
   if (r < rows && c < cols) {
      float dot = 0.0f;
      float sum = 0.0f;

      // Perform convolution in Y direction
      for(int rr = -center; rr <= center; rr++) {
         if(((r+rr) >= 0) && ((r+rr) < rows)) {
            dot += d_tempim[(r+rr)*cols+c] * s_kernel[center+rr];
            sum += s_kernel[center+rr];
         }
      }

      // Store final result in output buffer with boost factor
      d_smoothedim[r*cols+c] = (short int)(dot*BOOSTBLURFACTOR/sum + 0.5f);
   }
}


void complete_canny_pipeline_cuda(unsigned char *h_image, int rows, int cols, 
   float sigma, float tlow, float thigh, 
   unsigned char *h_edge)
{
   // Timing variables
   cudaEvent_t start, stop;
   float milliseconds = 0.0f;
   float total_kernel_time = 0.0f;
   float total_transfer_time = 0.0f;
   float total_cpu_time = 0.0f;
   
   // Create events for timing
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   
   printf("\n----- TIMING BREAKDOWN -----\n");
   
   
   // Create the Gaussian kernel
   float *h_kernel;
   int windowsize;
   make_gaussian_kernel(sigma, &h_kernel, &windowsize);


   // Allocate device memory
   cudaEventRecord(start, 0);
   
   unsigned char *d_image;
   float *d_tempim, *d_kernel;
   short int *d_smoothedim, *d_gradx, *d_grady, *d_magnitude;
   unsigned char *d_nms, *d_edge;
   int *d_hist, *d_changes;

   // Allocate all required device memory at once
   cudaMalloc((void**)&d_image, rows * cols * sizeof(unsigned char));
   cudaMalloc((void**)&d_tempim, rows * cols * sizeof(float));
   cudaMalloc((void**)&d_kernel, windowsize * sizeof(float));
   cudaMalloc((void**)&d_smoothedim, rows * cols * sizeof(short int));
   cudaMalloc((void**)&d_gradx, rows * cols * sizeof(short int));
   cudaMalloc((void**)&d_grady, rows * cols * sizeof(short int));
   cudaMalloc((void**)&d_magnitude, rows * cols * sizeof(short int));
   cudaMalloc((void**)&d_nms, rows * cols * sizeof(unsigned char));
   cudaMalloc((void**)&d_edge, rows * cols * sizeof(unsigned char));
   cudaMalloc((void**)&d_hist, 32768 * sizeof(int));
   cudaMalloc((void**)&d_changes, sizeof(int));
   
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Memory: Device allocation: %.3f ms\n", milliseconds);
   
   // Initialize histogram to zeros
   
   cudaMemset(d_hist, 0, 32768 * sizeof(int));


   // Copy input data to device
   cudaEventRecord(start, 0);
   cudaMemcpy(d_image, h_image, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Transfer: Image to device: %.3f ms\n", milliseconds);
   total_transfer_time += milliseconds;
   
   cudaMemcpy(d_kernel, h_kernel, windowsize * sizeof(float), cudaMemcpyHostToDevice);

   // Set up grid and block dimensions
   dim3 blockSize(16, 16);
   dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

   // Step 1: Gaussian smooth X direction
   cudaEventRecord(start, 0);
   opt_gaussian_smooth_x_kernel<<<gridSize, blockSize, windowsize * sizeof(float)>>>(
   d_image, rows, cols, d_kernel, windowsize, d_tempim);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel: Gaussian X direction: %.3f ms\n", milliseconds);
   total_kernel_time += milliseconds;

   // Step 2: Gaussian smooth Y direction
   cudaEventRecord(start, 0);
   opt_gaussian_smooth_y_kernel<<<gridSize, blockSize, windowsize * sizeof(float)>>>(
   d_tempim, rows, cols, d_kernel, windowsize, d_smoothedim);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel: Gaussian Y direction: %.3f ms\n", milliseconds);
   total_kernel_time += milliseconds;

   // Step 3: Compute derivatives and magnitude
   cudaEventRecord(start, 0);
   derivative_magnitude_kernel<<<gridSize, blockSize>>>(
   d_smoothedim, rows, cols, d_magnitude, d_gradx, d_grady);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel: Derivatives and magnitude: %.3f ms\n", milliseconds);
   total_kernel_time += milliseconds;

   // Step 4: Non-maximum suppression
   cudaEventRecord(start, 0);
   non_max_supp_kernel<<<gridSize, blockSize>>>(
   d_magnitude, d_gradx, d_grady, rows, cols, d_nms);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel: Non-maximum suppression: %.3f ms\n", milliseconds);
   total_kernel_time += milliseconds;

   // Step 5: Initialize edges and histogram
   cudaEventRecord(start, 0);
   init_edges_kernel<<<gridSize, blockSize>>>(
   d_nms, d_edge, d_magnitude, d_hist, rows, cols);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel: Edge initialization: %.3f ms\n", milliseconds);
   total_kernel_time += milliseconds;

   // Compute thresholds - CPU
   cudaEventRecord(start, 0);
   int *h_hist = (int*)malloc(32768 * sizeof(int));
   cudaMemcpy(h_hist, d_hist, 32768 * sizeof(int), cudaMemcpyDeviceToHost);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Transfer: Histogram to host: %.3f ms\n", milliseconds);
   total_transfer_time += milliseconds;

   // CPU threshold computation
   cudaEventRecord(start, 0);
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
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("CPU: Threshold computation: %.3f ms\n", milliseconds);
   total_cpu_time += milliseconds;

   // Step 6: Apply threshold
   cudaEventRecord(start, 0);
   apply_threshold_kernel<<<gridSize, blockSize>>>(
   d_edge, d_magnitude, highthreshold, rows, cols);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Kernel: Threshold application: %.3f ms\n", milliseconds);
   total_kernel_time += milliseconds;

   // Step 7: Iterative hysteresis expansion
   int h_changes = 1;
   int iteration = 0;
   float hysteresis_time = 0.0f;

   while (h_changes) {
      h_changes = 0;
      
      cudaEventRecord(start, 0);
      cudaMemcpy(d_changes, &h_changes, sizeof(int), cudaMemcpyHostToDevice);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      total_transfer_time += milliseconds;
      
      bool final_pass = iteration >= 4;

      cudaEventRecord(start, 0);
      expansion_and_cleanup_kernel<<<gridSize, blockSize>>>(
      d_edge, d_magnitude, lowthreshold, rows, cols, d_changes, final_pass);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      hysteresis_time += milliseconds;
      total_kernel_time += milliseconds;
      
      cudaEventRecord(start, 0);
      cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      total_transfer_time += milliseconds;

      if (!h_changes && !final_pass) {
         final_pass = true;
         
         cudaEventRecord(start, 0);
         expansion_and_cleanup_kernel<<<gridSize, blockSize>>>(
         d_edge, d_magnitude, lowthreshold, rows, cols, d_changes, final_pass);
         cudaEventRecord(stop, 0);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&milliseconds, start, stop);
         hysteresis_time += milliseconds;
         total_kernel_time += milliseconds;
         
         break;
      }

      iteration++;
   }
   printf("Kernel: Hysteresis expansion (%d iterations): %.3f ms\n", iteration, hysteresis_time);

   // Copy final result back to host
   cudaEventRecord(start, 0);
   cudaMemcpy(h_edge, d_edge, rows * cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Transfer: Result to host: %.3f ms\n", milliseconds);
   total_transfer_time += milliseconds;

   // Free all device memory
   cudaEventRecord(start, 0);
   cudaFree(d_image);
   cudaFree(d_tempim);
   cudaFree(d_kernel);
   cudaFree(d_smoothedim);
   cudaFree(d_gradx);
   cudaFree(d_grady);
   cudaFree(d_magnitude);
   cudaFree(d_nms);
   cudaFree(d_edge);
   cudaFree(d_hist);
   cudaFree(d_changes);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Memory: Device cleanup: %.3f ms\n", milliseconds);

   // Free host memory
   free(h_kernel);
   free(h_hist);
   
   // Print summary
   printf("Total CPU time: %.3f ms\n", total_cpu_time);
   printf("Total kernel execution time: %.3f ms\n", total_kernel_time);
   printf("Total memory transfer time: %.3f ms\n", total_transfer_time);
   
   // Cleanup timing events
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
}
void canny_cuda(unsigned char *image, int rows, int cols, float sigma,
   float tlow, float thigh, unsigned char **edge, char *fname)
{
   // Allocate output memory for the edge image
   if((*edge=(unsigned char *)malloc(rows*cols*sizeof(unsigned char))) == NULL) {
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
   }

   // Execute complete pipeline with single function - minimizes memory transfers
   complete_canny_pipeline_cuda(image, rows, cols, sigma, tlow, thigh, *edge);
}
