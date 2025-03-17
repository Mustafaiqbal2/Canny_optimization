typedef long long fixed;
#define fixeddot 16

#define VERBOSE 0
#define BOOSTBLURFACTOR 90.0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "optimized_hysteresis.cu"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void gaussian_smooth_cuda(unsigned char *h_image, int rows, int cols, float sigma, 
                        short int *h_smoothedim);
                        
void derivative_x_y_cuda(short int *h_smoothedim, int rows, int cols,
                       short int **h_delta_x, short int **h_delta_y);
                       
void radian_direction_cuda(short int *h_delta_x, short int *h_delta_y, int rows, int cols, 
                         float **h_dir_radians, int xdirtag, int ydirtag);
                         
void magnitude_x_y_cuda(short int *h_delta_x, short int *h_delta_y, int rows, int cols,
                      short int **h_magnitude);
                      
void non_max_supp_cuda(short *h_mag, short *h_gradx, short *h_grady, 
                     int rows, int cols, unsigned char *h_result);
                     
void apply_hysteresis_cuda(short int *h_mag, unsigned char *h_nms, int rows, int cols,
                         float tlow, float thigh, unsigned char *h_edge);


int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols);
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval);

void canny(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname);
void gaussian_smooth(unsigned char *image, int rows, int cols, float sigma,
        short int **smoothedim);
void make_gaussian_kernel(float sigma, float **kernel, int *windowsize);

double angle_radians(double x, double y);
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
   if(read_pgm_image(infilename, &image, &rows, &cols) == 0){
      fprintf(stderr, "Error reading the input image, %s.\n", infilename);
      exit(1);
   }

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

   free(image);
   return 0;
}


void canny_cuda(unsigned char *image, int rows, int cols, float sigma,
         float tlow, float thigh, unsigned char **edge, char *fname)
{
   FILE *fpdir=NULL;          /* File to write the gradient image to.     */
   unsigned char *nms;        /* Points that are local maximal magnitude. */
   short int *smoothedim,     /* The image after gaussian smoothing.      */
             *delta_x,        /* The first devivative image, x-direction. */
             *delta_y,        /* The first derivative image, y-direction. */
             *magnitude;      /* The magnitude of the gadient image.      */
   float *dir_radians=NULL;   /* Gradient direction image.                */

   /****************************************************************************
   * Perform gaussian smoothing on the image using the input standard
   * deviation. (CUDA accelerated)
   ****************************************************************************/
   // Allocate memory for smoothedim
   smoothedim = (short int *)malloc(rows * cols * sizeof(short int));
   if(smoothedim == NULL) {
      fprintf(stderr, "Error allocating the smoothed image.\n");
      exit(1);
   }
   // Call CUDA version of gaussian_smooth
   gaussian_smooth_cuda(image, rows, cols, sigma, smoothedim);

   /****************************************************************************
   * Compute the first derivative in the x and y directions. (CUDA accelerated)
   ****************************************************************************/
   derivative_x_y_cuda(smoothedim, rows, cols, &delta_x, &delta_y);

   /****************************************************************************
   * This option to write out the direction of the edge gradient was added
   * to make the information available for computing an edge quality figure
   * of merit.
   ****************************************************************************/
   if(fname != NULL)
   {
      /*************************************************************************
      * Compute the direction up the gradient, in radians that are
      * specified counteclockwise from the positive x-axis. (CUDA accelerated)
      *************************************************************************/
      radian_direction_cuda(delta_x, delta_y, rows, cols, &dir_radians, -1, -1);

      /*************************************************************************
      * Write the gradient direction image out to a file.
      *************************************************************************/
      if((fpdir = fopen(fname, "wb")) == NULL)
      {
         fprintf(stderr, "Error opening the file %s for writing.\n", fname);
         exit(1);
      }
      fwrite(dir_radians, sizeof(float), rows*cols, fpdir);
      fclose(fpdir);
      free(dir_radians);
   }

   /****************************************************************************
   * Compute the magnitude of the gradient. (CUDA accelerated)
   ****************************************************************************/
   magnitude_x_y_cuda(delta_x, delta_y, rows, cols, &magnitude);

   /****************************************************************************
   * Perform non-maximal suppression.
   * TODO: Implement non_max_supp_cuda in optimized_canny_kernels.cu
   ****************************************************************************/
   if((nms = (unsigned char *) malloc(rows*cols*sizeof(unsigned char)))==NULL)
   {
      fprintf(stderr, "Error allocating the nms image.\n");
      exit(1);
   }
   non_max_supp_cuda(magnitude, delta_x, delta_y, rows, cols, nms);

   /****************************************************************************
   * Use hysteresis to mark the edge pixels.
   * TODO: Implement apply_hysteresis_cuda in optimized_hysteresis.cu
   ****************************************************************************/
   if((*edge=(unsigned char *)malloc(rows*cols*sizeof(unsigned char))) ==NULL)
   {
      fprintf(stderr, "Error allocating the edge image.\n");
      exit(1);
   }
   apply_hysteresis_cuda(magnitude, nms, rows, cols, tlow, thigh, *edge);

   /****************************************************************************
   * Free all of the memory that we allocated except for the edge image that
   * is still being used to store out result.
   ****************************************************************************/
   free(smoothedim);
   free(delta_x);
   free(delta_y);
   free(magnitude);
   free(nms);
}

double angle_radians(double x, double y)
{
   double xu, yu, ang;

   xu = fabs(x);
   yu = fabs(y);

   if((xu == 0) && (yu == 0)) return(0);

   ang = atan(yu/xu);

   if(x >= 0){
      if(y >= 0) return(ang);
      else return(2*M_PI - ang);
   }
   else{
      if(y >= 0) return(M_PI - ang);
      else return(M_PI + ang);
   }
}

__global__ void derivative_x_y_kernel(short int *d_smoothedim, int rows, int cols, 
   short int *d_delta_x, short int *d_delta_y)
{
   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;
   int pos = r * cols + c;

   if (r < rows && c < cols) 
   {
      // X derivative
      if (c == 0)
         d_delta_x[pos] = d_smoothedim[pos+1] - d_smoothedim[pos];
      else if (c == cols-1)
         d_delta_x[pos] = d_smoothedim[pos] - d_smoothedim[pos-1];
      else
         d_delta_x[pos] = d_smoothedim[pos+1] - d_smoothedim[pos-1];

      // Y derivative
      if (r == 0)
         d_delta_y[pos] = d_smoothedim[pos+cols] - d_smoothedim[pos];
      else if (r == rows-1)
         d_delta_y[pos] = d_smoothedim[pos] - d_smoothedim[pos-cols];
      else
         d_delta_y[pos] = d_smoothedim[pos+cols] - d_smoothedim[pos-cols];
   }
}

   // Host function
void derivative_x_y_cuda(short int *h_smoothedim, int rows, int cols,
   short int **h_delta_x, short int **h_delta_y)
{
   // Allocate host memory for output
   *h_delta_x = (short int*)malloc(rows * cols * sizeof(short int));
   *h_delta_y = (short int*)malloc(rows * cols * sizeof(short int));

   // Allocate device memory
   short int *d_smoothedim, *d_delta_x, *d_delta_y;
   cudaMalloc((void**)&d_smoothedim, rows * cols * sizeof(short int));
   cudaMalloc((void**)&d_delta_x, rows * cols * sizeof(short int));
   cudaMalloc((void**)&d_delta_y, rows * cols * sizeof(short int));

   // Copy input to device
   cudaMemcpy(d_smoothedim, h_smoothedim, rows * cols * sizeof(short int),
   cudaMemcpyHostToDevice);

   // Launch kernel
   dim3 blockSize(16, 16);
   dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
   (rows + blockSize.y - 1) / blockSize.y);

   derivative_x_y_kernel<<<gridSize, blockSize>>>(d_smoothedim, rows, cols, d_delta_x, d_delta_y);

   // Copy results back to host
   cudaMemcpy(*h_delta_x, d_delta_x, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);
   cudaMemcpy(*h_delta_y, d_delta_y, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(d_smoothedim);
   cudaFree(d_delta_x);
   cudaFree(d_delta_y);
}


void make_gaussian_kernel(float sigma, float **kernel, int *windowsize)
{
   int i, center;
   float x, fx, sum=0.0;

   *windowsize = 1 + 2 * ceil(2.5 * sigma);
   center = (*windowsize) / 2;

   if(VERBOSE) printf("      The kernel has %d elements.\n", *windowsize);
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

// On the host (CPU)
void gaussian_smooth_cuda(unsigned char *h_image, int rows, int cols, float sigma, 
   short int *h_smoothedim)
{
   // Create the Gaussian kernel on the host
   float *h_kernel;
   int windowsize;
   make_gaussian_kernel(sigma, &h_kernel, &windowsize);

   // Allocate device memory for kernel
   float *d_kernel;
   cudaMalloc((void**)&d_kernel, windowsize * sizeof(float));

   // Copy kernel to device
   cudaMemcpy(d_kernel, h_kernel, windowsize * sizeof(float), cudaMemcpyHostToDevice);

   // Allocate device memory for image, temp image, and result
   unsigned char *d_image;
   float *d_tempim;
   short int *d_smoothedim;

   cudaMalloc((void**)&d_image, rows * cols * sizeof(unsigned char));
   cudaMalloc((void**)&d_tempim, rows * cols * sizeof(float));
   cudaMalloc((void**)&d_smoothedim, rows * cols * sizeof(short int));

   // Copy input image to device
   cudaMemcpy(d_image, h_image, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice);

   // Set up grid and block dimensions
   dim3 blockSize(16, 16);
   dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                (rows + blockSize.y - 1) / blockSize.y);

   // First pass: X-direction blur
   opt_gaussian_smooth_x_kernel<<<gridSize, blockSize, windowsize * sizeof(float)>>>(
      d_image, rows, cols, d_kernel, windowsize, d_tempim);

   // Wait for X-direction kernel to finish
   cudaDeviceSynchronize();
   
   // Second pass: Y-direction blur
   opt_gaussian_smooth_y_kernel<<<gridSize, blockSize, windowsize * sizeof(float)>>>(
      d_tempim, rows, cols, d_kernel, windowsize, d_smoothedim);
   
   // Wait for Y-direction kernel to finish
   cudaDeviceSynchronize();

   // Copy final result back to host
   cudaMemcpy(h_smoothedim, d_smoothedim, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);

   // Free device memory
   cudaFree(d_kernel);
   cudaFree(d_image);
   cudaFree(d_tempim);
   cudaFree(d_smoothedim);

   // Free host memory
   free(h_kernel);
}

__global__ void magnitude_x_y_kernel(short int *d_delta_x, short int *d_delta_y, 
                                    int rows, int cols, short int *d_magnitude) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = r * cols + c;
    
    if (r < rows && c < cols) {
        int sq1 = (int)d_delta_x[pos] * (int)d_delta_x[pos];
        int sq2 = (int)d_delta_y[pos] * (int)d_delta_y[pos];
        d_magnitude[pos] = (short)(0.5f + sqrtf((float)sq1 + (float)sq2));
    }
}

// Host function
void magnitude_x_y_cuda(short int *h_delta_x, short int *h_delta_y, int rows, int cols,
                       short int **h_magnitude) 
{
    // Allocate host memory for output
    *h_magnitude = (short int*)malloc(rows * cols * sizeof(short int));
    if(*h_magnitude == NULL) {
        fprintf(stderr, "Error allocating the magnitude image.\n");
        exit(1);
    }
    
    // Allocate device memory
    short int *d_delta_x, *d_delta_y, *d_magnitude;
    cudaMalloc((void**)&d_delta_x, rows * cols * sizeof(short int));
    cudaMalloc((void**)&d_delta_y, rows * cols * sizeof(short int));
    cudaMalloc((void**)&d_magnitude, rows * cols * sizeof(short int));
    
    // Copy input to device
    cudaMemcpy(d_delta_x, h_delta_x, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_y, h_delta_y, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                 (rows + blockSize.y - 1) / blockSize.y);
                 
    magnitude_x_y_kernel<<<gridSize, blockSize>>>(d_delta_x, d_delta_y, rows, cols, d_magnitude);
    
    // Copy results back to host
    cudaMemcpy(*h_magnitude, d_magnitude, rows * cols * sizeof(short int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_magnitude);
}

// Device version of angle_radians function
__device__ float angle_radians_device(double x, double y)
{
   double xu, yu, ang;

   xu = fabs(x);
   yu = fabs(y);

   if((xu == 0) && (yu == 0)) return(0);

   ang = atan(yu/xu);

   if(x >= 0){
      if(y >= 0) return(ang);
      else return(2*M_PI - ang);
   }
   else{
      if(y >= 0) return(M_PI - ang);
      else return(M_PI + ang);
   }
}

__global__ void radian_direction_kernel(short int *d_delta_x, short int *d_delta_y, 
                                      int rows, int cols, float *d_dir_radians,
                                      int xdirtag, int ydirtag)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < rows && c < cols) {
        int pos = r * cols + c;
        double dx = (double)d_delta_x[pos];
        double dy = (double)d_delta_y[pos];
        
        if(xdirtag == 1) dx = -dx;
        if(ydirtag == -1) dy = -dy;
        
        d_dir_radians[pos] = angle_radians_device(dx, dy);
    }
}

// Host function
void radian_direction_cuda(short int *h_delta_x, short int *h_delta_y, int rows, int cols, 
                         float **h_dir_radians, int xdirtag, int ydirtag)
{
    // Allocate host memory for output
    *h_dir_radians = (float*)malloc(rows * cols * sizeof(float));
    if(*h_dir_radians == NULL) {
        fprintf(stderr, "Error allocating the gradient direction image.\n");
        exit(1);
    }
    
    // Allocate device memory
    short int *d_delta_x, *d_delta_y;
    float *d_dir_radians;
    
    cudaMalloc((void**)&d_delta_x, rows * cols * sizeof(short int));
    cudaMalloc((void**)&d_delta_y, rows * cols * sizeof(short int));
    cudaMalloc((void**)&d_dir_radians, rows * cols * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_delta_x, h_delta_x, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_y, h_delta_y, rows * cols * sizeof(short int), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                 (rows + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    radian_direction_kernel<<<gridSize, blockSize>>>(d_delta_x, d_delta_y, rows, cols, 
                                                  d_dir_radians, xdirtag, ydirtag);
    
    // Copy results back to host
    cudaMemcpy(*h_dir_radians, d_dir_radians, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_dir_radians);
}