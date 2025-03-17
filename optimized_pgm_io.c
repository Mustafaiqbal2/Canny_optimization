/*******************************************************************************
* FILE: pgm_io.c
* This code was written by Mike Heath. heath@csee.usf.edu (in 1995).
* Modified for optimization with C++ threads.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>  // For POSIX threads
#include <stdint.h>   // For fixed-width integer types
#include <unistd.h>   // For sysconf

// Number of threads to use (adjust based on CPU cores)
#define NUM_THREADS 4
#define BUFFER_SIZE 4096

// Thread data structure
typedef struct {
    unsigned char *src_red;
    unsigned char *src_grn;
    unsigned char *src_blu;
    unsigned char *dst_buffer;
    long start_idx;
    long end_idx;
    int interleave; // 1 for interleaving, 0 for deinterleaving
} ThreadData;

// Function prototypes for thread workers
void* interleave_worker(void* arg);
void* deinterleave_worker(void* arg);
int get_num_threads();

// Get optimal number of threads based on system cores
int get_num_threads() {
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    return (num_cores > 0) ? (int)num_cores : NUM_THREADS;
}

// Thread worker for interleaving RGB channels for PPM writing
void* interleave_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    long i, j = 0;
    
    for(i = data->start_idx; i < data->end_idx; i++, j += 3) {
        data->dst_buffer[j] = data->src_red[i];
        data->dst_buffer[j+1] = data->src_grn[i];
        data->dst_buffer[j+2] = data->src_blu[i];
    }
    
    return NULL;
}

// Thread worker for deinterleaving RGB channels for PPM reading
void* deinterleave_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    unsigned char* buffer = data->dst_buffer;
    long i, j = 0;
    
    for(i = data->start_idx; i < data->end_idx; i++, j += 3) {
        data->src_red[i] = buffer[j];
        data->src_grn[i] = buffer[j+1];
        data->src_blu[i] = buffer[j+2];
    }
    
    return NULL;
}

/******************************************************************************
* Function: read_pgm_image
* Purpose: This function reads in an image in PGM format. The image can be
* read in from either a file or from standard input. The image is only read
* from standard input when infilename = NULL. Because the PGM format includes
* the number of columns and the number of rows in the image, these are read
* from the file. Memory to store the image is allocated in this function.
* All comments in the header are discarded in the process of reading the
* image. Upon failure, this function returns 0, upon sucess it returns 1.
******************************************************************************/
int read_pgm_image(char *infilename, unsigned char **image, int *rows,
    int *cols)
{
   FILE *fp;
   char buf[71];

   /***************************************************************************
   * Open the input image file for reading if a filename was given. If no
   * filename was provided, set fp to read from standard input.
   ***************************************************************************/
   if(infilename == NULL) fp = stdin;
   else{
      if((fp = fopen(infilename, "r")) == NULL){
         fprintf(stderr, "Error reading the file %s in read_pgm_image().\n",
            infilename);
         return(0);
      }
   }

   /***************************************************************************
   * Verify that the image is in PGM format, read in the number of columns
   * and rows in the image and scan past all of the header information.
   ***************************************************************************/
   fgets(buf, 70, fp);
   if(strncmp(buf,"P5",2) != 0){
      fprintf(stderr, "The file %s is not in PGM format in ", infilename);
      fprintf(stderr, "read_pgm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */
   sscanf(buf, "%d %d", cols, rows);
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */

   /***************************************************************************
   * Allocate memory to store the image then read the image from the file.
   ***************************************************************************/
   if(((*image) = (unsigned char *) malloc((*rows)*(*cols))) == NULL){
      fprintf(stderr, "Memory allocation failure in read_pgm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   if((*rows) != fread((*image), (*cols), (*rows), fp)){
      fprintf(stderr, "Error reading the image data in read_pgm_image().\n");
      if(fp != stdin) fclose(fp);
      free((*image));
      return(0);
   }

   if(fp != stdin) fclose(fp);
   return(1);
}

/******************************************************************************
* Function: write_pgm_image
* Purpose: This function writes an image in PGM format. The file is either
* written to the file specified by outfilename or to standard output if
* outfilename = NULL. A comment can be written to the header if coment != NULL.
******************************************************************************/
int write_pgm_image(char *outfilename, unsigned char *image, int rows,
    int cols, char *comment, int maxval)
{
   FILE *fp;

   /***************************************************************************
   * Open the output image file for writing if a filename was given. If no
   * filename was provided, set fp to write to standard output.
   ***************************************************************************/
   if(outfilename == NULL) fp = stdout;
   else{
      if((fp = fopen(outfilename, "w")) == NULL){
         fprintf(stderr, "Error writing the file %s in write_pgm_image().\n",
            outfilename);
         return(0);
      }
   }

   /***************************************************************************
   * Write the header information to the PGM file.
   ***************************************************************************/
   fprintf(fp, "P5\n%d %d\n", cols, rows);
   if(comment != NULL)
      if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
   fprintf(fp, "%d\n", maxval);

   /***************************************************************************
   * Write the image data to the file.
   ***************************************************************************/
   if(rows != fwrite(image, cols, rows, fp)){
      fprintf(stderr, "Error writing the image data in write_pgm_image().\n");
      if(fp != stdout) fclose(fp);
      return(0);
   }

   if(fp != stdout) fclose(fp);
   return(1);
}

/******************************************************************************
* Function: read_ppm_image
* Purpose: This function reads in an image in PPM format with optimizations.
******************************************************************************/
int read_ppm_image(char *infilename, unsigned char **image_red, 
    unsigned char **image_grn, unsigned char **image_blu, int *rows,
    int *cols)
{
   FILE *fp;
   char buf[71];
   long size;
   unsigned char *buffer;

   /***************************************************************************
   * Open the input image file for reading if a filename was given. If no
   * filename was provided, set fp to read from standard input.
   ***************************************************************************/
   if(infilename == NULL) fp = stdin;
   else{
      if((fp = fopen(infilename, "rb")) == NULL){ // Using binary mode
         fprintf(stderr, "Error reading the file %s in read_ppm_image().\n",
            infilename);
         return(0);
      }
   }

   /***************************************************************************
   * Verify that the image is in PPM format, read in the number of columns
   * and rows in the image and scan past all of the header information.
   ***************************************************************************/
   fgets(buf, 70, fp);
   if(strncmp(buf,"P6",2) != 0){
      fprintf(stderr, "The file %s is not in PPM format in ", infilename);
      fprintf(stderr, "read_ppm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */
   sscanf(buf, "%d %d", cols, rows);
   do{ fgets(buf, 70, fp); }while(buf[0] == '#');  /* skip all comment lines */

   /***************************************************************************
   * Allocate memory to store the image then read the image from the file.
   ***************************************************************************/
   size = (*rows) * (*cols);
   if(((*image_red) = (unsigned char *) aligned_alloc(16, size)) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      if(fp != stdin) fclose(fp);
      return(0);
   }
   if(((*image_grn) = (unsigned char *) aligned_alloc(16, size)) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      free(*image_red);
      if(fp != stdin) fclose(fp);
      return(0);
   }
   if(((*image_blu) = (unsigned char *) aligned_alloc(16, size)) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      free(*image_red);
      free(*image_grn);
      if(fp != stdin) fclose(fp);
      return(0);
   }

   // Allocate buffer for entire image (RGB interleaved)
   long buffer_size = size * 3;
   if((buffer = (unsigned char *) malloc(buffer_size)) == NULL){
      fprintf(stderr, "Memory allocation failure in read_ppm_image().\n");
      free(*image_red);
      free(*image_grn);
      free(*image_blu);
      if(fp != stdin) fclose(fp);
      return(0);
   }
   
   // Read all image data at once
   if(buffer_size != fread(buffer, 1, buffer_size, fp)){
      fprintf(stderr, "Error reading the image data in read_ppm_image().\n");
      free(buffer);
      free(*image_red);
      free(*image_grn);
      free(*image_blu);
      if(fp != stdin) fclose(fp);
      return(0);
   }
   
   // Use threads to deinterleave the RGB data
   int num_threads = get_num_threads();
   pthread_t threads[num_threads];
   ThreadData thread_data[num_threads];
   long chunk_size = size / num_threads;
   
   for(int i = 0; i < num_threads; i++) {
      thread_data[i].src_red = *image_red;
      thread_data[i].src_grn = *image_grn;
      thread_data[i].src_blu = *image_blu;
      thread_data[i].dst_buffer = buffer;
      thread_data[i].start_idx = i * chunk_size;
      thread_data[i].end_idx = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
      
      if(pthread_create(&threads[i], NULL, deinterleave_worker, &thread_data[i])) {
         fprintf(stderr, "Error creating thread.\n");
         free(buffer);
         free(*image_red);
         free(*image_grn);
         free(*image_blu);
         if(fp != stdin) fclose(fp);
         return(0);
      }
   }
   
   // Wait for all threads to complete
   for(int i = 0; i < num_threads; i++) {
      pthread_join(threads[i], NULL);
   }
   
   free(buffer);
   if(fp != stdin) fclose(fp);
   return(1);
}

/******************************************************************************
* Function: write_ppm_image
* Purpose: This function writes an image in PPM format with optimizations.
******************************************************************************/
int write_ppm_image(char *outfilename, unsigned char *image_red,
    unsigned char *image_grn, unsigned char *image_blu, int rows,
    int cols, char *comment, int maxval)
{
   FILE *fp;
   long size;
   unsigned char *buffer;

   /***************************************************************************
   * Open the output image file for writing if a filename was given. If no
   * filename was provided, set fp to write to standard output.
   ***************************************************************************/
   if(outfilename == NULL) fp = stdout;
   else{
      if((fp = fopen(outfilename, "wb")) == NULL){ // Using binary mode
         fprintf(stderr, "Error writing the file %s in write_ppm_image().\n",
            outfilename);
         return(0);
      }
   }

   /***************************************************************************
   * Write the header information to the PPM file.
   ***************************************************************************/
   fprintf(fp, "P6\n%d %d\n", cols, rows);
   if(comment != NULL)
      if(strlen(comment) <= 70) fprintf(fp, "# %s\n", comment);
   fprintf(fp, "%d\n", maxval);

   /***************************************************************************
   * Write the image data to the file using multi-threaded approach.
   ***************************************************************************/
   size = (long)rows * (long)cols;
   
   // Allocate buffer for entire image (RGB interleaved)
   long buffer_size = size * 3;
   if((buffer = (unsigned char *) malloc(buffer_size)) == NULL){
      fprintf(stderr, "Memory allocation failure in write_ppm_image().\n");
      if(fp != stdout) fclose(fp);
      return(0);
   }
   
   // Use threads to interleave the RGB data
   int num_threads = get_num_threads();
   pthread_t threads[num_threads];
   ThreadData thread_data[num_threads];
   long chunk_size = size / num_threads;
   
   for(int i = 0; i < num_threads; i++) {
      thread_data[i].src_red = image_red;
      thread_data[i].src_grn = image_grn;
      thread_data[i].src_blu = image_blu;
      thread_data[i].dst_buffer = buffer;
      thread_data[i].start_idx = i * chunk_size;
      thread_data[i].end_idx = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
      
      if(pthread_create(&threads[i], NULL, interleave_worker, &thread_data[i])) {
         fprintf(stderr, "Error creating thread.\n");
         free(buffer);
         if(fp != stdout) fclose(fp);
         return(0);
      }
   }
   
   // Wait for all threads to complete
   for(int i = 0; i < num_threads; i++) {
      pthread_join(threads[i], NULL);
   }
   
   // Write the entire buffer at once
   if(buffer_size != fwrite(buffer, 1, buffer_size, fp)){
      fprintf(stderr, "Error writing the image data in write_ppm_image().\n");
      free(buffer);
      if(fp != stdout) fclose(fp);
      return(0);
   }
   
   free(buffer);
   if(fp != stdout) fclose(fp);
   return(1);
}
