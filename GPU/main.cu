// PAZAM: A CUDA Music Identification Tool
// Michael Wilner - Cody Van Etten - Ahmed Suhyl
// ...
// Naive CPU Implementation based on Tyler Simon's WAV FFT Tool

#include <stdio.h>
#include <math.h>
#include "tinydir.h"
#include "kernel.cu"
#include <time.h>

#define BYTES_PER_SAMPLE 2
#define MAXSONGS 10
#define COLSPERSONG  10
 #define FREQBANDWIDTH 50
#define MAXCOLS (MAXSONGS*COLSPERSONG)
#define THREADS_PER_BLOCK N
#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)
//static float A[2*N]; /* available for modifying transform */
//static float Z[2*N];

int FUZ_FACTOR = 2;

long hash1(long p1, long p2, long p3, long p4) {
  return (p4 - (p4 % FUZ_FACTOR)) * 100000000 + (p3 - (p3 % FUZ_FACTOR))
      * 100000 + (p2 - (p2 % FUZ_FACTOR)) * 100
      + (p1 - (p1 % FUZ_FACTOR));
}

int generatehashes(char *input_file, int mysongid, int * hash_songs)
{
  //float* Z = (float*) malloc(sizeof(float) * N * 2);
  int i, sect, sectcnt;
  FILE * inp;
  char riff[4];
  int  sread; /* bytes read/written */
  int  fsize;
  char wave[4];
  char fmt[4];
  int  nbytes;
  short  ccode;
  short  channels;
  int rate;
  int avgrate; /* average rate in samples per second */
  short blockalign;
  short bps; /* bits per sample */
  char data[4];
  int csize;
  char stuf;
  short soundin; /* sample of sound */
  int bad; /* flags bad data in read */
  //int nbread; /* number of bytes read */
  
  inp = fopen(input_file, "rb");
  if(inp == NULL)
  {
    printf("can not open %s for reading. \n", input_file);
    return 0;
  }

  //printf("reading %s \n", input_file);
  
  sread = fread(&riff[0], 1, 4, inp);
  //printf("first 4 bytes should be RIFF, <%c%c%c%c>\n",  riff[0],riff[1],riff[2],riff[3]);
  
  sread = fread(&fsize, 1, 4, inp);
  //printf("file has %d +8 bytes \n", fsize);
  
  sread = fread(&wave[0], 1, 4, inp);
  //printf("should be WAVE, <%c%c%c%c>\n",wave[0],wave[1],wave[2],wave[3]);
  
  sread = fread(&fmt[0], 1, 4, inp);
  //printf("should be fmt, <%c%c%c%c>\n",fmt[0],fmt[1],fmt[2],fmt[3]);
  
  sread = fread(&nbytes, 1, 4, inp);
  //printf("block has %d more bytes \n", nbytes);
  
  sread = fread(&ccode, 1, 2, inp);
  //printf("compression code = %d \n", ccode);
  nbytes = nbytes-2;
  
  sread = fread(&channels, 1, 2, inp);
  //printf("channels = %d \n", channels);
  nbytes = nbytes-2;
  
  sread = fread(&rate, 1, 4, inp);
  //printf("rate = %d  \n", rate);
  nbytes = nbytes-4;
  
  sread = fread(&avgrate, 1, 4, inp);
  //printf("avg rate = %d \n", avgrate);
  nbytes = nbytes-4;
  
  sread = fread(&blockalign, 1, 2, inp);
  //printf("blockalign = %d  \n", blockalign);
  nbytes = nbytes-2;
  
  sread = fread(&bps, 1, 2, inp);
  //printf("bits per sample = %d \n", bps);
  nbytes = nbytes-2;
  //printf("bytes left in fmt = %d \n", nbytes);
  for(i=0; i<nbytes; i++) sread = fread(&stuf, 1, 1, inp);
  
  sread = fread(&data[0], 1, 4, inp);
  //printf("should be data, <%c%c%c%c>\n",data[0],data[1],data[2],data[3]);
  
  sread = fread(&csize, 1, 4, inp);
  //printf("chunk has %d more bytes \n", csize);
  //nbread = 44+nbytes;
  //printf("%d bytes read so far \n", nbread);
  
  bad = 0;
  sect = 0;
  sectcnt = 0;

  float* G = (float*) malloc(sizeof(float) * csize);
  
  for(i=0; i<csize; i+=BYTES_PER_SAMPLE)
  {
    if(sect<N)
    {
      sread = fread(&soundin, 1, BYTES_PER_SAMPLE, inp); //We have to make sure we're reading both channels
      if(sread != BYTES_PER_SAMPLE && bad==0) { bad=1; printf("no read on byte %d \n", i); }
      G[sectcnt*2*N + 2*sect] = (float)soundin;
      G[sectcnt*2*N + 2*sect+1] = 0.0; /* no complex component */      
      sect++;
    }
    else
    {
      
      sectcnt++;
      i-=BYTES_PER_SAMPLE;
      sect = 0;
    }
  }

  float *in_h;
  float *in_d;
  int * out_h, * out_d;
  cudaError_t cuda_ret;
  dim3 dim_grid, dim_block;
  in_h = G;
  out_h = (int *) malloc(sizeof(int) * sectcnt); //list of keys
  cuda_ret = cudaMalloc((void**)&in_d, sectcnt * N * sizeof(float));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate input device memory");
  cuda_ret = cudaMalloc((void**)&out_d, sectcnt*sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate output device memory");
  cudaDeviceSynchronize();
  cuda_ret = cudaMemcpy(in_d, in_h, sectcnt * N * sizeof(float),cudaMemcpyHostToDevice);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy input audio to device");
  cudaDeviceSynchronize();
  dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
  dim_grid.x = (sectcnt - 1)/THREADS_PER_BLOCK + 1; dim_grid.y = dim_grid.z = 1;
  parallelhash<<<dim_grid, dim_block>>>(in_d, out_d, sectcnt, hash_songs, mysongid);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Kernel 1 failed");
  cuda_ret = cudaMemcpy(out_h, out_d, sectcnt*sizeof(int), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy output to host");
  cudaDeviceSynchronize();
  cudaFree(in_d); cudaFree(out_d);

  /*for(i=0; i<sectcnt; i++)
  {
      int key = out_h[i];
      int n = 0;
      for (n = 0 ; n < MAXCOLS; n++)
      {
        if(hashtable[key][n]==0)
        {
          hashtable[key][n] = mysongid;
          numhashes++;
          break;
        }
      }
  } // end i<csize loop */
  fclose(inp);  
  free(G);
  free(out_h);

  return csize>>6;
}



int main(int argc, char * argv[])
{
  int i = 0, n=0;
  float count = 0;
  int numsongs = 0;
  char filenames [MAXSONGS+1][_TINYDIR_FILENAME_MAX];
  int filesizes [MAXSONGS+1];
  int songscores [MAXSONGS+1];
  float songmatch [MAXSONGS+1];
  clock_t start, diff;
  clock_t start_total, diff_total;
  int msec;
  int * hash_songs;

  printf("pazam_gpu.c running \n");
  if(argc<2)
  {
    printf("no excerpt file to open \n");
    exit(1);
  }
  start_total = clock();

  cudaError_t cuda_ret;

  cuda_ret = cudaMalloc((void**)&hash_songs, MAXELEMS*(MAXSONGS+1)*sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate output device memory");
  cuda_ret = cudaMemset(hash_songs, 0, MAXELEMS*(MAXSONGS+1)*sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to zero out device memory");

  printf("Generating hashes for original files.. \n");
  tinydir_dir dir;
  tinydir_open(&dir, "../data");
  while (dir.has_next)
  {
      tinydir_file file;
      tinydir_readfile(&dir, &file);
      if (file.is_reg)
      {
          numsongs++;
          start = clock();
          filesizes[numsongs] = generatehashes(file.path, numsongs, hash_songs);
          diff = clock() - start;
          msec = diff * 1000 / CLOCKS_PER_SEC;
          printf("%d:%d hashes for %s\n", numsongs, filesizes[numsongs], file.path);
          printf("Time taken: %d seconds %d milliseconds\n", msec/1000, msec%1000);
          strcpy(filenames[numsongs],file.name);
      }
      tinydir_next(&dir);
  }
  tinydir_close(&dir);
  printf("Generating hashes for recorded file.. \n");
  generatehashes(argv[1], 0, hash_songs);

  printf("Calculating score.. \n");

  int * songscores_d;
  cuda_ret = cudaMalloc((void**)&songscores_d, (MAXSONGS+1)*sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to allocate output device memory");
  cuda_ret = cudaMemset(songscores_d, 0, (MAXSONGS+1)*sizeof(int));
  if(cuda_ret != cudaSuccess) FATAL("Unable to zero out device memory");
  dim3 dim_grid, dim_block;
  dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
  dim_grid.x = (MAXELEMS - 1)/THREADS_PER_BLOCK + 1; dim_grid.y = dim_grid.z = 1;
  calc_scores<<<dim_grid, dim_block>>>(hash_songs, songscores_d);
  cuda_ret = cudaDeviceSynchronize();
  if(cuda_ret != cudaSuccess) FATAL("Kernel 1 failed");
  cuda_ret = cudaMemcpy(songscores, songscores_d, (MAXSONGS+1)*sizeof(int), cudaMemcpyDeviceToHost);
  if(cuda_ret != cudaSuccess) FATAL("Unable to copy output to host");
  cudaDeviceSynchronize();
  cudaFree(songscores_d);
  cudaFree(hash_songs);
  
  for(i = 1; i<=numsongs; i++){
    songmatch[i] = ((float)songscores[i])/((float)filesizes[i]);
    printf("Score for %s = %f\n", filenames[i], songmatch[i]);
    if(songmatch[i]>count){
      count = songmatch[i];
      n = i;
    }
  }
  printf("Best Score: %s\n", filenames[n]);
  diff_total = clock() - start_total;
  msec = diff_total * 1000 / CLOCKS_PER_SEC;
  printf("Total time taken: %d seconds %d milliseconds\n", msec/1000, msec%1000);

  /*for(i =0; i < MAXELEMS; i++)
  {
    free(hashtable[i]);
  }
  free(hashtable);*/

  /*if(argc<3) printf("fft1_wave done. new fingerprint.txt file written \n");
  else  printf("fft1_wave done. new %s file written \n", argv[2]);*/
    return 0;
} /* end fft1_wave .c */
