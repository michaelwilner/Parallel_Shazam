// PAZAM: A CUDA Music Identification Tool
// Michael Wilner - Cody Van Etten - Ahmed Suhyl
// ...
// Naive CPU Implementation based on Tyler Simon's WAV FFT Tool

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "fftc.h"
#include "tinydir.h"

#define N 1024
#define BYTES_PER_SAMPLE 2
#define MAXSONGS 10
#define MAXELEMS 200000
#define FREQBANDWIDTH 50
//static float A[2*N]; /* available for modifying transform */
//static float Z[2*N];

int FUZ_FACTOR = 2;

long hash1(long p1, long p2, long p3, long p4) {
  return (p4 - (p4 % FUZ_FACTOR)) * 100000000 + (p3 - (p3 % FUZ_FACTOR))
      * 100000 + (p2 - (p2 % FUZ_FACTOR)) * 100
      + (p1 - (p1 % FUZ_FACTOR));
}

unsigned long
hash(unsigned char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

int generatehashes(char *input_file, int** hashtable, int mysongid)
{
  float* Z = (float*) malloc(sizeof(float) * N * 2);
  int i, k, sect, sectcnt, first;
  int freq1, freq2, freq3, freq4, freq5, tempfreq, magnitude;
  int pt1,pt2,pt3,pt4, pt5;
  int numhashes = 0;
  FILE * inp;
  double sum, avg;
  double amplitude;
  char riff[4];
  int  sread, swrite; /* bytes read/written */
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
  int  ibyte;
  char more[4];
  int smin;
  int smax;
  int savg;
  int bad; /* flags bad data in read */
  int nbread; /* number of bytes read */
  
  inp = fopen(input_file, "rb");
  if(inp == NULL)
  {
    printf("can not open %s for reading. \n", input_file);
    return 0;
  }

  //printf("reading %s \n", input_file);
  
  sread = fread(&riff[0], 1, 4, inp);
  //printf("first 4 bytes should be RIFF, <%c%c%c%c>\n",riff[0],riff[1],riff[2],riff[3]);
  
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
  nbread = 44+nbytes;
  //printf("%d bytes read so far \n", nbread);
  
  bad = 0;
  savg = 0;
  sect = 0;
  sectcnt = 0;
  
  for(i=0; i<csize; i+=BYTES_PER_SAMPLE)
  {
    if(sect<N)
    {
      sread = fread(&soundin, 1, BYTES_PER_SAMPLE, inp); //We have to make sure we're reading both channels
      if(sread != BYTES_PER_SAMPLE && bad==0) { bad=1; printf("no read on byte %d \n", i); }
      Z[2*sect] = (float)soundin;
      Z[2*sect+1] = 0.0; /* no complex component */
      sect++;
    }
    else /* process N samples */
    {
      sect = 0;
      i-=BYTES_PER_SAMPLE;
      sectcnt++;
      /* transform */
/*      float * oldZ = Z;
      Z=fft_cooley_tukey(oldZ,N);
      free(oldZ);*/
      four1(Z,N);
      freq1 = freq2 = freq3 = freq4 = freq5 = 0;
      pt1 = pt2 = pt3 = pt4 = pt5 = 0;

      for(k=0; k<N; k++){
        tempfreq = (Z[2*k] > 0) ? (int)Z[2*k] : (int)(0-Z[2*k]);
        magnitude = (int)(log10((double)(tempfreq+1)) * 1000);
        if(k>=FREQBANDWIDTH && k<FREQBANDWIDTH*2 && magnitude>freq1) 
          {
            freq1 = magnitude; 
            pt1=k;
          }
        else if(k>=FREQBANDWIDTH*2 && k<FREQBANDWIDTH*3 && magnitude>freq2) 
          {
            freq2 = magnitude; 
            pt2=k;
          }
        else if(k>=FREQBANDWIDTH*3 && k<FREQBANDWIDTH*4 && magnitude>freq3) 
          {
            freq3 = magnitude; 
            pt3=k;
          }
        else if(k>=FREQBANDWIDTH*4 && k<FREQBANDWIDTH*5 && magnitude>freq4) 
          {
            freq4 = magnitude; 
            pt4=k;
          }
        else if(k>=FREQBANDWIDTH*5 && k<FREQBANDWIDTH*6 && magnitude>freq5) 
          {
            freq5 = magnitude; 
            pt5=k;
          }
      }
      char buffer [50];
      sprintf (buffer, "%d%d%d%d%d", pt1,pt2,pt3,pt4,pt5);
      unsigned long hashresult = hash(buffer) % MAXELEMS;
      int key = (int) hashresult;
      //printf ("key:%lu ",key);
      //printf("value:%d\n",sectcnt);
      if (key < 0)
        printf("Invalid key %d\n", key);

      int n = 0;
      hashtable[key][mysongid]++;
      numhashes++;
      
    } /* end else part to write out transformed values */
  } /* end i<csize loop */
  fclose(inp);
  free(Z);

  return numhashes;
}



int main(int argc, char * argv[])
{
  int ** hashtable;
  int i = 0, n=0;
  float count = 0;
  int numsongs = 0;
  char filenames [MAXSONGS+1][_TINYDIR_FILENAME_MAX];
  int filesizes [MAXSONGS+1];
  int songscores [MAXSONGS+1];
  float songmatch [MAXSONGS+1];
  clock_t start, diff;
  int msec;

  printf("pazam_cpu.c running \n");
  if(argc<2)
  {
    printf("no excerpt file to open \n");
    exit(1);
  }
  
  hashtable = (int **) calloc (MAXELEMS, sizeof(int *));
  for(i =0; i < MAXELEMS; i++)
  {
    hashtable[i] = (int *) calloc (MAXSONGS+1, sizeof(int));
  }

/*for(i = 0; i < MAXELEMS; i++)
{
  hashtable[i] = 0;
  excerpt[i] = 0;
}*/

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
          int numhashes = 0;
          filesizes[numsongs] = generatehashes(file.path, hashtable, numsongs);
          diff = clock() - start;
          msec = diff * 1000 / CLOCKS_PER_SEC;
          songscores[numsongs] = 0;
          printf("%d:%d hashes for %s\n", numsongs, filesizes[numsongs], file.path);
          printf("Time taken: %d seconds %d milliseconds\n", msec/1000, msec%1000);
          strcpy(filenames[numsongs],file.name);
      }
      tinydir_next(&dir);
  }
  tinydir_close(&dir);
  printf("Generating hashes for recorded file.. \n");
  generatehashes(argv[1], hashtable, 0);

  printf("Calculating score.. \n");
  
  for(i = 0; i < MAXELEMS; i++)
  {
    if(hashtable[i][0]>0)
      {
        //printf("Key: %d\n",i);
        for(n = 1; n <= MAXSONGS; n++)
        {
          //printf("%d ",hashtable[i][n]);
          songscores[n] = (hashtable[i][n]>=hashtable[i][0]) ? songscores[n]+hashtable[i][0] : songscores[n]+hashtable[i][n];
        }
        //printf("\n");
        /*for(n = 0; n < MAXCOLS; n++)
        {
          printf("%d ",excerpt[i][n]);
        }
        printf("\n\n");*/
      }
    //for(n = 1; n<=numsongs; n++) songseen[n] = 0;
  }
  for(i = 1; i<=numsongs; i++){
    songmatch[i] = ((float)songscores[i])/((float)filesizes[i]);
    printf("Score for %s = %f\n", filenames[i], songmatch[i]);
    if(songmatch[i]>count){
      count = songmatch[i];
      n = i;
    }
  }
  printf("Best Score: %s\n", filenames[n]);

  for(i =0; i < MAXELEMS; i++)
  {
    free(hashtable[i]);
  }
  free(hashtable);

  /*if(argc<3) printf("fft1_wave done. new fingerprint.txt file written \n");
  else  printf("fft1_wave done. new %s file written \n", argv[2]);*/
    return 0;
} /* end fft1_wave .c */
