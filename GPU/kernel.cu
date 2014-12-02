#define BLOCK_SIZE 1024
#define N 1024
#define MAXELEMS 200000
#define FREQBANDWIDTH 50
#define MAXSONGS 10

__device__ char nthdigit(int x, int n);
__device__ int generate_hash_string (char* buffer, int a, int b, int c, int d, int e);
__device__ void four1(float* data, int nn);
__device__ unsigned long long int hash(char *str);
__global__ void calc_scores(int * hash_songs, int * songscores_d)
{

    __shared__ int data[MAXSONGS+1];

    if (threadIdx.x < MAXSONGS+1) data[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int hashsongs_base_index = i*(MAXSONGS+1);
    int temp;
    if(i<MAXELEMS && hash_songs[hashsongs_base_index]>0)
    {
        int n;
        for(n = 1; n <= MAXSONGS; n++)
        {
          temp = (hash_songs[hashsongs_base_index+n]>=hash_songs[hashsongs_base_index]) ? hash_songs[hashsongs_base_index] : hash_songs[hashsongs_base_index+n];
          atomicAdd(&(data[n]),temp);
        }
    }
    __syncthreads();
    if (threadIdx.x < MAXSONGS+1) atomicAdd(&(songscores_d[threadIdx.x]),data[threadIdx.x]);
}

__device__ char nthdigit(int x, int n)
{
    int powersof10[] = {1, 10, 100, 1000};
    return ((x / powersof10[n]) % 10) + '0';
}

__device__ int generate_hash_string (char* buffer, int a, int b, int c, int d, int e)
{
  int i = 0;
  if(buffer == NULL)
    return 0;
  if(a >= 10)
  {
    buffer[i++] = nthdigit(a,1);
  }
  else
  {
    buffer[i++] = nthdigit(a,0);
  }
  if(b >= 100)
  {
    buffer[i++] = nthdigit(b,0);
  }
  else if(b >= 10)
  {
    buffer[i++] = nthdigit(b,1);
  }
  else
  {
    buffer[i++] = nthdigit(b,2);
  } 
  if(c >= 100)
  {
    buffer[i++] = nthdigit(c,0);
  }
  else if(c >= 10)
  {
    buffer[i++] = nthdigit(c,1);
  }
  else
  {
    buffer[i++] = nthdigit(c,2);
  } 
  if(d >= 100)
  {
    buffer[i++] = nthdigit(d,0);
  }
  else if(d >= 10)
  {
    buffer[i++] = nthdigit(d,1);
  }
  else
  {
    buffer[i++] = nthdigit(d,2);
  }
  if(e >= 100)
  {
    buffer[i++] = nthdigit(e,0);
  }
  else if(e >= 10)
  {
    buffer[i++] = nthdigit(e,1);
  }
  else
  {
    buffer[i++] = nthdigit(e,2);
  } 

  return i;
}

__global__ void parallelhash(float* in, int* out, int n, int* hash_table, int song_id)
{

  int i, k;
  float freq1, freq2, freq3, freq4, freq5;
  float tempfreq, magnitude;
  int pt1,pt2,pt3,pt4, pt5, key;

  i = threadIdx.x + blockIdx.x*blockDim.x; //My chunk

  if(i < n) //if my chunk ID < total number of chunks do stuff
  {

    float* Z = &in[N*i];      
    four1(Z,N);
    freq1 = freq2 = freq3 = freq4 = freq5 = 0;
    pt1 = pt2 = pt3 = pt4 = pt5 = 0;

    for(k=FREQBANDWIDTH; k<FREQBANDWIDTH*6; k++){
    tempfreq = abs(Z[2*k]);
    magnitude = log10(tempfreq+1);
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
    unsigned long long int hashresult = 0;
    char buffer [15];
    int k = 0, j = 0;
    for(k = 0; k < 15; k ++) buffer[0] = 0;
    j = generate_hash_string (buffer, pt1,pt2,pt3,pt4,pt5);
    /* hashresult = hash(buffer) % MAXELEMS;*/
    unsigned long long int hash = 5381;
    k = 0;
    for(k = 0; k < j; k ++)
        hash = ((hash << 5) + hash) + buffer[k]; /* hash * 33 + c */
    hashresult = hash % MAXELEMS;
    key = (int) hashresult;
    out[i] = key;
    atomicAdd(&(hash_table[(key*(MAXSONGS+1))+(song_id)]),1);
  }


  
}

__device__  unsigned long long int hash(char *str)
{
     unsigned long long int hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

__device__ void four1(float* data, int nn)
{
    int n, mmax, m, j, istep, i;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;
 
    // reverse-binary reindexing
    n = nn<<1;
    j=1;
    for (i=1; i<n; i+=2) 
    {
        if (j>i) //data swap should be parallelizeable
        {
            float temp = data[j-1];
            data[j-1] = data[i-1];
            data[i-1] = temp;
            temp = data[j];
            data[j] = data[i];
            data[i] = temp;
        }
        m = nn;
        while (m>=2 && j>m) 
        {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
 
    // here begins the Danielson-Lanczos section
    mmax=2;
    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*data[j-1] - wi*data[j];
                tempi = wr * data[j] + wi*data[j-1];
 
                data[j-1] = data[i-1] - tempr;
                data[j] = data[i] - tempi;
                data[i-1] += tempr;
                data[i] += tempi;
            }
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }
        mmax=istep;
    }
}

