/* fft64.c  complex data stored  re, im, re, im, re, ... im */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void fft64(float Z[128]) /* input data points and output  [0] to [127] */
{
  static float W[128];    /* scratch vector, used many times, */
  static float E[66] =    /* constants for FFT algorithm */
    {1.0,      0.0,       0.995185,  0.0980171,
     0.980785, 0.19509,   0.95694,   0.290285,
     0.92388,  0.382683,  0.881921,  0.471397,
     0.83147,  0.55557,   0.77301,   0.634393,
     0.707107, 0.707107,  0.634393,  0.77301,
     0.55557,  0.83147,   0.471397,  0.881921,
     0.382683, 0.92388,   0.290285,  0.95694,
     0.19509,  0.980785,  0.0980171, 0.995185,
     0.0,      1.0,      -0.0980171, 0.995185,
    -0.19509,  0.980785, -0.290285,  0.95694,
    -0.382683, 0.92388,  -0.471397,  0.881921,
    -0.55557,  0.83147,  -0.634393,  0.77301,
    -0.707107, 0.707107, -0.77301,   0.634393,
    -0.83147,  0.55557,  -0.881921,  0.471397,
    -0.92388,  0.382683, -0.95694,   0.290285,
    -0.980785, 0.19509,  -0.995185,  0.0980171,
    -1.0, 0.0};
  float Tre, Tim;
  int i, j, k, l, m;

   m = 32;
   l = 1;
   while(1)
   { 
      k = 0;
      j = l;
      i = 0;
      while(1)
      { 
         while(1)
	 {
	   /* W[i+k] = Z[i] + Z[m+i]; complex */
           W[2*(i+k)]   = Z[2*i]   + Z[2*(m+i)];
           W[2*(i+k)+1] = Z[2*i+1] + Z[2*(m+i)+1];

           /* W[i+j] = E[k] * (Z[i] - Z[m+i]); complex */
           Tre = Z[2*i]   - Z[2*(m+i)];
           Tim = Z[2*i+1] - Z[2*(m+i)+1];
           W[2*(i+j)]   = E[2*k] * Tre - E[2*k+1] * Tim;
           W[2*(i+j)+1] = E[2*k] * Tim + E[2*k+1] * Tre; 
           i++;
           if(i >= j) break;
	 }
         k = j;
         j = k+l;
         if(j > m) break;
      }
      l = l+l;
                  /* work back other way without copying */
      k = 0;
      j = l;
      i = 0;
      while(1)
      {
        while(1)
	{
          /* Z[i+k] = W[i] + W[m+i]; complex */
          Z[2*(i+k)]   = W[2*i]   + W[2*(m+i)];
          Z[2*(i+k)+1] = W[2*i+1] + W[2*(m+i)+1];

          /* Z[i+j] = E[k] * (W[i] - W[m+i]); complex */
          Tre = W[2*i]   - W[2*(m+i)];
          Tim = W[2*i+1] - W[2*(m+i)+1];
          Z[2*(i+j)]   = E[2*k] * Tre - E[2*k+1] * Tim;
          Z[2*(i+j)+1] = E[2*k] * Tim + E[2*k+1] * Tre;
          i++;
          if(i >= j) break;
	}
        k = j;
        j = k+l;
        if(j > m) break;
      }
      l = l+l;
      if(l > m) break; // result is in Z
   }
} /* end fft64 */

float* fft_cooley_tukey(float* x, int N /* must be a power of 2 */) {
    float* X = (float*) malloc(sizeof(float) * N * 2);
    float * d, * e, * D, * E;
    int k;

    if (N == 1) {
        X[0] = x[0];
        X[1] = x[1];
        return X;
    }

    e = (float*) malloc(sizeof(float) * N);
    d = (float*) malloc(sizeof(float) * N);
    for(k = 0; k < N/2; k++) {
        e[2*k] = x[2*(2*k)];
        e[2*k+1] = x[2*(2*k) + 1];
        d[2*k] = x[2*(2*k+1)];
        d[2*k+1] = x[2*(2*k+1) + 1];
    }

    E = fft_cooley_tukey(e, N/2);
    D = fft_cooley_tukey(d, N/2);
    
    for(k = 0; k < N/2; k++) {
        /* Multiply entries of D by the twiddle factors e^(-2*pi*i/N * k) */
        float right_re = D[2*k];
        float right_im = D[2*k+1];
        float left_re = cos(-2.0*M_PI*k/N);
        float left_im = sin(-2.0*M_PI*k/N);
        D[2*k] = left_re*right_re - left_im*right_im;
        D[2*k+1] = left_re*right_im + left_im*right_re;
    }

    for(k = 0; k < N/2; k++) {
        //complex add
        X[2*k] = E[2*k] + D[2*k];
        X[2*k+1] = E[2*k+1] + D[2*k+1];
        //complex subtract
        X[2*(k+N/2)] = E[2*(k+N/2)] - D[2*(k+N/2)];
        X[2*(k+N/2)+1] = E[2*(k+N/2)+1] - D[2*(k+N/2)+1];
    }

    free(D);
    free(E);
    return X;
}

void four1(float* data, int nn)
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
