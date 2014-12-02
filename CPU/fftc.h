/* fftc.h          header file for various fft's and inverses */
void  fft8  (short int Z[16]);   /* standard complex FFT */

void  fft16  (float Z[32]);   /* standard complex FFT */
void  fft32  (float Z[64]);   /* standard complex FFT */
void  fft64  (float Z[128]);  /* standard complex FFT */
void  fft128 (float Z[256]);  /* standard complex FFT */
void  fft256 (float Z[512]);  /* standard complex FFT */
void  fft512 (float Z[1024]); /* standard complex FFT */
void  fft1024(float Z[2048]); /* standard complex FFT */
void  fft2048(float Z[4096]); /* standard complex FFT */
void  fft4096(float Z[8192]); /* standard complex FFT */

float* fft_cooley_tukey(float* x, int N /* must be a power of 2 */);
void four1(float* data, int nn);
