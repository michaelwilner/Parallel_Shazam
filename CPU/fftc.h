/* fftc.h          header file for various fft's and inverses */

float* fft_cooley_tukey(float* x, int N /* must be a power of 2 */);
void four1(float* data, int nn);
