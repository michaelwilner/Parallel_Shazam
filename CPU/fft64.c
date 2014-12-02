/* fft64.c  complex data stored  re, im, re, im, re, ... im */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float* fft_cooley_tukey(float* x, int N /* must be a power of 2 */) {
    float* X = (float*) malloc(sizeof(float) * N * 2);
    float * d, * e, * D, * E;
    int k;

    if (N == 1) { //base case
        X[0] = x[0];
        X[1] = x[1];
        return X;
    }

    //Divide
    e = (float*) malloc(sizeof(float) * N);
    d = (float*) malloc(sizeof(float) * N);
    for(k = 0; k < N/2; k++) {
        e[2*k] = x[2*(2*k)];
        e[2*k+1] = x[2*(2*k) + 1];
        d[2*k] = x[2*(2*k+1)];
        d[2*k+1] = x[2*(2*k+1) + 1];
    }
    //Conquer
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
