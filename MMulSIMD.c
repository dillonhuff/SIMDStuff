#include <math.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <smmintrin.h>

// B = A + B for A, B of size 4
inline void sum_4(double *a, double *b)  {
  __m128d b_00_01, b_10_11;

  b_00_01 = _mm_load_pd(&b[0]);
  b_10_11 = _mm_load_pd(&b[2]);

  b_00_01 += _mm_load_pd(&a[0]);
  b_10_11 += _mm_load_pd(&a[2]);

  _mm_store_pd(&b[0], b_00_01);
  _mm_store_pd(&b[2], b_10_11);
}

// C = AB + C for 2 double precision matrices in row major order
inline void mmmul_2x2(double *a, double *b, double *c) {
  __m128d c_00_01, c_10_11;
  __m128d a_00, a_01, a_10, a_11;
  __m128d b_00_01, b_10_11;

  c_00_01 = _mm_load_pd(&c[0]);
  c_10_11 = _mm_load_pd(&c[2]);

  a_00 = _mm_loaddup_pd(&a[0]);
  a_01 = _mm_loaddup_pd(&a[1]);
  a_10 = _mm_loaddup_pd(&a[2]);
  a_11 = _mm_loaddup_pd(&a[3]);

  b_00_01 = _mm_load_pd(&b[0]);
  b_10_11 = _mm_load_pd(&b[2]);

  c_00_01 += a_00 * b_00_01;
  c_10_11 += a_10 * b_00_01;

  c_00_01 += a_01 * b_10_11;
  c_10_11 += a_11 * b_10_11;

  _mm_store_pd(&c[0], c_00_01);
  _mm_store_pd(&c[2],c_10_11);
}

inline void mmmul_4x4(double *a, double *b, double *c) {
  printf ("starting\n");

  mmmul_2x2(&a[0], &b[0], &c[0]);
  mmmul_2x2(&a[4], &b[8], &c[0]);

  mmmul_2x2(&a[0], &b[4], &c[4]);
  mmmul_2x2(&a[4], &b[12], &c[4]);

  mmmul_2x2(&a[8], &b[0], &c[8]);
  mmmul_2x2(&a[12], &b[8], &c[8]);

  mmmul_2x2(&a[8], &b[4], &c[12]);
  mmmul_2x2(&a[12], &b[12], &c[12]);
}

void rand_doubles(int size, double *rands) {
  int i;
  for (i = 0; i < size; i++) {
    rands[i] = rand() % 10;
  }
}

void simple_mmmul(int n, double *a, double *b, double *c)  {
  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        c[i*n + j] += a[i*n + k] * b[k*n + j];
      }
    }
  }
}

double diff_buffer(int size, double *buf1, double *buf2)  {
  int i;
  double diff = 0.0;
  for (i = 0; i < size; i++)  {
    diff += abs(buf1[i] - buf2[i]);
  }
  return diff;
}

void copy_buffer(int size, double *src, double *dest) {
  int i;
  for (i = 0; i < size; i++) {
    dest[i] = src[i];
  }
}

int main()  {
  double *a = (double *) _aligned_malloc(16*sizeof(double), 16);
  double *b = (double *) _aligned_malloc(16*sizeof(double), 16);
  double *c = (double *) _aligned_malloc(16*sizeof(double), 16);
  double *other_c = (double *) _aligned_malloc(16*sizeof(double), 16);

  rand_doubles(16, a);
  rand_doubles(16, b);
  rand_doubles(16, c);

  copy_buffer(16, c, other_c);

  mmmul_2x2(a, b, c);
  simple_mmmul(2, a, b, other_c);

  double diff = diff_buffer(16, c, other_c);
  printf("diff = %f\n", diff);

  return 0;
}