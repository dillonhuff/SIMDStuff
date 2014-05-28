#include <stdio.h>
#include <smmintrin.h>

// C = AB + C for 2 double precision matrices in
// row major order
inline void gemm_2x2(double *a, double *b, double *c) {
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

int main()  {
  
  double a[4];
  double b[4];
  double c[4];

  a[0] = 1.0;
  a[1] = 2.0;
  a[2] = 3.0;
  a[3] = 4.0;

  b[0] = 5.0;
  b[1] = 6.0;
  b[2] = 7.0;
  b[3] = 8.0;

  c[0] = 9.0;
  c[1] = 10.0;
  c[2] = 11.0;
  c[3] = 12.0;

  gemm_2x2(a, b, c);

  printf("c[0] = %f\nc[1] = %f\nc[2] = %f\nc[3] = %f\n", c[0], c[1], c[2], c[3]);
  return 0;
}