#include "Utils.h"

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

int main()  {
  int dim = 2;
  int n = dim*dim;

  double *a = (double *) alloc_aligned_16(n*sizeof(double));
  double *b = (double *) alloc_aligned_16(n*sizeof(double));
  double *c = (double *) alloc_aligned_16(n*sizeof(double));
  double *other_c = (double *) alloc_aligned_16(n*sizeof(double));

  rand_doubles(n, a);
  rand_doubles(n, b);
  rand_doubles(n, c);

  copy_buffer(n, c, other_c);

  mmmul_2x2(a, b, c);
  simple_mmmul(dim, a, b, other_c);

  print_square_mat(dim, c);
  printf("\n");
  print_square_mat(dim, other_c);

  double diff = diff_buffer(n, c, other_c);
  printf("diff = %f\n", diff);

  return 0;
}