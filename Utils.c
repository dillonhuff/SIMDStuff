#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void *alloc_aligned_16(size_t size) {
  return (void *) _aligned_malloc(size, 16);
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

void print_square_mat(int dim, double *mat) {
  int i, j;
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      printf("%-5.1f ", mat[dim*i + j]);
    }
    printf("\n");
  }
}