#ifndef UTILS_H
#define UTILS_H

#include <smmintrin.h>
#include <stdio.h>
#include <time.h>

void *alloc_aligned_16(size_t size);

void rand_doubles(int size, double *rands);

void simple_mmmul(int n, double *a, double *b, double *c);

double diff_buffer(int size, double *buf1, double *buf2);

void copy_buffer(int size, double *src, double *dest);

void print_square_mat(int dim, double *mat);

#endif