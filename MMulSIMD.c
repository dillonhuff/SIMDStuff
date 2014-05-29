#include "Utils.h"

#define A(i,j) a[ (j)*lda + (i)]
#define B(i,j) b[ (j)*ldb + (i)]
#define C(i,j) c[ (j)*ldc + (i)]

// Naive version, loops are not even ordered well
void mmmul_1(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j, p;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (p = 0; p < k; p++) {
        C(i, j) += A(i, p) + B(p, j);
      }
    }
  }
}

// Naive version, loops are now correct
void mmmul_2(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j, p;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      for (p = 0; p < k; p++) {
        C(i, j) += A(i, p) + B(p, j);
      }
    }
  }
}

void add_dot(int k, double *a, double *b, double *c, int a_stride)  {
  int p;
  double tmp = *c;
  for (p = 0; p < k; p++) {
    tmp += a[a_stride*p] + b[p];
  }
  *c = tmp;
}

// Uses Add dot subroutine
void mmmul_3(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      add_dot(k, &A(i, 0), &B(0, j), &C(i, j), lda);
    }
  }
}

// Uses Add dot subroutine, unrolled by 4
void mmmul_4(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=1) {
      add_dot(k, &A(i, 0), &B(0, j), &C(i, j), lda);
      add_dot(k, &A(i, 0), &B(0, j+1), &C(i, j+1), lda);
      add_dot(k, &A(i, 0), &B(0, j+2), &C(i, j+2), lda);
      add_dot(k, &A(i, 0), &B(0, j+3), &C(i, j+3), lda);
    }
  }
}

void add_dot_1x4(int k, double *a, double *b, double *c, int lda, int ldb, int ldc)  {
  int p;
  for (p = 0; p < k; p++) {
    C(0, 0) += A(0, p) * B(p, 0);
    C(0, 1) += A(0, p) * B(p, 1);
    C(0, 2) += A(0, p) * B(p, 2);
    C(0, 3) += A(0, p) * B(p, 3);
  }
}

// Merged, unrolled dot products
void mmmul_5(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=1) {
      add_dot_1x4(k, &A(i, 0), &B(0, j), &C(i, j), lda, ldb, ldc);
    }
  }
}

typedef struct {
  double time1;
  double time2;
} cmp_times;

char *method_1_name = "mmmul_4";
char *method_2_name = "mmmul_5";

void timed_mmmul(int n, cmp_times *times, double *a, double *b, double *c1, double *c2) {
  int size = n*n;

  clock_t begin, end;
  double method_1_time, method_2_time;

  begin = clock();
  mmmul_4(n, n, n, a, b, c1, n, n, n);
  end = clock();
  times->time1 = (double) (end - begin) / CLOCKS_PER_SEC;

  begin = clock();
  mmmul_5(n, n, n, a, b, c2, n, n, n);
  end = clock();
  times->time2 = (double) (end - begin) / CLOCKS_PER_SEC;
}

void save_results_to_file(int n, int *dims, char *m1_name, double *m1_times, char *m2_name, double *m2_times)  {
  printf("Saving results\n");
  char file_name[200];
  strcpy(file_name, "gemm_time_cmp_");
  strcat(file_name, m1_name);
  strcat(file_name, "_");
  strcat(file_name, m2_name);
  strcat(file_name, ".csv");

  FILE *out_file = fopen(file_name, "w");
  fprintf(out_file, "dim,%s,%s\n", m1_name, m2_name);

  int i;
  for (i = 0; i < n; i++) {
    fprintf(out_file, "%d,%f,%f\n", dims[i], m1_times[i], m2_times[i]);
  }

  fclose(out_file);
}

int main()  {
  int num_dims = 150;
  int increments = 4;
  int dimensions[num_dims];
  int i;
  for (i = 1; i <= num_dims; i++)  {
    dimensions[i - 1] = increments*i;
  }

  // Allocate matrices used for testing
  int max_n = increments*num_dims;
  int max_size = max_n*max_n;
  double *a = (double *) alloc_aligned_16(max_size*sizeof(double));
  double *b = (double *) alloc_aligned_16(max_size*sizeof(double));
  double *c1 = (double *) alloc_aligned_16(max_size*sizeof(double));
  double *c2 = (double *) alloc_aligned_16(max_size*sizeof(double));

  rand_doubles(max_size, a);
  rand_doubles(max_size, b);
  rand_doubles(max_size, c1);
  copy_buffer(max_size, c1, c2);

  cmp_times times;
  double m1_times[num_dims], m2_times[num_dims];
  for (i = 0; i < num_dims; i++)  {
    timed_mmmul(dimensions[i], &times, a, b, c1, c2);
    printf("%s time = %f for n = %d\n", method_1_name, times.time1, dimensions[i]);
    printf("%s time = %f for n = %d\n", method_2_name, times.time2, dimensions[i]);
    printf("\n");
    m1_times[i] = times.time1;
    m2_times[i] = times.time2;
  }

  free(a);
  free(b);
  free(c1);
  free(c2);

  save_results_to_file(num_dims, dimensions, method_1_name, m1_times, method_2_name, m2_times);
  return 0;
}