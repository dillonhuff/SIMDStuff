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

typedef struct {
  double time1;
  double time2;
} cmp_times;

char *method_1_name = "mmmul_1";
char *method_2_name = "mmmul_2";

void timed_mmmul(int n, cmp_times *times, double *a, double *b, double *c1, double *c2) {
  int size = n*n;

  clock_t begin, end;
  double method_1_time, method_2_time;

  begin = clock();
  mmmul_1(n, n, n, a, b, c1, n, n, n);
  end = clock();
  times->time1 = (double) (end - begin) / CLOCKS_PER_SEC;

  begin = clock();
  mmmul_2(n, n, n, a, b, c2, n, n, n);
  end = clock();
  times->time2 = (double) (end - begin) / CLOCKS_PER_SEC;
}

void save_results_to_file(int n, int *dims, char *m1_name, double *m1_times, char *m2_name, double *m2_times)  {
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
  int num_dims = 130;
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