#include "Utils.h"

#define A(i,j) a[ (j)*lda + (i)]
#define B(i,j) b[ (j)*ldb + (i)]
#define C(i,j) c[ (j)*ldc + (i)]

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

typedef struct {
  double time1;
  double time2;
} cmp_times;

void timed_mmmul(int n, cmp_times *times, double *a, double *b, double *c1, double *c2) {
  int size = n*n;

  clock_t begin, end;
  double method_1_time, method_2_time;

  begin = clock();
  mmmul_1(n, n, n, a, b, c1, n, n, n);
  end = clock();
  times->time1 = (double) (end - begin) / CLOCKS_PER_SEC;

  begin = clock();
  mmmul_1(n, n, n, a, b, c2, n, n, n);
  end = clock();
  times->time2 = (double) (end - begin) / CLOCKS_PER_SEC;
}

int main()  {
  char *method_1_name = "mmmul_1";
  char *method_2_name = "mmmul_1";
  int num_dims = 130;
  int increments = 4;
  int dimensions[num_dims];
  int i;
  for (i = 1; i <= num_dims; i++)  {
    dimensions[i - 1] = 4*i;
  }

  // Allocate matrices used for testing
  int max_n = increments*num_dims;
  int max_size = max_n*max_n;
  double *a = (double *) malloc(max_size*sizeof(double));
  double *b = (double *) malloc(max_size*sizeof(double));
  double *c1 = (double *) malloc(max_size*sizeof(double));
  double *c2 = (double *) malloc(max_size*sizeof(double));

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
  return 0;
}