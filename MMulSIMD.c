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

// Naive version, loops are now correct for column major order
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

void add_dot_4x4(int k, double *a, double *b, double *c, int lda, int ldb, int ldc)  {
  int p;

  for (p = 0; p < k; p++) {
        C(0, 0) += A(0, p) * B(p, 0);
        C(0, 1) += A(0, p) * B(p, 1);
        C(0, 2) += A(0, p) * B(p, 2);
        C(0, 3) += A(0, p) * B(p, 3);

        C(1, 0) += A(1, p) * B(p, 0);
        C(1, 1) += A(1, p) * B(p, 1);
        C(1, 2) += A(1, p) * B(p, 2);
        C(1, 3) += A(1, p) * B(p, 3);

        C(2, 0) += A(2, p) * B(p, 0);
        C(2, 1) += A(2, p) * B(p, 1);
        C(2, 2) += A(2, p) * B(p, 2);
        C(2, 3) += A(2, p) * B(p, 3);

        C(3, 0) += A(3, p) * B(p, 0);
        C(3, 1) += A(3, p) * B(p, 1);
        C(3, 2) += A(3, p) * B(p, 2);
        C(3, 3) += A(3, p) * B(p, 3);
      }
}

// 4x4 merged, inlined dot products
void mmmul_6(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=4) {
      add_dot_4x4(k, &A(i, 0), &B(0, j), &C(i, j), lda, ldb, ldc);
    }
  }
}

void add_dot_4x4_regs(int k, double *a, double *b, double *c, int lda, int ldb, int ldc)  {
  int p;
  register double
    c_00_reg, c_01_reg, c_02_reg, c_03_reg,
    c_10_reg, c_11_reg, c_12_reg, c_13_reg,
    c_20_reg, c_21_reg, c_22_reg, c_23_reg,
    c_30_reg, c_31_reg, c_32_reg, c_33_reg;
  register double
    a_0p_reg,
    a_1p_reg,
    a_2p_reg,
    a_3p_reg;

  c_00_reg = 0;
  c_01_reg = 0;
  c_02_reg = 0;
  c_03_reg = 0;
  c_10_reg = 0;
  c_11_reg = 0;
  c_12_reg = 0;
  c_13_reg = 0;
  c_20_reg = 0;
  c_21_reg = 0;
  c_22_reg = 0;
  c_23_reg = 0;
  c_30_reg = 0;
  c_31_reg = 0;
  c_32_reg = 0;
  c_33_reg = 0;

  for (p = 0; p < k; p++) {
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        c_00_reg += a_0p_reg * B(p, 0);
        c_01_reg += a_0p_reg * B(p, 1);
        c_02_reg += a_0p_reg * B(p, 2);
        c_03_reg += a_0p_reg * B(p, 3);

        c_10_reg += a_1p_reg * B(p, 0);
        c_11_reg += a_1p_reg * B(p, 1);
        c_12_reg += a_1p_reg * B(p, 2);
        c_13_reg += a_1p_reg * B(p, 3);

        c_20_reg += a_2p_reg * B(p, 0);
        c_21_reg += a_2p_reg * B(p, 1);
        c_22_reg += a_2p_reg * B(p, 2);
        c_23_reg += a_2p_reg * B(p, 3);

        c_30_reg += a_3p_reg * B(p, 0);
        c_31_reg += a_3p_reg * B(p, 1);
        c_32_reg += a_3p_reg * B(p, 2);
        c_33_reg += a_3p_reg * B(p, 3);
      }

      C(0, 0) += c_00_reg;
      C(0, 1) += c_00_reg;
      C(0, 2) += c_00_reg;
      C(0, 3) += c_00_reg;
      C(1, 0) += c_10_reg;
      C(1, 1) += c_11_reg;
      C(1, 2) += c_12_reg;
      C(1, 3) += c_13_reg;
      C(2, 0) += c_20_reg;
      C(2, 1) += c_21_reg;
      C(2, 2) += c_22_reg;
      C(2, 3) += c_23_reg;
      C(3, 0) += c_30_reg;
      C(3, 1) += c_31_reg;
      C(3, 2) += c_32_reg;
      C(3, 3) += c_33_reg;
}

// 4x4 merged, inlined dot products at a time with register storage
// for frequently used vars
void mmmul_7(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=4) {
      add_dot_4x4_regs(k, &A(i, 0), &B(0, j), &C(i, j), lda, ldb, ldc);
    }
  }
}

void add_dot_4x4_regs_bptrs(int k, double *a, double *b, double *c, int lda, int ldb, int ldc)  {
  int p;
  register double
    c_00_reg, c_01_reg, c_02_reg, c_03_reg,
    c_10_reg, c_11_reg, c_12_reg, c_13_reg,
    c_20_reg, c_21_reg, c_22_reg, c_23_reg,
    c_30_reg, c_31_reg, c_32_reg, c_33_reg;
  register double
    a_0p_reg,
    a_1p_reg,
    a_2p_reg,
    a_3p_reg,
    b_p0_reg,
    b_p1_reg,
    b_p2_reg,
    b_p3_reg;

  double
    *b_p0_pntr,
    *b_p1_pntr,
    *b_p2_pntr,
    *b_p3_pntr;

  c_00_reg = 0;
  c_01_reg = 0;
  c_02_reg = 0;
  c_03_reg = 0;
  c_10_reg = 0;
  c_11_reg = 0;
  c_12_reg = 0;
  c_13_reg = 0;
  c_20_reg = 0;
  c_21_reg = 0;
  c_22_reg = 0;
  c_23_reg = 0;
  c_30_reg = 0;
  c_31_reg = 0;
  c_32_reg = 0;
  c_33_reg = 0;

  b_p0_pntr = &B(0, 0);
  b_p1_pntr = &B(0, 1);
  b_p2_pntr = &B(0, 2);
  b_p3_pntr = &B(0, 3);

  for (p = 0; p < k; p++) {
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        c_00_reg += a_0p_reg * *b_p0_pntr;
        c_01_reg += a_0p_reg * *b_p1_pntr;
        c_02_reg += a_0p_reg * *b_p2_pntr;
        c_03_reg += a_0p_reg * *b_p3_pntr;

        c_10_reg += a_1p_reg * *b_p0_pntr;
        c_11_reg += a_1p_reg * *b_p1_pntr;
        c_12_reg += a_1p_reg * *b_p2_pntr;
        c_13_reg += a_1p_reg * *b_p3_pntr;

        c_20_reg += a_2p_reg * *b_p0_pntr;
        c_21_reg += a_2p_reg * *b_p1_pntr;
        c_22_reg += a_2p_reg * *b_p2_pntr;
        c_23_reg += a_2p_reg * *b_p3_pntr;

        c_30_reg += a_3p_reg * *b_p0_pntr++;
        c_31_reg += a_3p_reg * *b_p1_pntr++;
        c_32_reg += a_3p_reg * *b_p2_pntr++;
        c_33_reg += a_3p_reg * *b_p3_pntr++;
      }

      C(0, 0) += c_00_reg;
      C(0, 1) += c_00_reg;
      C(0, 2) += c_00_reg;
      C(0, 3) += c_00_reg;
      C(1, 0) += c_10_reg;
      C(1, 1) += c_11_reg;
      C(1, 2) += c_12_reg;
      C(1, 3) += c_13_reg;
      C(2, 0) += c_20_reg;
      C(2, 1) += c_21_reg;
      C(2, 2) += c_22_reg;
      C(2, 3) += c_23_reg;
      C(3, 0) += c_30_reg;
      C(3, 1) += c_31_reg;
      C(3, 2) += c_32_reg;
      C(3, 3) += c_33_reg;
}

// 4x4 merged, inlined dot products with register storage
// for frequently used vars and pointers for elements of b
void mmmul_8(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=4) {
      add_dot_4x4_regs(k, &A(i, 0), &B(0, j), &C(i, j), lda, ldb, ldc);
    }
  }
}

void add_dot_4x4_abcregs(int k, double *a, double *b, double *c, int lda, int ldb, int ldc)  {
  int p;
  register double
    c_00_reg, c_01_reg, c_02_reg, c_03_reg,
    c_10_reg, c_11_reg, c_12_reg, c_13_reg,
    c_20_reg, c_21_reg, c_22_reg, c_23_reg,
    c_30_reg, c_31_reg, c_32_reg, c_33_reg;
  register double
    a_0p_reg,
    a_1p_reg,
    a_2p_reg,
    a_3p_reg,
    b_p0_reg,
    b_p1_reg,
    b_p2_reg,
    b_p3_reg;

  double
    *b_p0_pntr,
    *b_p1_pntr,
    *b_p2_pntr,
    *b_p3_pntr;

  c_00_reg = 0;
  c_01_reg = 0;
  c_02_reg = 0;
  c_03_reg = 0;
  c_10_reg = 0;
  c_11_reg = 0;
  c_12_reg = 0;
  c_13_reg = 0;
  c_20_reg = 0;
  c_21_reg = 0;
  c_22_reg = 0;
  c_23_reg = 0;
  c_30_reg = 0;
  c_31_reg = 0;
  c_32_reg = 0;
  c_33_reg = 0;

  b_p0_pntr = &B(0, 0);
  b_p1_pntr = &B(0, 1);
  b_p2_pntr = &B(0, 2);
  b_p3_pntr = &B(0, 3);

  for (p = 0; p < k; p++) {
        a_0p_reg = A(0, p);
        a_1p_reg = A(1, p);
        a_2p_reg = A(2, p);
        a_3p_reg = A(3, p);

        b_p0_reg = *b_p0_pntr++;
        b_p1_reg = *b_p1_pntr++;
        b_p2_reg = *b_p2_pntr++;
        b_p3_reg = *b_p3_pntr++;

        c_00_reg += a_0p_reg * b_p0_reg;
        c_01_reg += a_0p_reg * b_p1_reg;
        c_02_reg += a_0p_reg * b_p2_reg;
        c_03_reg += a_0p_reg * b_p3_reg;

        c_10_reg += a_1p_reg * b_p0_reg;
        c_11_reg += a_1p_reg * b_p1_reg;
        c_12_reg += a_1p_reg * b_p2_reg;
        c_13_reg += a_1p_reg * b_p3_reg;

        c_20_reg += a_2p_reg * b_p0_reg;
        c_21_reg += a_2p_reg * b_p1_reg;
        c_22_reg += a_2p_reg * b_p2_reg;
        c_23_reg += a_2p_reg * b_p3_reg;

        c_30_reg += a_3p_reg * b_p0_reg;
        c_31_reg += a_3p_reg * b_p1_reg;
        c_32_reg += a_3p_reg * b_p2_reg;
        c_33_reg += a_3p_reg * b_p3_reg;
      }

      C(0, 0) += c_00_reg;
      C(0, 1) += c_00_reg;
      C(0, 2) += c_00_reg;
      C(0, 3) += c_00_reg;
      C(1, 0) += c_10_reg;
      C(1, 1) += c_11_reg;
      C(1, 2) += c_12_reg;
      C(1, 3) += c_13_reg;
      C(2, 0) += c_20_reg;
      C(2, 1) += c_21_reg;
      C(2, 2) += c_22_reg;
      C(2, 3) += c_23_reg;
      C(3, 0) += c_30_reg;
      C(3, 1) += c_31_reg;
      C(3, 2) += c_32_reg;
      C(3, 3) += c_33_reg;
}

// 4x4 merged, inlined dot products with register storage
// for frequently used vars and pointers for elements of b
void mmmul_9(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=4) {
      add_dot_4x4_abcregs(k, &A(i, 0), &B(0, j), &C(i, j), lda, ldb, ldc);
    }
  }
}

// Vector double array union
typedef union {
  __m128d v;
  double d[2];
} v2df_t;

// Uses SIMD instructions
void add_dot_4x4_vregs(int k, double *a, double *b, double *c, int lda, int ldb, int ldc)  {
  int p;
  v2df_t
    c_00_10_vreg, c_01_11_vreg, c_02_12_vreg, c_03_13_vreg,
    c_20_30_vreg, c_21_31_vreg, c_22_32_vreg, c_23_33_vreg;
  v2df_t
    a_0p_1p_vreg,
    a_2p_3p_vreg,
    b_p0_vreg,
    b_p1_vreg,
    b_p2_vreg,
    b_p3_vreg;

  double
    *b_p0_pntr,
    *b_p1_pntr,
    *b_p2_pntr,
    *b_p3_pntr;

  c_00_10_vreg.v = _mm_setzero_pd();
  c_01_11_vreg.v = _mm_setzero_pd();
  c_02_12_vreg.v = _mm_setzero_pd();
  c_03_13_vreg.v = _mm_setzero_pd();
  c_20_30_vreg.v = _mm_setzero_pd();
  c_21_31_vreg.v = _mm_setzero_pd();
  c_22_32_vreg.v = _mm_setzero_pd();
  c_23_33_vreg.v = _mm_setzero_pd();

  b_p0_pntr = &B(0, 0);
  b_p1_pntr = &B(0, 1);
  b_p2_pntr = &B(0, 2);
  b_p3_pntr = &B(0, 3);

  for (p = 0; p < k; p++) {
        a_0p_1p_vreg.v = _mm_load_pd((double *) &A(0, p));
        a_2p_3p_vreg.v = _mm_load_pd((double *) &A(2, p));

        b_p0_vreg.v = _mm_loaddup_pd((double *) b_p0_pntr++);
        b_p1_vreg.v = _mm_loaddup_pd((double *) b_p1_pntr++);
        b_p2_vreg.v = _mm_loaddup_pd((double *) b_p2_pntr++);
        b_p3_vreg.v = _mm_loaddup_pd((double *) b_p3_pntr++);

        // First and second rows
        c_00_reg += a_0p_reg * b_p0_reg;
        c_10_reg += a_1p_reg * b_p0_reg;

        c_01_reg += a_0p_reg * b_p1_reg;
        c_11_reg += a_1p_reg * b_p1_reg;

        c_02_reg += a_0p_reg * b_p2_reg;
        c_12_reg += a_1p_reg * b_p2_reg;

        c_03_reg += a_0p_reg * b_p3_reg;
        c_13_reg += a_1p_reg * b_p3_reg;

        // Third and fourth rows
        c_20_reg += a_2p_reg * b_p0_reg;
        c_30_reg += a_3p_reg * b_p0_reg;

        c_21_reg += a_2p_reg * b_p1_reg;
        c_31_reg += a_3p_reg * b_p1_reg;

        c_22_reg += a_2p_reg * b_p2_reg;
        c_32_reg += a_3p_reg * b_p2_reg;

        c_23_reg += a_2p_reg * b_p3_reg;
        c_33_reg += a_3p_reg * b_p3_reg;
      }

      C(0, 0) += c_00_10_vreg.d[0];
      C(0, 1) += c_;
      C(0, 2) += c_00_reg;
      C(0, 3) += c_00_reg;
      C(1, 0) += c_10_reg;
      C(1, 1) += c_11_reg;
      C(1, 2) += c_12_reg;
      C(1, 3) += c_13_reg;
      C(2, 0) += c_20_reg;
      C(2, 1) += c_21_reg;
      C(2, 2) += c_22_reg;
      C(2, 3) += c_23_reg;
      C(3, 0) += c_30_reg;
      C(3, 1) += c_31_reg;
      C(3, 2) += c_32_reg;
      C(3, 3) += c_33_reg;
}

// 4x4 merged using SIMD instructions
void mmmul_10(int m, int n, int k, double *a, double *b, double *c, int lda, int ldb, int ldc) {
  int i, j;
  for (j = 0; j < n; j+=4) {
    for (i = 0; i < m; i+=4) {
      add_dot_4x4_vregs(k, &A(i, 0), &B(0, j), &C(i, j), lda, ldb, ldc);
    }
  }
}

typedef struct {
  double time1;
  double time2;
} cmp_times;

char *method_1_name = "mmmul_8";
char *method_2_name = "mmmul_9";

void timed_mmmul(int n, cmp_times *times, double *a, double *b, double *c1, double *c2) {
  int size = n*n;

  clock_t begin, end;
  double method_1_time, method_2_time;

  begin = clock();
  mmmul_8(n, n, n, a, b, c1, n, n, n);
  end = clock();
  times->time1 = (double) (end - begin) / CLOCKS_PER_SEC;

  begin = clock();
  mmmul_9(n, n, n, a, b, c2, n, n, n);
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

  printf("Done with timing data\n");

  free(a);
  free(b);
  free(c1);
  free(c2);

  printf("Done freeing test data\n");

  save_results_to_file(num_dims, dimensions, method_1_name, m1_times, method_2_name, m2_times);
  return 0;
}