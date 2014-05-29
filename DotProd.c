#include "Utils.h"

typedef union {
	__m128d v;
	double d[2];
} v2d_t;

double simd_dp(int n, double *a, double *b, double alpha)	{
	v2d_t a_1_2, b_1_2, res_prods;
	int i;
	for (i = 0; i < n; i+=4)	{
		a_1_2.v = _mm_load_pd(&a[i]);
		b_1_2.v = _mm_load_pd(&b[i]);
		res_prods.v = a_1_2.v * b_1_2.v;
		alpha += res_prods.d[0] + res_prods.d[1];

		a_1_2.v = _mm_load_pd(&a[i+2]);
		b_1_2.v = _mm_load_pd(&b[i+2]);
		res_prods.v = a_1_2.v * b_1_2.v;
		alpha += res_prods.d[0] + res_prods.d[1];
	}
	return alpha;
}

double scalar_dp(int n, double *a, double *b, double alpha)	{
	int i;
	for (i = 0; i < n; i += 4)	{
		alpha += a[i] * b[i];
		alpha += a[i+1] * b[i+1];
		alpha += a[i+2] * b[i+2];
		alpha += a[i+3] * b[i+3];
	}
	return alpha;
}

int main()	{
	int n = 100000000;
	double *a = (double *) alloc_aligned_16(n*sizeof(double));
	double *b = (double *) alloc_aligned_16(n*sizeof(double));
	rand_doubles(n, a);
	rand_doubles(n, b);
	double alpha = 0.0;
	printf("start computing\n");
	double scalar_res = scalar_dp(n, a, b, alpha);
	printf("scalar_res = %f\n", scalar_res);
	double SIMD_res = simd_dp(n, a, b, alpha);
	printf("SIMD_res = %f\n", SIMD_res);
	printf("diff = %f\n", SIMD_res - scalar_res);
	return 0;
}