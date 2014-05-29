#include "Utils.h"

double simp_dp(int n, double *a, double *b, double alpha)	{
	int i;
	for (i = 0; i < n; i++)	{
		alpha += a[i] * b[i];
	}
	return alpha;
}

int main()	{
	int n = 1000;
	double *a = aligned_malloc_16(n*sizeof(double));
	double *b = aligned_malloc_16(n*sizeof(double));
	alpha = 0.0;
	res = simp_dp(n, a, b, alpha);
	printf("res = %f\n", res);
	return 0;
}