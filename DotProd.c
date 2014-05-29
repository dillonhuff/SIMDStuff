#include "Utils.h"

double simp_dp(int n, double *a, double *b, double alpha)	{
	int i;
	for (i = 0; i < n; i++)	{
		alpha += a[i] * b[i];
	}
	return alpha;
}

int main()	{
	return 0;
}