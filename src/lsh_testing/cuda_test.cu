#include "cuda_proximity.h"
#include "cuda_timer.h"
#include <cudpp/cudpp.h>
#include <cutil.h>
#include "reduce_kernel.h"
#include <time.h>
#include "cuda_test.h"

void test_reduce_kernel() //must be the power of 2
{
	int n = 1024;
	float* data = new float[n];
	for(int i = 0; i < n; ++i)
	{
		data[i] = rand() / (float)RAND_MAX * 100;
	}
	
	for(int i = 0; i < n; ++i)
		printf("%f ", data[i]);
	printf("\n");
	
	float* d_data = NULL;
	GPUMALLOC((void**)&d_data, sizeof(float) * n);
	
	float* d_odata = NULL;
	GPUMALLOC((void**)&d_odata, sizeof(float) * n);
	
	float zero = 0;
	
	TOGPU(d_data, data, sizeof(float) * n);
	for(int i = 0; i < 100; ++i)
	{
		float result = reduce_max(d_data, n);
		int id = get_elem_index(d_data, n, result);
		printf("%f ", data[id]);
		TOGPU((d_data + id), &zero, sizeof(float));
	}
	printf("\n");
}

void testall()
{
	unsigned int nSamples  = 10000;
	unsigned int nQueries = 2000;
	unsigned int dim = 20;
	unsigned int K = 5;
	
	float* data = NULL;
	float* query = NULL;
	unsigned int* KNNResult = NULL;
	unsigned int* KNNResult_query = NULL;
	CPUMALLOC((void**)&data, sizeof(float) * nSamples * dim);
	CPUMALLOC((void**)&query, sizeof(float) * nQueries * dim);
	CPUMALLOC((void**)&KNNResult, sizeof(unsigned int) * nSamples * K);
	CPUMALLOC((void**)&KNNResult_query, sizeof(unsigned int) * nQueries * K);
	
	for(unsigned int i = 0; i < nSamples * dim; ++i)
	{
		data[i] = rand() / (RAND_MAX + 1.0f);
	}
	
	for(unsigned int i = 0; i < nQueries * dim; ++i)
	{
		query[i] = rand() / (RAND_MAX + 1.0f);
	}
	
	//FILE* filea = fopen("dist.txt", "w");
	//for(int i = 0; i < nSamples; ++i)
	//{
	//	for(int j = 0; j < nSamples; ++j)
	//	{
	//		float d = 0;
	//		for(int k = 0; k < dim; ++k)
	//		{
	//			d += (data[k * nSamples + i] - data[k * nSamples + j]) * (data[k * nSamples + i] - data[k * nSamples + j]);
	//		}
	//		fprintf(filea, "%f ", d);
	//	}
	//	fprintf(filea, "\n");
	//}
	//fclose(filea);
	//
	//FILE* fileb = fopen("dist2.txt", "w");
	//for(int i = 0; i < nQueries; ++i)
	//{
	//	for(int j = 0; j < nSamples; ++j)
	//	{
	//		float d = 0;
	//		for(int k = 0; k < dim; ++k)
	//		{
	//			d += (query[k * nQueries + i] - data[k * nSamples + j]) * (query[k * nQueries + i] - data[k * nSamples + j]);
	//		}
	//		fprintf(fileb, "%f ", d);
	//	}
	//	fprintf(fileb, "\n");
	//}
	//fclose(fileb);
	
	float* d_data = NULL;
	float* d_query = NULL;
	unsigned int* d_KNNResult = NULL;
	unsigned int* d_KNNResult_query = NULL;
	GPUMALLOC((void**)&d_data, sizeof(float) * nSamples * dim);
	GPUMALLOC((void**)&d_query, sizeof(float) * nQueries * dim);
	GPUMALLOC((void**)&d_KNNResult, sizeof(unsigned int) * nSamples * K);
	GPUMALLOC((void**)&d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
	
	TOGPU(d_data, data, sizeof(float) * nSamples * dim);
	TOGPU(d_query, query, sizeof(float) * nQueries * dim);


    //brute force
	{
		unsigned int timer = 0;
		startTimer(&timer);
			
		proximityComputation_bruteforce(d_data, nSamples,d_data, nSamples, dim, K, 0.0f, d_KNNResult);
		FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
		
		FILE* file1 = fopen("knn_bf.txt", "w");
		for(unsigned int i = 0; i < nSamples; ++i)
		{
			for(unsigned int j = 0; j < K; ++j)
			{
				fprintf(file1, "%d ", KNNResult[j * nSamples + i]);
			}
			fprintf(file1, "\n");
		}
		fclose(file1);
		
		proximityComputation_bruteforce(d_data, nSamples, d_query, nQueries, dim, K, 0.0f, d_KNNResult_query);
		FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
		
		FILE* file2 = fopen("knn_query_bf.txt", "w");
		for(unsigned int i = 0; i < nQueries; ++i)
		{
			for(unsigned int j = 0; j < K; ++j)
			{
				fprintf(file2, "%d ", KNNResult_query[j * nQueries + i]);
			}
			fprintf(file2, "\n");
		}
		fclose(file2);
		
		endTimer("brute-force KNN", &timer);
		
		unsigned int timer2 = 0;
		startTimer(&timer2);
		
		proximityComputation_bruteforce2(d_data, nSamples, d_data, nSamples, dim, K, 0.0f, d_KNNResult);	
		FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
		
		FILE* file3 = fopen("knn_bf2.txt", "w");
		for(unsigned int i = 0; i < nSamples; ++i)
		{
			for(unsigned int j = 0; j < K; ++j)
			{
				fprintf(file3, "%d ", KNNResult[j * nSamples + i]);
			}
			fprintf(file3, "\n");
		}
		fclose(file3);
		
		proximityComputation_bruteforce2(d_data, nSamples, d_query, nQueries, dim, K, 0.0f, d_KNNResult_query);
		FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
		
		FILE* file4 = fopen("knn_query_bf2.txt", "w");
		for(unsigned int i = 0; i < nQueries; ++i)
		{
			for(unsigned int j = 0; j < K; ++j)
			{
				fprintf(file4, "%d ", KNNResult_query[j * nQueries + i]);
			}
			fprintf(file4, "\n");
		}
		fclose(file4);
		
		endTimer("brute-force KNN", &timer2);
	}
	
	{
		unsigned int timer = 0;
		startTimer(&timer);
		
		float* h_lower = NULL;
		float* h_upper = NULL;
		CPUMALLOC((void**)&h_lower, sizeof(float) * dim);
		CPUMALLOC((void**)&h_upper, sizeof(float) * dim);
		
		for(unsigned int i = 0; i < dim; ++i)
		{
			h_upper[i] = 1;
			h_lower[i] = 0;
		}
		
		int LSH_L = 5;
		
		proximityComputation_LSH(d_data, nSamples, d_data, nSamples, dim, K, LSH_L, 0.0f, h_upper, h_lower, d_KNNResult);
		FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
		
		FILE* file1 = fopen("knn_lsh.txt", "w");
		for(unsigned int i = 0; i < nSamples; ++i)
		{
			for(unsigned int j = 0; j < K; ++j)
			{
				fprintf(file1, "%d ", KNNResult[j * nSamples + i]);
			}
			fprintf(file1, "\n");
		}
		fclose(file1);
	
	
		proximityComputation_LSH(d_data, nSamples, d_query, nQueries, dim, K, LSH_L, 0.0f, h_upper, h_lower, d_KNNResult_query);
		FROMGPU(KNNResult_query, d_KNNResult_query, sizeof(unsigned int) * nQueries * K);
		
		FILE* file2 = fopen("knn_query_lsh.txt", "w");
		for(unsigned int i = 0; i < nQueries; ++i)
		{
			for(unsigned int j = 0; j < K; ++j)
			{
				fprintf(file2, "%d ", KNNResult_query[j * nQueries + i]);
			}
			fprintf(file2, "\n");
		}
		fclose(file2);
		
		
		CPUFREE(h_lower);
		CPUFREE(h_upper);		
		endTimer("LSH KNN", &timer);
	}

	
	GPUFREE(d_data);
	GPUFREE(d_KNNResult);
	CPUFREE(data);
	CPUFREE(KNNResult);
	
	GPUFREE(d_query);
	GPUFREE(d_KNNResult_query);
	CPUFREE(query);
	CPUFREE(KNNResult_query);
	
}

void test_bruteforce()
{
	unsigned int nSamples  = 10000;
	unsigned int dim = 20;
	unsigned int K = 10;
	
	float* data = NULL;
	unsigned int* KNNResult = NULL;
	CPUMALLOC((void**)&data, sizeof(float) * nSamples * dim);
	CPUMALLOC((void**)&KNNResult, sizeof(unsigned int) * nSamples * K);
	
	for(unsigned int i = 0; i < nSamples * dim; ++i)
	{
		data[i] = rand() / (RAND_MAX + 1.0f);
	}
	
	float* d_data = NULL;
	unsigned int* d_KNNResult = NULL;
	GPUMALLOC((void**)&d_data, sizeof(float) * nSamples * dim);
	GPUMALLOC((void**)&d_KNNResult, sizeof(unsigned int) * nSamples * K);
	
	TOGPU(d_data, data, sizeof(float) * nSamples * dim);

	unsigned int timer = 0;
	startTimer(&timer);
		
	proximityComputation_bruteforce(d_data, nSamples, d_data, nSamples, dim, K, 0.0f, d_KNNResult);
	FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
	
	FILE* file = fopen("knn_bf.txt", "w");
	for(unsigned int i = 0; i < nSamples; ++i)
	{
		for(unsigned int j = 0; j < K; ++j)
		{
			fprintf(file, "%d ", KNNResult[j * nSamples + i]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
	
	endTimer("brute-force KNN", &timer);
	
	for(int i = 0; i < K; ++i)
	{
		int testid = 1;
		int knnid = KNNResult[i * nSamples + 1];
		float t = 0;
		for(int j = 0; j < dim; ++j)
		{
			t += (data[j * nSamples + 1] - data[j * nSamples + knnid]) * (data[j * nSamples + 1] - data[j * nSamples + knnid]);
		}
		printf("%f ", t);
	}
	printf("\n");
	
	unsigned int timer2 = 0;
	startTimer(&timer2);
	
	proximityComputation_bruteforce2(d_data, nSamples, d_data, nSamples, dim, K, 0.0f, d_KNNResult);	
	FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
	
	FILE* file2 = fopen("knn_bf2.txt", "w");
	for(unsigned int i = 0; i < nSamples; ++i)
	{
		for(unsigned int j = 0; j < K; ++j)
		{
			fprintf(file2, "%d ", KNNResult[j * nSamples + i]);
		}
		fprintf(file2, "\n");
	}
	fclose(file2);
	
	endTimer("brute-force KNN", &timer2);
	
	for(int i = 0; i < K; ++i)
	{
		int testid = 1;
		int knnid = KNNResult[i * nSamples + 1];
		float t = 0;
		for(int j = 0; j < dim; ++j)
		{
			t += (data[j * nSamples + 1] - data[j * nSamples + knnid]) * (data[j * nSamples + 1] - data[j * nSamples + knnid]);
		}
		printf("%f ", t);
	}
	printf("\n");

	
	GPUFREE(d_data);
	GPUFREE(d_KNNResult);
	CPUFREE(data);
	CPUFREE(KNNResult);
}

void test_LSH()
{	
	unsigned int nSamples = 50000;
	unsigned int dim = 20;
	unsigned int K = 10;
	
	float* data = NULL;
	unsigned int* KNNResult = NULL;
	CPUMALLOC((void**)&data, sizeof(float) * nSamples * dim);
	CPUMALLOC((void**)&KNNResult, sizeof(unsigned int) * nSamples * K);
	
	for(unsigned int i = 0; i < nSamples * dim; ++i)
	{
		data[i] = rand() / (RAND_MAX + 1.0f);
	}
	

	
	float* d_data = NULL;
	unsigned int* d_KNNResult = NULL;
	GPUMALLOC((void**)&d_data, sizeof(float) * nSamples * dim);
	GPUMALLOC((void**)&d_KNNResult, sizeof(unsigned int) * nSamples * K);
	
	TOGPU(d_data, data, sizeof(float) * nSamples * dim);
	
	float* h_lower = NULL;
	float* h_upper = NULL;
	CPUMALLOC((void**)&h_lower, sizeof(float) * dim);
	CPUMALLOC((void**)&h_upper, sizeof(float) * dim);
	
	for(unsigned int i = 0; i < dim; ++i)
	{
		h_upper[i] = 1;
		h_lower[i] = 0;
	}
	
	int LSH_L = 5;
	
	unsigned int timer = 0;
	startTimer(&timer);
	
	proximityComputation_LSH(d_data, nSamples, d_data, nSamples, dim, K, LSH_L, 0.0f, h_upper, h_lower, d_KNNResult);
	
	endTimer("LSH KNN", &timer);
	
	FROMGPU(KNNResult, d_KNNResult, sizeof(unsigned int) * nSamples * K);
	
	FILE* file = fopen("knn_lsh.txt", "w");
	for(unsigned int i = 0; i < nSamples; ++i)
	{
		for(unsigned int j = 0; j < K; ++j)
		{
			fprintf(file, "%d ", KNNResult[j * nSamples + i]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
	
	GPUFREE(d_data);
	GPUFREE(d_KNNResult);
	
	CPUFREE(h_lower);
	CPUFREE(h_upper);
	CPUFREE(data);
	CPUFREE(KNNResult);
	
}

void test_Grouping()
{
	int N = 1000000;
	unsigned int* data = new unsigned int[N];
	for(int i = 0; i < N; ++i)
	{
		data[i] = rand() % 10;
	}
	
	//for(int i = 0; i < N; ++i)
	//	printf("%d ", data[i]);
	//printf("\n");
	
	unsigned int* d_data = NULL;
	GPUMALLOC((void**)&d_data, sizeof(unsigned int) * N);
	TOGPU(d_data, data, sizeof(unsigned int) * N);
	
	unsigned int* d_indices = NULL;
	GPUMALLOC((void**)&d_indices, sizeof(unsigned int) * N);
	
	dim3 grid = makeGrid((int)ceilf(N / (float)PROXIMITY_THREADS));
	dim3 threads = dim3(PROXIMITY_THREADS, 1, 1);
	computePermutation <<< grid, threads>>>(N, d_indices);
	
	unsigned int timer1 = 0;
	startTimer(&timer1);
	hashingBasedElementGrouping(d_data, d_indices, N);
	endTimer("1", &timer1);
	
	
	//FROMGPU(data, d_data, sizeof(unsigned int) * N);
	//for(int i = 0; i < N; ++i)
	//	printf("%d ", data[i]);
	//printf("\n");
	
	
	//return;
	
	
	TOGPU(d_data, data, sizeof(unsigned int) * N);
	
	computePermutation <<< grid, threads>>>(N, d_indices);
	
	unsigned int timer2 = 0;
	startTimer(&timer2);
	nvRadixSort::RadixSort rdsort(N);
	rdsort.sort(d_data, d_indices, N, 32);
	endTimer("2", &timer2);
	
	

	GPUFREE(d_data);
	delete [] data;
	
}