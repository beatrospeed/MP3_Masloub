
#include "cuda_runtime.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <random>
#include <algorithm>
#include <chrono>
#include <map>




__global__ void MatrixMultElement(float* A, float* B, float* C, int w) {
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	float Cval = 0;
	if (row < w && col < w)
	{
		for (int i = 0; i < w; i++)
		{
			Cval += A[row * w + i] * B[i * w + col];
			//printf("Element Compairason:  matrix N: %f------ matrix M: %f matrix R:------ %f \n", A[idx], B[idx], C[idx]);
		}
		
	}
	C[row * w + col] = Cval;
}



void sumMatrixOnHost(float* A, float* B, float* C, int w)
{
	float* N = A;
	float* M = B;
	float* R = C;
	float Cval = 0;

	for (int k = 0; k < w; k++)
	{
		for (int p = 0; p < w; p++)
		{
		    Cval = 0; 
			for (int i = 0; i < w; i++){
			Cval += N[k*w + i] * M[i*w+p];
			//printf("row: %f, col: %f, Result: %f\n\n", N[k * w + i], M[i * w + p], Cval); 
			}

		  R[k*w + p] = Cval; 
		}
		
	}
	return;
}

void checkResult(float* CPU, float* GPU, const int dim) {
	double Margin = 1.0E-8;
	for (int i = 0; i < dim; i++)
	{
		if (abs(CPU[i] - GPU[i]) > Margin)
		{
			printf("CPU %f GPU %f ", CPU[i], GPU[i]);
			printf("Matricies do not match.\n\n");
			break;
		}
		
			
		
	}
	printf("Test PASSED\n\n");
}

void initialData(float* Matrix, const int dim)
{
	int i;
	for (i = 0; i < dim; i++)
	{
		Matrix[i] = (float)(rand() & 0xFF) / 10.0f;
	}

}


void TimeDataTransfer(int S) {

	float* H_N, * H_M, * H_R, * H_R1;
	//Size of matrix dimension i.e. 1024


	// Multiply each dimension to get the matrix and then multiply by size of int to get the value in bytes

	size_t sizeInFloats = S * S * sizeof(float);
	//input host vector N

	H_N = (float*)malloc(sizeInFloats);
	H_M = (float*)malloc(sizeInFloats);
	H_R = (float*)malloc(sizeInFloats);
	H_R1 = (float*)malloc(sizeInFloats);

	initialData(H_N, S * S);
	initialData(H_M, S * S);

	memset(H_R, 0, S);
	memset(H_R1, 0, S);

	float* D_N, * D_M, * D_R;
	cudaMalloc((void**)&D_N, sizeInFloats);
	cudaMalloc((void**)&D_M, sizeInFloats);
	cudaMalloc((void**)&D_R, sizeInFloats);


	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);

	//printf("The transfer from host to device took : %d micro seconds.\n\n", duration);


	cudaEvent_t start, stop,start1,stop1;
	float time;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	cudaMemcpy(H_N, D_N, sizeInFloats, cudaMemcpyDeviceToHost);
	cudaMemcpy(H_M, D_M, sizeInFloats, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);

	printf("The data transfer from device to host took : %f milli seconds for %d x %d elements\n\n", time,S,S);

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);

	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);

	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time, start1, stop1);

	printf("The data transfer from host to device took : %f milli seconds for %d x %d elements\n\n", time, S,S);

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
}
void computeMatrixGPU(int S, int s){ 
	float* D_N, * D_M, * D_R; 
	float* H_N, * H_M, *H_R; 

	size_t sizeInFloats = S * S * sizeof(float);

	H_N = (float*)malloc(sizeInFloats);
	H_M = (float*)malloc(sizeInFloats);
	H_R = (float*)malloc(sizeInFloats);

	cudaMalloc((void**)&D_N, sizeInFloats);
	cudaMalloc((void**)&D_M, sizeInFloats);
	cudaMalloc((void**)&D_R, sizeInFloats);

	initialData(H_N, S * S);
	initialData(H_M, S * S);


	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float time;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	dim3 block(s, s);
	dim3 thread((S + block.x - 1) / block.x, (S + block.y - 1) / block.y);


	MatrixMultElement << < thread, block >> > (D_N, D_M, D_R, S);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);



	printf("The GPU took %f microseconds to complete the computation with one thread per element.\n\n", time * 1000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(H_R, D_R, sizeInFloats, cudaMemcpyDeviceToHost);

	cudaFree(D_N);
	cudaFree(D_M);
	cudaFree(D_R);

	free(H_N);
	free(H_M);
	free(H_R);
	
	
	// reset device

	cudaDeviceReset();
}
void computeMatrix(int S, int s) {

	float* H_N, *H_M, *H_R, *H_R1;
	//Size of matrix dimension i.e. 1024


	// Multiply each dimension to get the matrix and then multiply by size of int to get the value in bytes

	size_t sizeInFloats = S * S * sizeof(float);
	//input host vector N

	H_N = (float*)malloc(sizeInFloats);
	H_M = (float*)malloc(sizeInFloats);
	H_R = (float*)malloc(sizeInFloats);
	H_R1 = (float*)malloc(sizeInFloats);



	initialData(H_N, S * S);
	initialData(H_M, S * S);

	memset(H_R, 0, S);
	memset(H_R1, 0, S);

	auto t1 = std::chrono::high_resolution_clock::now();
	sumMatrixOnHost(H_N, H_M, H_R, S);
	auto t2 = std::chrono::high_resolution_clock::now();
	float* D_N, * D_M, * D_R;
	float duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	printf("The CPU took %f microseconds to complete the computation.\n\n", duration);

	cudaMalloc((void**)&D_N, sizeInFloats);
	cudaMalloc((void**)&D_M, sizeInFloats);
	cudaMalloc((void**)&D_R, sizeInFloats);
	
	cudaMemcpy(D_N, H_N, sizeInFloats, cudaMemcpyHostToDevice);
	cudaMemcpy(D_M, H_M, sizeInFloats, cudaMemcpyHostToDevice);
	
	
	cudaEvent_t start,stop;
	float time;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	dim3 block(s, s);
	dim3 thread((S + block.x - 1) / block.x, (S + block.y - 1) / block.y);
	

	MatrixMultElement << < thread, block >> > (D_N, D_M, D_R, S);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	


	printf("The GPU took %f microseconds to complete the computation with one thread per element.\n\n", time*1000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(H_R1, D_R, sizeInFloats, cudaMemcpyDeviceToHost);
	checkResult(H_R, H_R1, S * S);


	cudaFree(D_N);
	cudaFree(D_M);
	cudaFree(D_R);
	
	free(H_N);
	free(H_M);
	free(H_R);
	free(H_R1);
	// reset device

	cudaDeviceReset();


}

int main()
	
{	//Part 1
	
	printf("Data transfer from host to device and device to host\n\n");
	TimeDataTransfer(100);
	TimeDataTransfer(200);
	TimeDataTransfer(500);
	TimeDataTransfer(1000); 
	TimeDataTransfer(1500);
	TimeDataTransfer(5000); 
	//Part 2

	
	int block = 1;
	printf(" Part 2 : multiplication with %d blocks \n\n",block);
	printf("100 x 100 matrix multiplication.\n");
	computeMatrix(100,block);
	printf("200 x 200 matrix multiplication.\n");
	computeMatrix(200, block);
	printf("500 x 500 matrix multiplication.\n");
	computeMatrix(500, block);
	printf("1000 x 1000 matrix multiplication.\n");
	computeMatrix(1000, block);
	printf("1500 x 1500 matrix multiplication.\n");
	computeMatrix(1500, block);
	printf("5000 x 5000 matrix multiplication.\n");
	computeMatrix(5000, block);

	
	int blockSize[] = {2,4,10,20,25};
	for (int i = 0; i < 5; i++) {
		int block = blockSize[i];
		printf("Part 3: multiplication with %d blocks \n\n", block);
		printf("100 x 100 matrix multiplication.\n");
		computeMatrixGPU(100, block);
		printf("200 x 200 matrix multiplication.\n");
		computeMatrixGPU(200, block);
		printf("500 x 500 matrix multiplication.\n");
		computeMatrixGPU(500, block);
		printf("1000 x 1000 matrix multiplication.\n");
		computeMatrixGPU(1000, block);
		printf("1500 x 1500 matrix multiplication.\n");
		computeMatrixGPU(1500, block);
		printf("5000 x 5000 matrix multiplication.\n");
		computeMatrixGPU(5000, block);
	} 
	

	
	return 0; 


}