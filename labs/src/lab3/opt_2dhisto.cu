#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#define BIN_COUNT 1024



void* AllocateDevice(size_t size)
{
	void *addr;
	cudaMalloc(&addr, size);
	return addr;
}

void MemCpyToDevice(void* dest, void* src, size_t size)
{
	cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* dest, void* src, size_t size)
{
	cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}


void FreeDevice(void* addr)
{
	cudaFree(addr);
}

__global__ void GenerateHist(uint32_t* input, size_t height, size_t width, uint32_t* global_bins)
{

	int globalTid  = blockIdx.x * blockDim.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;
	
	__shared__ int s_Hist[BIN_COUNT];	

	//clear partial histogram buffer
	#pragma unroll
	for (int pos = threadIdx.x; pos < BIN_COUNT; pos += numThreads) {
		s_Hist[pos] = 0;
	}
	__syncthreads ();

	//generate partial histogram
	#pragma unroll
	for (int pos = globalTid; pos < height * width; pos += numThreads) {
		if (s_Hist[input[pos]] < 255) {
			atomicAdd (s_Hist + input[pos], 1);
		}
	}
	__syncthreads();

	//update global histogram
	#pragma unroll
	for(int pos = threadIdx.x; pos < BIN_COUNT; pos += numThreads) {
		if(global_bins[threadIdx.x] < 255) {
			atomicAdd(global_bins + pos, s_Hist[pos]);
		}

	}
}


__global__ void Trans32to8(uint32_t* global_bins, uint8_t* device_bins)
{
	int globalTid  = blockIdx.x * blockDim.x + threadIdx.x;
	if(global_bins[globalTid] < 255) {
		device_bins[globalTid] = (uint8_t)global_bins[globalTid];
	}
	else {
		device_bins[globalTid] = (uint8_t)255;
	}	
}



void opt_2dhisto(uint32_t* device_input, int height, int width,  uint32_t* global_bins, uint8_t* device_bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
	cudaMemset(global_bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
    	GenerateHist<<<16, 1024>>>(device_input, height, width, global_bins);
	Trans32to8<<<1, 1024>>>(global_bins, device_bins);
	cudaThreadSynchronize();
}




/* Include below the implementation of any other functions you need */

