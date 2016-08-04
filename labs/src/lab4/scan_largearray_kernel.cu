#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>
#include <cutil_inline.h>
#include <iostream>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif


float** g_scanBlockSums;
unsigned int g_numEltAlloc = 0;
unsigned int g_numLevelAlloc = 0;


//function declare
template <bool storeSum, bool isNP2>
__global__ void prescan_kernel(float *g_odata, const float *g_idata, float *g_blockSums,
                        int n, int blockIndex, int baseIndex);

__global__ void g_add(float *g_data, float *sames, int n, 
			int blockOffset, int baseIndex);


// Lab4: Host Helper Functions (allocate your own data structure...)
inline bool isPowerOfTwo(int n)
{
    return ((n&(n-1))==0);
}

inline int floorPow2(int n)
{
#ifdef WIN32
    return 1 << (int)logb((float)n);
#else
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}


void preOperation(unsigned int maxNumElements)
{
    assert(g_numEltAlloc == 0); // shouldn't be called

    g_numEltAlloc = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));
    g_numLevelAlloc = level;

    numElts = maxNumElements;
    level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            cutilSafeCall(cudaMalloc((void**) &g_scanBlockSums[level++],
                                      numBlocks * sizeof(float)));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    cutilCheckMsg("preOperation");
}

void prevOperation()
{
    for (unsigned int i = 0; i < g_numLevelAlloc; i++)
    {
        cudaFree(g_scanBlockSums[i]);
    }

    cutilCheckMsg("prevOperation");

    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltAlloc = 0;
    g_numLevelAlloc = 0;
}


void recursiveCall(float *outArray,
                           const float *inArray,
                           int numElements,
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks =
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock =
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);

        unsigned int padding = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock =
            sizeof(float) * (2 * numThreadsLastBlock + padding);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int padding = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + padding);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltAlloc >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3  threads(numThreads, 1, 1);

    // make sure there are no CUDA errors before we start
    cutilCheckMsg("recursiveCall before kernels");

    // execute the scan
    if (numBlocks > 1)
    {
        prescan_kernel<true, false><<< grid, threads, sharedMemSize >>>(outArray,
                                                                 inArray,
                                                                 g_scanBlockSums[level],
                                                                 numThreads * 2, 0, 0);
        cutilCheckMsg("prescanWithBlockSums");
        if (np2LastBlock)
        {
            prescan_kernel<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock,
                 numBlocks - 1, numElements - numEltsLastBlock);
            cutilCheckMsg("prescanNP2WithBlockSums");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we
        // need to take all of the last values of the sub-blocks and scan those.
        // This will give us a new value that must be sdded to each block to
        // get the final results.
        // recursive (CPU) call
        recursiveCall(g_scanBlockSums[level],
                              g_scanBlockSums[level],
                              numBlocks,
                              level+1);
	printf("1\n");
        g_add<<< grid, threads >>>(outArray,
                                        g_scanBlockSums[level],
                                        numElements - numEltsLastBlock,
                                        0, 0);
        cutilCheckMsg("add");
        if (np2LastBlock)
        {
            g_add<<< 1, numThreadsLastBlock >>>(outArray,
                                                     g_scanBlockSums[level],
                                                     numEltsLastBlock,
                                                     numBlocks - 1,
                                                     numElements - numEltsLastBlock);
            cutilCheckMsg("add");
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        prescan_kernel<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
        cutilCheckMsg("prescan");
    }
    else
    {
         prescan_kernel<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numElements, 0, 0);
         cutilCheckMsg("prescanNP2");
    }
}


// Lab4: Device Functions
template <bool isNP2>
__device__ void loadSharedFromMem(float *s_data, const float *g_idata,int n, int baseIndex,
                                       int& ai, int& bi, int& mem_ai, int& mem_bi, 
                                       int& bankOffsetA, int& bankOffsetB)
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;

    // compute spacing to avoid bank conflicts
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    // pad values beyond n with zeros
    s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
    
    if (isNP2) // compile-time decision
    {
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    }
    else
    {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}


template <bool isNP2>
__device__ void storeSharedToMem(float* g_odata, 
                                      const float* s_data,
                                      int n, 
                                      int ai, int bi, 
                                      int mem_ai, int mem_bi,
                                      int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    // write results to global memory
    g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
    if (isNP2) // compile-time decision
    {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
    else
    {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}



template <bool storeSum>
__device__ void clearLastElement(float* s_data, 
                                 float *g_blockSums, 
                                 int blockIndex)
{
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        
        if (storeSum) // compile-time decision
        {
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}



__device__ unsigned int sum(float *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__ void scanRootToLeaves(float *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum>
__device__ void prescanBlock(float *data, int blockIndex, float *blockSums)
{
    int stride = sum(data);               // build the sum in place up the tree
    clearLastElement<storeSum>(data, blockSums, 
                               (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves(data, stride);            // traverse down tree to build the scan 
}



// Lab4: Kernel Functions

__global__ void g_add(float *g_data, float *sames, int n, 
                           int blockOffset, int baseIndex)
{
    __shared__ float shareI;
    if (threadIdx.x == 0)
        shareI = sames[blockIdx.x + blockOffset];
    
    unsigned int addr = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    g_data[addr] += shareI;
    g_data[addr + blockDim.x] += (threadIdx.x + blockDim.x < n) * shareI;
}

template <bool storeSum, bool isNP2>
__global__ void prescan_kernel(float *g_odata, const float *g_idata, float *g_blockSums, 
                        int n, int blockIndex, int baseIndex)
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ float s_data[];

    // load data into shared memory
    loadSharedFromMem<isNP2>(s_data, g_idata, n, 
                                  (baseIndex == 0) ? 
                                  __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi, 
                                  bankOffsetA, bankOffsetB); 
    // scan the data in each block
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    // write results to device memory
    storeSharedToMem<isNP2>(g_odata, s_data, n, 
                                 ai, bi, mem_ai, mem_bi, 
                                 bankOffsetA, bankOffsetB);  
}





// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
	recursiveCall(outArray, inArray, numElements, 0);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_

