///
/// vecAddKernel01.cu
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
/// 

// This kernel is coalesced by having each thread calculate 'strided' elements
// of the array. So, on the first iteration thread 0 calculates element 0, thread
// 1 calculates element 1, etc. Then on the second iteration, thread 0 calculates
// num_threads + 0, thread 1 calculates num_threads + 1, etc. 
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // number of threads accessing together 
    int stride_size = (blockDim.x)*(gridDim.x);
    int blockStartIndex  = blockIdx.x * blockDim.x;
    int threadStartIndex = blockStartIndex + threadIdx.x;

    int thread_id = (blockIdx.x)*(blockDim.x) + (threadIdx.x);

    int i, curr_index;
    
    for (i=0; i < N; i++) {
        curr_index = threadStartIndex + i*stride_size;
        C[curr_index] = A[curr_index] + B[curr_index];
    }
    
}
