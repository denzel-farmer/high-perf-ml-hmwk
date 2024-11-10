#include <iostream>
#include <sstream>
#include <chrono>
#include "partb.h"

using namespace std;
using namespace std::chrono;


__global__ void SumArrays(const float* A, const float* B, float* C, int N)
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


float *h_array1, *d_array1, *h_array2, *d_array2, *h_out, *d_out;

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
    // Free device vectors
    if (d_array1)
        cudaFree(d_array1);
    if (d_array2)
        cudaFree(d_array2);
    if (d_out)
        cudaFree(d_out);

    // Free host memory
    if (h_array1)
        free(h_array1);
    if (h_array2)
        free(h_array2);
    if (h_out)
        free(h_out);
    error = cudaDeviceReset();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}


alloc_sum_result time_sum_arrays(size_t count, size_t blocks, size_t threads) {
    alloc_sum_result result;
    result.alloc_time = microseconds(-1);
    result.populate_time = microseconds(-1);
    result.calc_time = microseconds(-1);
    result.total_time = microseconds(-1);
    auto t0 = high_resolution_clock::now();
    
    h_array1 = (float *)malloc(count * sizeof(float));
    if (!h_array1) Cleanup(false);
    h_array2 = (float *)malloc(count * sizeof(float));
    if (!h_array2) Cleanup(false);
    h_out = (float *)malloc(count * sizeof(float));
    if (!h_out) Cleanup(false);

    cudaError_t error;
    error = cudaMalloc((void**)&d_array1, count * sizeof(float));
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_array2, count * sizeof(float));
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_out, count * sizeof(float));
    if (error != cudaSuccess) Cleanup(false);

    auto t1 = high_resolution_clock::now();
    result.alloc_time = duration_cast<microseconds>(t1 - t0);

    for (size_t i = 0; i < count; i++) {
        h_array1[i] = (float) i;
        h_array2[i] = (float) count - i;
    }

    // Copy to device 
    error = cudaMemcpy(d_array1, h_array1, count*sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_array2, h_array2, count*sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    auto t2 = high_resolution_clock::now();
    result.populate_time = duration_cast<microseconds>(t2 - t1);

    dim3 dimGrid(blocks);                    
    dim3 dimBlock(threads);       

    size_t values_per_thread = (count) / (blocks * threads);

     // Invoke kernel
    SumArrays<<<dimGrid, dimBlock>>>(d_array1, d_array2, d_out, values_per_thread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    auto t3 = high_resolution_clock::now();
    result.calc_time = duration_cast<microseconds>(t3-t2);
    
    error = cudaMemcpy(h_out, d_out, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    // This is a bit pointless here
    if (!verify_sum_arrays(count, h_array1, h_array2, h_out)) {
        cout << "Verification failed!" << endl;
        Cleanup(false);
    }

    result.last_sum = h_out[count-1];

    if (h_array1) free(h_array1);
    if (h_array2) free(h_array2);
    if (h_out) free(h_out);
    if (d_array1) cudaFree(d_array1);
    if (d_array2) cudaFree(d_array2);
    if (d_out) cudaFree(d_out);

    auto t4 = high_resolution_clock::now();
    result.total_time = duration_cast<microseconds>(t4-t0);
    return result;
}

int main(int const argc, char *argv[]) {

    if (argc != 2){
        cout << "Usage: " << argv[0] << " ELEMS\n";
        exit(1);
    }

    size_t elems;
    stringstream sstream(argv[1]);
    sstream >> elems;
    if (elems == 0) {
        cout << "Invalid element size\n";
        exit(1);
    }

    run_cuda_tests(elems, "q2");
}