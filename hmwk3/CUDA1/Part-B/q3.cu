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

    int i, curr_index;
    
    for (i=0; i < N; i++) {
        curr_index = threadStartIndex + i*stride_size;
        C[curr_index] = A[curr_index] + B[curr_index];
    }
    
}


float *dh_array1, *dh_array2, *dh_out;

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
    // Free device vectors
    if (dh_array1)
        cudaFree(dh_array1);
    if (dh_array2)
        cudaFree(dh_array2);
    if (dh_out)
        cudaFree(dh_out);

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

    cudaError_t error;
    error = cudaMallocManaged((void**)&dh_array1, count * sizeof(float));
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&dh_array2, count * sizeof(float));
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMallocManaged((void**)&dh_out, count * sizeof(float));
    if (error != cudaSuccess) Cleanup(false);

    auto t1 = high_resolution_clock::now();
    result.alloc_time = duration_cast<microseconds>(t1 - t0);

    for (size_t i = 0; i < count; i++) {
        dh_array1[i] = (float) i;
        dh_array2[i] = (float) count - i;
    }


    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(dh_array1, count*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(dh_array2, count*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(dh_out, count*sizeof(float), device, NULL);
    


    auto t2 = high_resolution_clock::now();
    result.populate_time = duration_cast<microseconds>(t2 - t1);

    dim3 dimGrid(blocks);                    
    dim3 dimBlock(threads);       

    size_t values_per_thread = (count) / (blocks * threads);

     // Invoke kernel
    SumArrays<<<dimGrid, dimBlock>>>(dh_array1, dh_array2, dh_out, values_per_thread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaDeviceSynchronize();

    auto t3 = high_resolution_clock::now();
    result.calc_time = duration_cast<microseconds>(t3-t2);
    
    // This is a bit pointless here
    if (!verify_sum_arrays(count, dh_array1, dh_array2, dh_out)) {
        cout << "Verification failed!" << endl;
        Cleanup(false);
    }

    result.last_sum = dh_out[count-1];

    if (dh_array1) cudaFree(dh_array1);
    if (dh_array2) cudaFree(dh_array2);
    if (dh_out) cudaFree(dh_out);

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

    run_cuda_tests(elems, "q3");

}