#include <iostream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

struct alloc_sum_result {
    microseconds alloc_time;
    microseconds populate_time;
    microseconds calc_time;
    microseconds total_time;
    // Keep the last sum value and print it out to avoid optimizer skipping 
    float last_sum;
};

#define EPS ((float)1e-4)

bool verify_sum_arrays(size_t count, float *input1, float *input2, float *output) {
    for (size_t i = 0; i < count; i++) {
        float exp_sum = input1[i] + input2[i];
        if (abs(output[i] - exp_sum) > EPS) {
            cout << "Mismatch at index " << i << ": expected " << exp_sum << ", got " << output[i] << endl;
            return false;
        }
    }

    return true;

}

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

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
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

void print_results(alloc_sum_result result) {
    cout << "Total Time: " << duration_cast<milliseconds>(result.total_time).count() << "ms" << endl;
    cout << "Allocation Time: " << duration_cast<milliseconds>(result.alloc_time).count() << "ms" << endl;
    cout << "Populate Time: " << duration_cast<milliseconds>(result.populate_time).count() << "ms" << endl;
    cout << "Calculation Time: " << duration_cast<microseconds>(result.calc_time).count() << "us" << endl;
    cout << "Total Time: " << duration_cast<milliseconds>(result.total_time).count() << "ms" << endl;
    cout << "Last Sum Value: " << result.last_sum << endl;

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

    cout << "Using array of size: " << elems << "M" << endl;
    size_t blocks;
    size_t threads;

    elems *= 1e6;
    // Must be multiple of largest threads (256)
    elems = ((elems + 255) / 256) * 256;

    blocks = 1;
    threads = 1;
    cout << "\nRunning test with " << blocks << " blocks/grid and " << threads << " threads per block\n";
    auto result = time_sum_arrays(elems, blocks, threads);
    print_results(result);
   

    blocks = 1;
    threads = 256;
    cout << "\nRunning test with " << blocks << " blocks/grid and " << threads << " threads per block\n";
    result = time_sum_arrays(elems, blocks, threads);
    print_results(result);

    threads = 256;
    blocks = elems / threads;
    cout << "\nRunning test with " << blocks << " blocks/grid and " << threads << " threads per block\n";
    result = time_sum_arrays(elems, blocks, threads);
    print_results(result);



}