#include <stdio.h>
#include <chrono>
#include <iostream>

#include "ImageUtils.h"

using namespace std;
using namespace std::chrono;


// TODO should put in image utils once I figure out linking 
__host__ __device__ ELEM_TYPE GetFilterSetElement(const FilterSet& filters, int i, int j, int k, int l) {
    return filters.elements[(i * filters.height * filters.width * filters.depth) + (j * filters.height * filters.width) \
        + (k * filters.width) + l];
}

__host__ __device__ ELEM_TYPE GetImageElement(const Image &image, int i, int j, int k) {
    return image.elements[(i * image.height * image.width) + (j * image.width) + k];
}

__host__ __device__ void SetImageElement(Image &image, int i, int j, int k, ELEM_TYPE value) {
    image.elements[(i * image.height * image.width) + (j * image.width) + k] = value;
}

#define BLOCK_SIZE 16
#define BLOCK_DEPTH 1

#define FILTER_SIZE 3
#define IMAGE_CHANNELS 3
#define IMAGE_SIZE 1024

// // TODO calculate dynamically based on input
// #define OUT_SIZE 1024
// #define OUT_DEPTH 64


// #define OUT_X 1023
// #define OUT_Y 1023
// #define OUT_Z 1


#define TILE_SIZE (FILTER_SIZE + BLOCK_SIZE - 1)
#define TILE_DEPTH (IMAGE_CHANNELS)

__global__ void Convolution(Image in_image, FilterSet filters, Image out_image){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int bx = blockIdx.x;
    // int by = blockIdx.y;

    int thread_base_x = blockIdx.x * blockDim.x;
    int thread_base_y = blockIdx.y * blockDim.y;

    int in_x = thread_base_x + tx;
    int in_y = thread_base_y + ty;

    int k = blockIdx.z * blockDim.z + threadIdx.z;


    __shared__ ELEM_TYPE in_tile[TILE_DEPTH][TILE_SIZE][TILE_SIZE];
    __shared__ ELEM_TYPE filter[3][3][3];

    // load in filter
    int filter_x = FILTER_SIZE - 1 - (tx % FILTER_SIZE);
    int filter_y = FILTER_SIZE - 1 - (ty % FILTER_SIZE);
#pragma unroll
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
        filter[c][filter_x][filter_y] = GetFilterSetElement(filters, k, c, filter_y, filter_x);
    }
    
#pragma unroll
    for (int c = 0; c < IMAGE_CHANNELS; ++c) {
#pragma unroll
        for (int idx_y = ty; idx_y < TILE_SIZE; idx_y += blockDim.y) {
#pragma unroll
            for (int idx_x = tx; idx_x < TILE_SIZE; idx_x += blockDim.x) {                  
                in_tile[c][idx_y][idx_x] = GetImageElement(in_image, c, thread_base_y + idx_y, thread_base_x + idx_x);
            }
        }
    }

    __syncthreads();

    // To produce O[k,x,y], so k=z
ELEM_TYPE output_value = 0;
#pragma unroll
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
#pragma unroll
        for (int j = 0; j < FILTER_SIZE; j++) {
#pragma unroll
            for (int i = 0; i < FILTER_SIZE; i++) {

                // F[k, c, F W − 1 − i, F H − 1 − j]
                // I_0[c, x + i, y + j]
                int filter_x = FILTER_SIZE - 1 - i;
                int filter_y = FILTER_SIZE - 1 - j;
                output_value += filter[c][filter_x][filter_y] * in_tile[c][ty+j][tx+i];
            }
        }
    }

    SetImageElement(out_image, k, in_y, in_x, output_value);

}

microseconds ConvolutionalFilter(const Image in_image, const FilterSet filters, Image out_image) {
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_DEPTH);
    dim3 dimGrid(out_image.height / BLOCK_SIZE, out_image.width / BLOCK_SIZE, out_image.depth / BLOCK_DEPTH);


    Image d_in_image = MakeDeviceImage(in_image, true);
    FilterSet d_filters = MakeDeviceFilterSet(filters, true);
    Image d_out_image = MakeDeviceImage(out_image, false);

    cudaThreadSynchronize();
    auto t0 = high_resolution_clock::now();

    Convolution<<<dimGrid, dimBlock>>>(d_in_image, d_filters, d_out_image);

    cudaThreadSynchronize();
    auto t1 = high_resolution_clock::now();

    cudaMemcpy(out_image.elements, d_out_image.elements, GetElementCount(d_out_image)*sizeof(ELEM_TYPE), cudaMemcpyDeviceToHost);


    // ELEM_TYPE value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    // printf("Value at (%d, %d, %d): %f\n", OUT_Z, OUT_X, OUT_Y, value);

    cudaFree(d_in_image.elements);
    cudaFree(d_filters.elements);
    cudaFree(d_out_image.elements);

    return duration_cast<microseconds>(t1-t0);
}


int main(int argc, char* argv[]) {
    int maxprint = 1024*3*1;
    bool verbose = false;
    if (argc != 1 && argc != 2) {
        printf("Usage: %s verbose\n", argv[0]);
    }

    if (argc == 2) {
        verbose = (atoi(argv[1]) != 0);
    }

    // Create input image
    int H = 1024;
    int W = 1024; 
    int C = 3;
    int padding = 1;

    Image in_image = GenerateInputImage(C, H, W, padding);

    // if (verbose) {
    //     PrintImage(in_image, maxprint);
    // }

    // Create filter set 
    int K = 64;
    int FW = 3;
    int FH = 3;

    FilterSet filters = GenerateFilterSet(K, C, FH, FW);
    printf("Element at (5, 1, 100, 100) in filters: %f\n", GetFilterSetElement(filters, 5, 1, 100, 100));
    // Create output image
    Image out_image = AllocateHostImage(K, H, W);
    // Perform convolution
    microseconds kernel_time = ConvolutionalFilter(in_image, filters, out_image);

    // Print output image if verbose
    if (verbose) {
        PrintImage(out_image, maxprint);
    }
    // Print out checksum
    cout << fixed << "Checksum: " << ImageChecksum(out_image) << endl;
    cout << "Kernel execution time: " << duration_cast<milliseconds>(kernel_time).count() << "ms" << endl;

    // Free memory
    free(in_image.elements);
    free(filters.elements);
    free(out_image.elements);

    return 0;
}