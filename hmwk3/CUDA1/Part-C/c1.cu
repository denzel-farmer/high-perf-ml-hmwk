#include <stdio.h>
#include <chrono>
#include <iostream>

#include "ImageUtils.h"

using namespace std;
using namespace std::chrono;


// Because of linking issues, I have trouble defining these in ImageUtils.h
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


__global__ void Convolution(Image in_image, FilterSet filters, Image out_image){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    ELEM_TYPE output_value = 0;
    for (int c = 0; c < in_image.depth; c++) {
        for (int j = 0; j < filters.height; j++) {
            for (int i = 0; i < filters.width; i++) {

                output_value += GetFilterSetElement(filters, k, c, filters.width - 1 - i, filters.height - 1 - j) \
                        * GetImageElement(in_image, c, x + i, y + j);
            }
        }
    }

    SetImageElement(out_image, k, x, y, output_value);

}

constexpr int BLOCK_SIZE = 32;
constexpr int BLOCK_DEPTH = 1;

// Run convolution and return time 
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

    cudaFree(d_in_image.elements);
    cudaFree(d_filters.elements);
    cudaFree(d_out_image.elements);

    return duration_cast<microseconds>(t1-t0);
}

// Run convolution, if verbose print some number of input and output image elements
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

    if (verbose) {
        cout << "Input image:" << endl;
        PrintImage(in_image, maxprint);
    }

    // Create filter set 
    int K = 64;
    int FW = 3;
    int FH = 3;

    FilterSet filters = GenerateFilterSet(K, C, FH, FW);
    
    // Create output image
    Image out_image = AllocateHostImage(K, H, W);
    // Perform convolution
    microseconds kernel_time = ConvolutionalFilter(in_image, filters, out_image);

    // Print output image if verbose
    if (verbose) {
        cout << "Output image:" << endl;
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