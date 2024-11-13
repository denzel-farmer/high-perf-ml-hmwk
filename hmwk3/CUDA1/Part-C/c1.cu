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

size_t GetElementCount(FilterSet filters) {
    return filters.count*filters.depth*filters.height*filters.width;
}

size_t GetElementCount(Image image) {
    return image.depth*image.width*image.height;
}



constexpr int BLOCK_SIZE = 32;
constexpr int BLOCK_DEPTH = 1;

constexpr int OUT_SIZE = 1024;
constexpr int OUT_DEPTH = 64;

// constexpr int OUT_X = 1023;
// constexpr int OUT_Y = 1023;
// constexpr int OUT_Z = 1;



microseconds ConvolutionalFilter(const Image in_image, const FilterSet filters, Image out_image) {
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_DEPTH);
    dim3 dimGrid(OUT_SIZE / BLOCK_SIZE, OUT_SIZE / BLOCK_SIZE, OUT_DEPTH / BLOCK_DEPTH);

    Image d_in_image = MakeDeviceImage(in_image, true);
    FilterSet d_filters = MakeDeviceFilterSet(filters, true);
    Image d_out_image = MakeDeviceImage(out_image, false);

    cudaThreadSynchronize();
    auto t0 = high_resolution_clock::now();

    Convolution<<<dimGrid, dimBlock>>>(d_in_image, d_filters, d_out_image);

    cudaThreadSynchronize();
    auto t1 = high_resolution_clock::now();

    cudaMemcpy(out_image.elements, d_out_image.elements, GetElementCount(d_out_image)*sizeof(ELEM_TYPE), cudaMemcpyDeviceToHost);


    // double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
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
    C = 3;  
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

    // double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    // cout << "Value at (" << OUT_Z << ", " << OUT_X << ", " << OUT_Y << "): " << value << endl;
    // while (1) {
    //     int c, x, y;
    //     printf("Enter c x y (or -1 to exit): ");
    //     scanf("%d %d %d", &c, &x, &y);
    //     if (c == -1 || x == -1 || y == -1) {
    //         break;
    //     }
    //     if (c >= 0 && c < out_image.depth && x >= 0 && x < out_image.width && y >= 0 && y < out_image.height) {
    //         double value = GetImageElement(out_image, c, x, y);
    //         printf("Value at (%d, %d, %d): %f\n", c, x, y, value);
    //     } else {
    //         printf("Invalid coordinates.\n");
    //     }
    // }

    // Free memory
    free(in_image.elements);
    free(filters.elements);
    free(out_image.elements);

    return 0;
}