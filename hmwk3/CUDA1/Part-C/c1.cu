#include <stdio.h>
#include <chrono>
#include <iostream>

#include "ImageUtils.h"

using namespace std;
using namespace std::chrono;


#define BLOCK_SIZE 32
#define BLOCK_DEPTH 1

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
    // Input image I is of size C, W, H
    // Filter set F is of size K, C, FH, FW

    // Block size = 32x32
    // Shared tile size = 34x34

    // in_tile[tx][ty] = I[bx*bdx + tx][by*bdy + ty]

    // If globally out of bounds, then pad input tile with 0.0
        // in_tile[ty * tile size in shared memory + tx] = 0.0
    // If not out of bounds
        // in_tile[ty * tile size in shared memory + tx] = I[global x * W + global y]

    // To produce O[k,x,y], so k=z
    ELEM_TYPE output_value = 0;
    for (int c = 0; c < in_image.depth; c++) {
        for (int j = 0; j < filters.height; j++) {
            for (int i = 0; i < filters.width; i++) {

                // F[k, c, F W − 1 − i, F H − 1 − j]
                // I_0[c, x + i, y + j]
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


// TODO calculate dynamically based on input
#define OUT_SIZE 1024
#define OUT_DEPTH 64


#define OUT_X 1023
#define OUT_Y 1023
#define OUT_Z 1



microseconds ConvolutionalFilter(const Image in_image, const FilterSet filters, Image out_image) {
    
    size_t block_dim_x = BLOCK_SIZE;
    size_t block_dim_y = BLOCK_SIZE;
    size_t block_dim_z = BLOCK_DEPTH;

    dim3 dimBlock(block_dim_x, block_dim_y, block_dim_z);

    // TODO calculate OUT_SIZE and OUT_DEPTH from image/filter dimensions
    size_t num_blocks_x = OUT_SIZE / block_dim_x;
    size_t num_blocks_y = OUT_SIZE / block_dim_y;
    size_t num_blocks_z = OUT_DEPTH / block_dim_z;

    dim3 dimGrid(num_blocks_x, num_blocks_y, num_blocks_z);

    Image d_in_image = MakeDeviceImage(in_image, true);
    FilterSet d_filters = MakeDeviceFilterSet(filters, true);
    Image d_out_image = MakeDeviceImage(out_image, false);

    cudaThreadSynchronize();
    auto t0 = high_resolution_clock::now();

    Convolution<<<dimGrid, dimBlock>>>(d_in_image, d_filters, d_out_image);

    cudaThreadSynchronize();
    auto t1 = high_resolution_clock::now();

    cudaMemcpy(out_image.elements, d_out_image.elements, GetElementCount(d_out_image)*sizeof(ELEM_TYPE), cudaMemcpyDeviceToHost);


    double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    printf("Value at (%d, %d, %d): %f\n", OUT_Z, OUT_X, OUT_Y, value);

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

    double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    cout << "Value at (" << OUT_Z << ", " << OUT_X << ", " << OUT_Y << "): " << value << endl;
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