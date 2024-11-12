#include <stdio.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

struct Image {
    int width;
    int height;
    int depth;
    double *elements;
};

struct FilterSet {
    int width;
    int height;
    int depth;
    int count;
    double *elements;
};

#define BLOCK_SIZE 32
#define BLOCK_DEPTH 1

#define FILTER_SIZE 3

#define IMAGE_CHANNELS 3
#define IMAGE_SIZE 1024

// TODO calculate dynamically based on input
#define OUT_SIZE 1024
#define OUT_DEPTH 64


#define OUT_X 1023
#define OUT_Y 1023
#define OUT_Z 1

Image MakeDeviceImage(Image h_image, bool copy) {
    Image d_image;
    d_image.width = h_image.width;
    d_image.height = h_image.height;
    d_image.depth = h_image.depth;
    size_t size = h_image.width * h_image.height * h_image.depth * sizeof(double);
    cudaMalloc(&d_image.elements, size);
    if (copy) {
        cudaMemcpy(d_image.elements, h_image.elements, size, cudaMemcpyHostToDevice);
    }
    return d_image;
}

FilterSet MakeDeviceFilterSet(FilterSet h_filters, bool copy) {
    FilterSet d_filters;
    d_filters.width = h_filters.width;
    d_filters.height = h_filters.height;
    d_filters.depth = h_filters.depth;
    d_filters.count = h_filters.count;
    size_t size = h_filters.width * h_filters.height * h_filters.depth * h_filters.count * sizeof(double);
    cudaMalloc(&d_filters.elements, size);
    if (copy) {
        cudaMemcpy(d_filters.elements, h_filters.elements, size, cudaMemcpyHostToDevice);
    }
    return d_filters;
}

__host__ __device__ double GetFilterSetElement(const FilterSet& filters, int k, int c, int x, int y) {
    return filters.elements[(k * filters.height * filters.width * filters.depth) + (c * filters.height * filters.width) \
        + (y * filters.width) + x];
}

void SetFilterSetElement(FilterSet& filters, int k, int c, int x, int y, double value) {
    filters.elements[(k * filters.height * filters.width * filters.depth) + (c * filters.height * filters.width) \
        + (y * filters.width) + x] = value;
}


__host__ __device__ double GetImageElement(const Image &image, int c, int x, int y) {
    return image.elements[(c * image.height * image.width) + (y * image.width) + x];
}

__host__ __device__ void SetImageElement(Image &image, int c, int x, int y, double value) {
    image.elements[(c * image.height * image.width) + (y * image.width) + x] = value;
}


#define TILE_SIZE (FILTER_SIZE + BLOCK_SIZE)
#define TILE_DEPTH (IMAGE_CHANNELS)

__global__ void Convolution(Image in_image, FilterSet filters, Image out_image){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
   

    // Thread size is 

    // int global_x = blockDim.x*bx + tx;
    // Check if within input image, otherwise do padding 
    // int global_x = bx + tx - pad;
    // int global_y = by + ty - pad; 

    // If global_x, global_y within bounds, can index correctly
    // if (globalx > 0 && globalx < H && globaly > 0 && globaly < W) { // means that it is in bounds
        // Then, tile[ty * tile size in shared memory + tx] = I[global x * W + global y]
    // else out of bounds, 

    int k = blockIdx.z * blockDim.z + threadIdx.z;


    __shared__ double in_tile[TILE_SIZE][TILE_SIZE][TILE_DEPTH];
    __shared__ double filter[3][3][3];
    // Compute global indices
    int in_x = bx * blockDim.x + tx;
    int in_y = by * blockDim.y + ty;

    // load in filter
#pragma unroll
    for (int c = 0; c < IMAGE_CHANNELS; c++) {
#pragma unroll
       for (int j = 0; j < FILTER_SIZE; j++) {
#pragma unroll
            for (int i = 0; i < FILTER_SIZE; i++) {
                int filter_x = FILTER_SIZE - 1 - i;
                int filter_y = FILTER_SIZE - 1 - j;
                filter[c][filter_x][filter_y] = GetFilterSetElement(filters, k, c, filter_x, filter_y);
            }
        }
    }
    // Load input image data into shared memory
    for (int c = 0; c < IMAGE_CHANNELS; ++c) {
        in_tile[ty][tx][c] = GetImageElement(in_image, c, in_x, in_y);
    }


    // Second round: use first 6 groups of 32 threads to load in remaining 192 elements
    // Which pad the block by 3
    // Load the right fringe into shared memory
    if (tx < FILTER_SIZE) {
        int x = bx * BLOCK_SIZE + BLOCK_SIZE + tx;
        int y = by * BLOCK_SIZE + ty;
#pragma unroll
        for (int c = 0; c < IMAGE_CHANNELS; c++) {
            in_tile[ty][BLOCK_SIZE + tx][c] = GetImageElement(in_image, c, x, y);
        }
    }

    // Load the bottom fringe into shared memory
    if (ty < FILTER_SIZE) {
        int x = bx * BLOCK_SIZE + tx;
        int y = by * BLOCK_SIZE + BLOCK_SIZE + ty;
#pragma unroll
        for (int c = 0; c < IMAGE_CHANNELS; c++) {
            in_tile[BLOCK_SIZE + ty][tx][c] = GetImageElement(in_image, c, x, y);
        }
    }

    // Load the bottom-right corner into shared memory
    if (tx < FILTER_SIZE && ty < FILTER_SIZE) {
        int x = bx * BLOCK_SIZE + BLOCK_SIZE + tx;
        int y = by * BLOCK_SIZE + BLOCK_SIZE + ty;
#pragma unroll
        for (int c = 0; c < IMAGE_CHANNELS; c++) {
            in_tile[BLOCK_SIZE + ty][BLOCK_SIZE + tx][c] = GetImageElement(in_image, c, x, y);
        }
    }

   

    __syncthreads();

    // To produce O[k,x,y], so k=z
    double output_value = 0;
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
                output_value += filter[c][filter_x][filter_y] * in_tile[ty+j][tx+i][c];
            }
        }
    }

    SetImageElement(out_image, k, in_x, in_y, output_value);

}

size_t GetElementCount(FilterSet filters) {
    return filters.count*filters.depth*filters.height*filters.width;
}

size_t GetElementCount(Image image) {
    return image.depth*image.width*image.height;
}

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

    cudaMemcpy(out_image.elements, d_out_image.elements, GetElementCount(d_out_image)*sizeof(double), cudaMemcpyDeviceToHost);


    double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    printf("Value at (%d, %d, %d): %f\n", OUT_Z, OUT_X, OUT_Y, value);

    cudaFree(d_in_image.elements);
    cudaFree(d_filters.elements);
    cudaFree(d_out_image.elements);

    return duration_cast<microseconds>(t1-t0);
}



Image AllocateHostImage(int C, int H, int W) {
    Image image;
    image.depth = C;
    image.height = H;
    image.width = W;
    image.elements = (double *)malloc(C*H*W*sizeof(double));

    return image;
}
#define PAD_MULT 1

Image GenerateInputImage(int C, int H, int W, int padding) {
    int unpadded_H = H;
    int unpadded_W = W;

    H = H + padding*PAD_MULT;
    W = W + padding*PAD_MULT;

    Image image = AllocateHostImage(C, H, W);

    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                double value = c * (x + y);

                if ((padding != 0) && ((y >= unpadded_H) || (x >= unpadded_W)))
                    value = 0.0;

                SetImageElement(image, c, x, y, value);
            }
        }
    }

    return image;
}

FilterSet GenerateFilterSet(int K, int C, int FH, int FW) {
    FilterSet filters; 

    filters.count = K;
    filters.depth = C;
    filters.width = FW;
    filters.height = FH;
    filters.elements = (double *)malloc(K*C*FH*FW*sizeof(double));

    //F [k, c, i, j] = (c + k) · (i + j)
    // TODO I feel weird about i < FH and j < FW, should switch?
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    double value = (c + k)*(i+j);
                    SetFilterSetElement(filters, k, c, i, j, value);
                }
            }
        }
    }


    return filters;
}

void PrintImage(const Image& image, int max_elements) {
    int count = 0;
    for (int y = 0; y < image.height && count < max_elements; y++) {
        for (int x = 0; x < image.width && count < max_elements; x++) {
            printf("[");
            for (int c = 0; c < image.depth; c++) {
                printf("%.0f", image.elements[(c * image.height * image.width) + (y * image.width) + x]);
                if (c < image.depth - 1) {
                    printf(", ");
                }
            }
            printf("]");
            for (int pad = 0; pad < 18 - (image.depth * 3); pad++) {
                printf(" ");
            }
            printf(" ");
            count++;
        }
        printf("\n");
    }
}

double checksum(const Image& image) {
    double sum = 0.0;
    for (int elem = 0; elem < GetElementCount(image); elem++) {
        sum += image.elements[elem];
    }

    return sum;
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
    int padding = 2;

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
    cout << fixed << "Checksum: " << checksum(out_image) << endl;
    cout << "Kernel execution time: " << duration_cast<milliseconds>(kernel_time).count() << "ms" << endl;

    double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    cout << "Value at (" << OUT_Z << ", " << OUT_X << ", " << OUT_Y << "): " << value << endl;


    // Free memory
    free(in_image.elements);
    free(filters.elements);
    free(out_image.elements);

    return 0;
}