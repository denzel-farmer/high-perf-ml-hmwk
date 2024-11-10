#include <stdio.h>


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
__device__ double GetPaddedImageElement(const Image& image, int c, int x, int y) {

    if (x > image.width || y > image.height) {
        return 0.0;
    }

    return image.elements[(c * image.height * image.width) + (y * image.width) + x];
}

__host__ __device__ void SetImageElement(Image &image, int c, int x, int y, double value) {
    image.elements[(c * image.height * image.width) + (y * image.width) + x] = value;
}

__global__ void Convolution(Image in_image, FilterSet filters, Image out_image){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    // Input image I is of size C, W, H
    // Filter set F is of size K, C, FH, FW

    // To produce O[k,x,y], so k=z
    double output_value = 0;
    for (int c = 0; c < in_image.depth; c++) {
        for (int j = 0; j < filters.height; j++) {
            for (int i = 0; i < filters.width; i++) {

                // F[k, c, F W − 1 − i, F H − 1 − j]
                // I_0[c, x + i, y + j]
                output_value += GetFilterSetElement(filters, k, c, filters.width - 1 - i, filters.height - 1 - j) \
                        * GetPaddedImageElement(in_image, c, x + i, y + j);
            }
        }
    }

    if (x == OUT_X && y == OUT_Y && k == OUT_Z) {
        printf("Value Before: %f\n", output_value);
    }


    SetImageElement(out_image, k, x, y, output_value);

    if (x == OUT_X && y == OUT_Y && k == OUT_Z) {
        printf("Value After: %f\n", GetImageElement(out_image, k, x, y));
    }


}

size_t GetElementCount(FilterSet filters) {
    return filters.count*filters.depth*filters.height*filters.width;
}

size_t GetElementCount(Image image) {
    return image.depth*image.width*image.height;
}

void ConvolutionalFilter(const Image in_image, const FilterSet filters, Image out_image) {
    
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


    Convolution<<<dimGrid, dimBlock>>>(d_in_image, d_filters, d_out_image);
    cudaThreadSynchronize();

    cudaMemcpy(out_image.elements, d_out_image.elements, GetElementCount(d_out_image)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    printf("Value at (%d, %d, %d): %f\n", OUT_Z, OUT_X, OUT_Y, value);

    cudaFree(d_in_image.elements);
    cudaFree(d_filters.elements);
    cudaFree(d_out_image.elements);
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
    ConvolutionalFilter(in_image, filters, out_image);

    // Print output image if verbose
    if (verbose) {
        PrintImage(out_image, maxprint);
    }

    // Print out checksum
    printf("Checksum: %f\n", checksum(out_image));

    double value = GetImageElement(out_image, OUT_Z, OUT_X, OUT_Y);
    printf("Value at (%d, %d, %d): %f\n", OUT_Z, OUT_X, OUT_Y, value);
    while (1) {
        int c, x, y;
        printf("Enter c x y (or -1 to exit): ");
        scanf("%d %d %d", &c, &x, &y);
        if (c == -1 || x == -1 || y == -1) {
            break;
        }
        if (c >= 0 && c < out_image.depth && x >= 0 && x < out_image.width && y >= 0 && y < out_image.height) {
            double value = GetImageElement(out_image, c, x, y);
            printf("Value at (%d, %d, %d): %f\n", c, x, y, value);
        } else {
            printf("Invalid coordinates.\n");
        }
    }

    // Free memory
    free(in_image.elements);
    free(filters.elements);
    free(out_image.elements);

    return 0;
}