
#include "ImageUtils.h"
#include "stdio.h"

using namespace std;

Image MakeDeviceImage(Image h_image, bool copy) {
    Image d_image;
    d_image.width = h_image.width;
    d_image.height = h_image.height;
    d_image.depth = h_image.depth;
    size_t size = h_image.width * h_image.height * h_image.depth * sizeof(ELEM_TYPE);
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
    size_t size = h_filters.width * h_filters.height * h_filters.depth * h_filters.count * sizeof(ELEM_TYPE);
    cudaMalloc(&d_filters.elements, size);
    if (copy) {
        cudaMemcpy(d_filters.elements, h_filters.elements, size, cudaMemcpyHostToDevice);
    }
    return d_filters;
}


void SetFilterSetElement(FilterSet& filters, int i, int j, int k, int l, ELEM_TYPE value) {
    filters.elements[(i * filters.height * filters.width * filters.depth) + (j * filters.height * filters.width) \
        + (k * filters.width) + l] = value;
}


size_t GetElementCount(FilterSet filters) {
    return filters.count*filters.depth*filters.height*filters.width;
}

size_t GetElementCount(Image image) {
    return image.depth*image.width*image.height;
}




Image AllocateHostImage(int C, int H, int W) {
    Image image;
    image.depth = C;
    image.height = H;
    image.width = W;
    image.elements = (ELEM_TYPE *)malloc(C*H*W*sizeof(ELEM_TYPE));

    return image;
}

Image GenerateInputImage(int C, int H, int W, int padding) {
    int padded_width = W + 2*padding;
    int padded_height = H + 2*padding;
   
    Image image = AllocateHostImage(C, padded_height, padded_width);
    for (int c = 0; c < C; c++) {
        for (int x = 0; x < padded_width; x++) {
            for (int y = 0; y < padded_height; y++) {
                if (x < padding || y < padding || y >= (padded_width - padding) || x >= (padded_width - padding)) {
                    SetImageElement(image, c, x, y, 0);
                } else {
                    SetImageElement(image, c, x, y, c * ((x-1) + (y-1)));
                }
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
    filters.elements = (ELEM_TYPE *)malloc(K*C*FH*FW*sizeof(ELEM_TYPE));

    //F [k, c, i, j] = (c + k) · (i + j)
    // TODO I feel weird about i < FH and j < FW, should switch?
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++) {
                for (int j = 0; j < FW; j++) {
                    ELEM_TYPE value = (c + k)*(i+j);
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

double ImageChecksum(const Image& image) {
    double sum = 0.0;
    for (int elem = 0; elem < GetElementCount(image); elem++) {
        sum += image.elements[elem];
    }

    return sum;
}
