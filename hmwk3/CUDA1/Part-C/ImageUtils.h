#pragma once 

#define ELEM_TYPE double    

struct Image {
    int width;
    int height;
    int depth;
    ELEM_TYPE *elements;
};

struct FilterSet {
    int width;
    int height;
    int depth;
    int count;
    ELEM_TYPE *elements;
};


Image MakeDeviceImage(Image h_image, bool copy);

FilterSet MakeDeviceFilterSet(FilterSet h_filters, bool copy);
__host__ __device__ ELEM_TYPE GetFilterSetElement(const FilterSet& filters, int k, int c, int x, int y);
void SetFilterSetElement(FilterSet& filters, int k, int c, int x, int y, ELEM_TYPE value);

__host__ __device__ ELEM_TYPE GetImageElement(const Image &image, int c, int x, int y);
__host__ __device__ void SetImageElement(Image &image, int c, int x, int y, ELEM_TYPE value);

size_t GetElementCount(FilterSet filters);
size_t GetElementCount(Image image);

Image AllocateHostImage(int C, int H, int W);

Image GenerateInputImage(int C, int H, int W, int padding);

FilterSet GenerateFilterSet(int K, int C, int FH, int FW);

void PrintImage(const Image& image, int max_elements);

ELEM_TYPE ImageChecksum(const Image& image);