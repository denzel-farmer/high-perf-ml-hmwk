#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ImageUtils.h"

#define N 1        // Batch size
#define C 3        // Number of input channels
#define H 1024     // Input height
#define W 1024     // Input width
#define K 64       // Number of filters/output channels
#define FH 3       // Filter height
#define FW 3       // Filter width
#define PADDING 1


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

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



int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    
    // Initialize input tensor
    Image h_input = GenerateInputImage(C, H, W, 0);
    Image d_input = MakeDeviceImage(h_input, true);
    
    Image h_output = AllocateHostImage(64, H, W);
    Image d_output = MakeDeviceImage(h_output, false);


    FilterSet h_filter = GenerateFilterSet(K, C, FH, FW);
    FilterSet d_filter = MakeDeviceFilterSet(h_filter, true);


    // Print input dimensions
    std::cout << "Input dimensions: "
              << "C=" << h_input.depth << ", "
              << "H=" << h_input.height << ", "
              << "W=" << h_input.width << std::endl;

    // Print filter dimensions
    std::cout << "Filter dimensions: "
              << "K=" << h_filter.count << ", "
              << "C=" << h_filter.depth << ", "
              << "FH=" << h_filter.height << ", "
              << "FW=" << h_filter.width << std::endl;

    // Input tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1, h_input.depth, h_input.height, h_input.width));
    
    // Output tensor descriptor
    cudnnTensorDescriptor_t output_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_DOUBLE,
                               1, 64, h_output.height, h_output.width));

    // Filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc,
                               CUDNN_DATA_DOUBLE,
                               CUDNN_TENSOR_NCHW,
                               h_filter.count, h_filter.depth, h_filter.height, h_filter.width));

    cudnnConvolutionDescriptor_t conv_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                    /*pad_h=*/1, /*pad_w=*/1,
                                    /*stride_h=*/1, /*stride_w=*/1,
                                    /*dilation_h=*/1, /*dilation_w=*/1,
                                    CUDNN_CONVOLUTION,
                                    CUDNN_DATA_DOUBLE));

    // int N_out, K_out, H_out, W_out;
    // checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc,
    //                                       input_desc,
    //                                       filter_desc,
    //                                       &N_out, &K_out, &H_out, &W_out));



    const int max_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returned_algos;
    cudnnConvolutionFwdAlgoPerf_t perf_results[max_algos];

    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
                                         input_desc,
                                         filter_desc,
                                         conv_desc,
                                         output_desc,
                                         max_algos,
                                         &returned_algos,
                                         perf_results));

    cudnnConvolutionFwdAlgo_t conv_algo = perf_results[0].algo;


    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_desc,
                                            filter_desc,
                                            conv_desc,
                                            output_desc,
                                            conv_algo,
                                            &workspace_size));
    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_size);


    const double alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                            &alpha,
                            input_desc,
                            d_input.elements,
                            filter_desc,
                            d_filter.elements,
                            conv_desc,
                            conv_algo,
                            d_workspace,
                            workspace_size,
                            &beta,
                            output_desc,
                            d_output.elements));
    
    cudaMemcpy(h_output.elements, d_output.elements, GetElementCount(d_output)*sizeof(double), cudaMemcpyDeviceToHost);

    // Print output dimensions
    std::cout << "Output dimensions: "
              << "C=" << h_output.depth << ", "
              << "H=" << h_output.height << ", "
              << "W=" << h_output.width << std::endl;


    double checksum = ImageChecksum(h_output);

    // Print the checksum without scientific notation
    std::cout << std::fixed << "Checksum: " << checksum << std::endl;

    // Clean up
    free(h_input.elements);
    free(h_filter.elements);
    free(h_output.elements);
    cudaFree(d_input.elements);
    cudaFree(d_filter.elements);
    cudaFree(d_output.elements);
    cudaFree(d_workspace);
    checkCUDNN(cudnnDestroyTensorDescriptor(input_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_desc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}
