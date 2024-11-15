#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>

using namespace std;

#define ELEM_TYPE float
#define ELEM_TYPE_CUDNN CUDNN_DATA_FLOAT

constexpr int C = 3;            // Number of input channels
constexpr int H = 1024;         // Input height
constexpr int W = 1024;         // Input width
constexpr int K = 64;           // Number of filters/output channels
constexpr int FH = 3;           // Filter height
constexpr int FW = 3;           // Filter width

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }



int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    
    // Input tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_desc,
                               CUDNN_TENSOR_NCHW,
                               ELEM_TYPE_CUDNN,
                               1, C, H, W));
    
    // Output tensor descriptor
    cudnnTensorDescriptor_t output_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_desc,
                               CUDNN_TENSOR_NCHW,
                               ELEM_TYPE_CUDNN,
                               1, K, H, W));

    // Filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc,
                               ELEM_TYPE_CUDNN,
                               CUDNN_TENSOR_NCHW,
                               K, C, FH, FW));

    cudnnConvolutionDescriptor_t conv_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                    /*pad_h=*/1, /*pad_w=*/1,
                                    /*stride_h=*/1, /*stride_w=*/1,
                                    /*dilation_h=*/1, /*dilation_w=*/1,
                                    CUDNN_CONVOLUTION,
                                    ELEM_TYPE_CUDNN));


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

    // Initialize input tensor
    ELEM_TYPE *h_input = (ELEM_TYPE *)malloc(C*H*W*sizeof(ELEM_TYPE));
    for (int c = 0; c < C; c++) {
      for (int y = 0; y < H; y++) {
          for (int x = 0; x < W; x++) {
            ELEM_TYPE value = c * (x + y);
            h_input[(c*H*W) + (y*W) + x] = value;
          }
      }
  }
  
  ELEM_TYPE *d_input;
  cudaMalloc(&d_input, C*H*W*sizeof(ELEM_TYPE));
  cudaMemcpy(d_input, h_input, C*H*W*sizeof(ELEM_TYPE), cudaMemcpyHostToDevice);

  // Initialize filter descriptor
  ELEM_TYPE *h_filter = (ELEM_TYPE *)malloc(K*C*FH*FW*sizeof(ELEM_TYPE));
    for (int k = 0; k < K; k++) {
      for (int c = 0; c < C; c++) {
          for (int i = 0; i < FH; i++) {
              for (int j = 0; j < FW; j++) {
                  ELEM_TYPE value = (c + k)*(i+j);
                  h_filter[(k * C*FH*FW) + (c * FH*FW) + (j * FW) + i] = value;
              }
          }
      }
  }

  ELEM_TYPE *d_filter;
  cudaMalloc(&d_filter, K*C*FH*FW*sizeof(ELEM_TYPE));
  cudaMemcpy(d_filter, h_filter, K*C*FH*FW*sizeof(ELEM_TYPE), cudaMemcpyHostToDevice);

  // Create output tensor
  ELEM_TYPE *h_output = (ELEM_TYPE *)malloc(K*H*W*sizeof(ELEM_TYPE));
  ELEM_TYPE *d_output;
  cudaMalloc(&d_output, K*H*W*sizeof(ELEM_TYPE));
  
  // Start timing
  cudaDeviceSynchronize();
  auto start = chrono::high_resolution_clock::now();

  const ELEM_TYPE alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(cudnn,
              &alpha,
              input_desc,
              d_input,
              filter_desc,
              d_filter,
              conv_desc,
              conv_algo,
              d_workspace,
              workspace_size,
              &beta,
              output_desc,
              d_output));

  cudaDeviceSynchronize();
  auto end = chrono::high_resolution_clock::now();
  
  cudaMemcpy(h_output, d_output, K*H*W*sizeof(ELEM_TYPE), cudaMemcpyDeviceToHost);

    double checksum = 0;
    for (long elem = 0; elem < K*H*W; elem++) {
        checksum += h_output[elem];
    }


    // Print the checksum without scientific notation
    std::cout << std::fixed << "Checksum: " << checksum << std::endl;

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

  chrono::duration<double, milli> duration = end - start;
  std::cout << "Convolution forward pass time: " << duration.count() << " ms" << std::endl;


    // Clean up
    free(h_input);
    free(h_filter);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_workspace);
    checkCUDNN(cudnnDestroyTensorDescriptor(input_desc));
    checkCUDNN(cudnnDestroyTensorDescriptor(output_desc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    checkCUDNN(cudnnDestroy(cudnn));

    return 0;
}
