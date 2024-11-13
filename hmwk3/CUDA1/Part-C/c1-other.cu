#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

using namespace std;
using namespace chrono;

// checksum expected: 122756344698240.000000000000000

#define DTYPE double


struct Matrix3 {
    int d1; // C
    int d2; // W
    int d3; // H
    DTYPE* elements;
};

struct Matrix4 {
    int d1;
    int d2;
    int d3;
    int d4;
    DTYPE* elements;
};

__host__ __device__ DTYPE get_element(const Matrix3 M, int i, int j, int k) {
    return M.elements[i * M.d3 * M.d2 + j * M.d2 + k];
}

__host__ __device__ void set_element(const Matrix3 M, int i, int j, int k, DTYPE val) {
    M.elements[i * M.d3 * M.d2 + j * M.d2 + k] = val;
}

__host__ __device__ DTYPE get_element(const Matrix4 M, int i, int j, int k, int l) {
    return M.elements[i * M.d4 * M.d3 * M.d2 + j * M.d4 * M.d3 + k * M.d4 + l];
}

__host__ __device__ void set_element(const Matrix4 M, int i, int j, int k, int l, DTYPE val) {
    M.elements[i * M.d4 * M.d3 * M.d2 + j * M.d4 * M.d3 + k * M.d4 + l] = val;
}

Matrix3 MakeDeviceMatrix3(Matrix3 M, bool copy){
    Matrix3 newDeviceMatrix;
    newDeviceMatrix.d1 = M.d1;
    newDeviceMatrix.d2 = M.d2;
    newDeviceMatrix.d3 = M.d3;
    size_t size = M.d1 * M.d2 * M.d3 * sizeof(DTYPE);
    cudaMalloc((void**) &newDeviceMatrix.elements, size);
    if (copy)
        cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
    return newDeviceMatrix;
}

Matrix4 MakeDeviceMatrix4(Matrix4 M, bool copy){
    Matrix4 newDeviceMatrix;
    newDeviceMatrix.d1 = M.d1;
    newDeviceMatrix.d2 = M.d2;
    newDeviceMatrix.d3 = M.d3;
    newDeviceMatrix.d4 = M.d4;
    size_t size = M.d1 * M.d2 * M.d3 * M.d4 * sizeof(DTYPE);
    cudaMalloc((void**) &newDeviceMatrix.elements, size);
    if (copy)
        cudaMemcpy(newDeviceMatrix.elements, M.elements, size, cudaMemcpyHostToDevice);
    return newDeviceMatrix;
}

__global__ void convKernel(Matrix3 O, Matrix4 F, Matrix3 I_0) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // x = 0
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y = 0
    int k = blockIdx.z * blockDim.z + threadIdx.z; // k = 0, 1, ..., 64

    int C = I_0.d1;
    int FH = F.d4;
    int FW = F.d3;

    int c, j, i;

    DTYPE out = 0;
    DTYPE f, i0;

    for (c = 0; c < C; c++) {
        for (j = 0; j < FH; j++) {
            for (i = 0; i < FW; i++) {
               
                f = get_element(F, k, c, FW-1-i, FH-1-j);
                i0 = get_element(I_0, c, x+i, y+j);
                out += f * i0;
            }
        }

    }
    set_element(O, k, x, y, out);

}


int main(int argc, char* argv[]) {

    const int H = 1024;
    const int W = 1024;
    const int C = 3;
    const int FW = 3;
    const int FH = 3;
    const int K = 64;
    const int P = 1;
    int c, x, y, k, i, j;

    Matrix3 I_0;
    I_0.d1 = C;
    I_0.d2 = W+2*P;
    I_0.d3 = H+2*P;
    I_0.elements = new DTYPE[C * (H+2*P) * (W+2*P)];
    for (c = 0; c < C; c++) {
        for (x = 0; x < W + 2 * P; x++) {
            for (y = 0; y < H + 2 * P; y++) {

                if (x < P || y < P || y > W+P-1 || x > H+P-1) {
                    set_element(I_0, c, x, y, 0);
                } else{
                    set_element(I_0, c, x, y, c * (x-1 + y-1));
                }
               
            }
        }
    }

    Matrix4 F = {K, C, FH, FW, new DTYPE[K * C * FW * FH]};
    for (k = 0; k < K; k++) {
        for (c = 0; c < C; c++) {
            for (i = 0; i < FH; i++) {
                for (j = 0; j < FW; j++) {
                    set_element(F, k, c, i, j, (c + k) * (i + j));
                }
            }
        }
    }

    Matrix3 O = {K, W, H, new DTYPE[K * W * H]};

    Matrix3 d_I_0 = MakeDeviceMatrix3(I_0, true);
    Matrix4 d_F = MakeDeviceMatrix4(F, true);
    Matrix3 d_O = MakeDeviceMatrix3(O, true);

    int block_dim_x = 32;
    int block_dim_y = 32;
    int block_dim_z = 1;

    int dim_grid_x = W / block_dim_x;
    int dim_grid_y = H / block_dim_y;
    int dim_grid_z = K / block_dim_z;

    dim3 dimBlock(block_dim_x, block_dim_y, block_dim_z);
    dim3 dimGrid(dim_grid_x, dim_grid_y, dim_grid_z);

    convKernel<<<dimGrid, dimBlock>>>(d_O, d_F, d_I_0);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    convKernel<<<dimGrid, dimBlock>>>(d_O, d_F, d_I_0);
    cudaDeviceSynchronize();
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    cout << "[dimBlock] x: " << block_dim_x << " y: " << block_dim_y << " z: " << block_dim_z << endl;
    cout << "[dimGrid] x: " << dim_grid_x << " y: " << dim_grid_y << " z: " << dim_grid_z << endl;
    cout << "executed in: " << duration << " us " << endl;

    size_t size = O.d1 * O.d2 * O.d3 * sizeof(DTYPE);
    cudaMemcpy(O.elements, d_O.elements, size, cudaMemcpyDeviceToHost);

    DTYPE checksum = 0;

    for (i = 0; i < O.d1; i++) {
        for (j = 0; j < O.d2; j++) {
            for (k = 0; k < O.d3; k++) {
                checksum += get_element(O, i, j, k);
            }
        }
    }

    cout << "checksum: " << fixed << checksum << "\n";

}

