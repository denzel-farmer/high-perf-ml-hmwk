///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE 32

// TODO add support for footprint size of 64? Must split out separate 'warp size' or similar

#define NUM_THREADS (BLOCK_SIZE*BLOCK_SIZE)
#define NUM_TILE_ELEMS (FOOTPRINT_SIZE*FOOTPRINT_SIZE)
// For loading, we load rows (of size FOOTPRINT_SIZE) concurrently with NUM_THREADS threads 
#define LOAD_ROUNDS (NUM_TILE_ELEMS/NUM_THREADS)
// Number of tile rows loaded at once
#define CONCUR_ROWS (NUM_THREADS/FOOTPRINT_SIZE)

// The type Matrix is really a MATRIX DESCRIPTOR. 
// Matrices are stored in row major order:
//       M[row,col] = *(M.elements + row * M.stride + col)
//
// A sub matrix is not copied but allocated in the full matrix.
//
// This requires the stride of the full matrix to properly get to the
// next row of the sub matrix (a block).
//
// Stride is the width in bytes from one element of the larger matrix 
// to the element in the same column but one row down.



  // Block i,j computes on A/B/C i*FP_SIZE,j*FP_SIZE -> (i+1)*FP_SIZE-1,(j+1)*FP_SIZE-1
  
  // Example: inputs are 64x64, block size is 32x32 
  // Block 0,0 works on A/B/C [0,0] to [1*32-1,1*32-1] -> [0,0] to [31, 31]
  // Block 0,1 works on A/B/C [0*32,1*32] to [1*32-1, 2*32-1] -> [0,32] to [31, 63]

  // Within block 0,0 thread 0,0 computes result values C[0,0],C[0,1],C[1,0],C[1,1]
  // [0,0]: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
  // [0,1]: 

  // Phase 1: 
  // - Block 0,0 loads A tile 0,0 and B tile 0,0 -> computes partial results for each thread 
  // - Block 0,1 loads A tile 0,0 and B tile 0,1 -> computes partial results for each thread 
  // ....
  // Phase 2:
  // - Block 0,0 loads A tile 0,1 and B tile 1,0 -> adds to partial result for each thread
  // - Block 0,1 loads A tile 0,1 and B tile 1,1 -> add to partial result for each thread
  // ...
  
  // Overall structure of tiles and blocks
  // Total of WIDTH/TILE_SIZE = WIDTH/BLOCK_SIZE phases
  // In phase n:
    // Block 0,0 loads A tile 0,n and B tile n,0
    // Block 0,1 loads A tile 0,n and B tile n,1
    // Block i,j loads A tile i,n and B tile n,j

  // Within block,phase given input tiles A_tile and B_tile 
    // Each thread loads one lement of tile, block threads should coalesce loading rows from A_tile and B_tile 
    // Split tile into 2x2 chunks, each thread works on a chunk

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_col = threadIdx.x;
  int thread_row = threadIdx.y;
  // Consec_id = thread_col + thread_row * BLOCK_SIZE
  int consec_id = thread_col + thread_row * BLOCK_SIZE;
  int tile_load_row = (consec_id / FOOTPRINT_SIZE);
  int tile_load_col = (consec_id % FOOTPRINT_SIZE);

  // This thread working on chunk thread_row,thread_col 
  // Starts at element thread_row*2, thread_col*2
  int tile_calc_row = thread_row*2;
  int tile_calc_col = thread_col*2;

  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  float Cvalue[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  // For each phase 
  for (int phase = 0; phase < (A.width / FOOTPRINT_SIZE); ++phase){
    // Get descriptors for Asub and Bsub
    // Tile i,j has corners [i*BLOCK_SIZE, j*BLOCK_SIZE] and [(i+1)*BLOCK_SIZE-1, (j+1)*BLOCK_SIZE-1]
    // Asub is tile (Block.x, phase) -> starts at [Block.x*BLOCK_SIZE, phase*BLOCK_SIZE]
    // Index into elements is elements + row*stride + col -> (Block.x*BLOCK_SIZE)*A.stride + phase*BLOCK_SIZE
    //
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * phase];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * phase + FOOTPRINT_SIZE * block_col];

    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Do a coalesced load for tile of A and B (load each 32-element row)
    // Load row i with threads 0 to FOOTPRINT_SIZE-1
    // Concurrently, can load NUM_THREADS / FOOTPRINT_SIZE rows 
    // Require (FOOTPRINT_SIZE/BLOCK_SIZE)^2 iterations
    for (int ld_round = 0; ld_round < LOAD_ROUNDS; ld_round++){ 
      // Load each row using FOOTPRINT_SIZE consecutive threads
      // Need to map thread coordinate (16x16) onto tile load coordinate (32xLOAD_ROUNDS)
      int round_load_row = ld_round*CONCUR_ROWS + tile_load_row;
      int global_tile_index = round_load_row*A.stride + tile_load_col;
      shared_A[round_load_row][tile_load_col] = Asub[global_tile_index];
      shared_B[round_load_row][tile_load_col] = Bsub[global_tile_index];
    }

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing 4 Cvalues by accumulation
    // TODO allow for variable chunk size?
#pragma unroll
    for(int e=0; e<FOOTPRINT_SIZE; ++e) {
      Cvalue[0][0] += shared_A[tile_calc_row][e] * shared_B[e][tile_calc_col];
      Cvalue[0][1] += shared_A[tile_calc_row][e] * shared_B[e][tile_calc_col + 1];
      Cvalue[1][0] += shared_A[tile_calc_row+1][e] * shared_B[e][tile_calc_col];
      Cvalue[1][1] += shared_A[tile_calc_row+1][e] * shared_B[e][tile_calc_col + 1];      
    }
    
    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }


  // Write Csub to GLOBAL memory
  // Currently not coalesced--each thread writes its own 4 values 
  Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

  Csub[tile_calc_row*C.stride + tile_calc_col] = Cvalue[0][0];
  Csub[tile_calc_row*C.stride + (tile_calc_col+1)] = Cvalue[0][1];
  Csub[(tile_calc_row+1)*C.stride + tile_calc_col] = Cvalue[1][0];
  Csub[(tile_calc_row+1)*C.stride + (tile_calc_col+1)] = Cvalue[1][1];
}

// // Define a gpu kernel to perform matrix multiplication
// // of A x B = C.
// __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

//   // matrix blocks
//   float *Asub, *Bsub, *Csub;
//   // Putting these into registers speeds access.
//   int thread_row = threadIdx.y;
//   int thread_col = threadIdx.x;
//   int block_row = blockIdx.y;
//   int block_col = blockIdx.x;

//   // Each THREAD BLOCK computes one sub matrix Csub of C
//   // EACH THREAD creates its own matrix descriptor Csub
//   Csub = &C.elements[C.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col];

//   // Each thread computes one element of Csub in its copy of CValue
//   float Cvalue = 0;

//   // Loop over all sub matrices in block_row of A and block_col of B
//   // required to compute Csub. Block multiply each pair of sub matrices
//   // and accumulate results
//   for (int m = 0;  m < (A.width / BLOCK_SIZE); ++m){
//     // Get Asub and Bsub descriptors
//     Asub = &A.elements[A.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * m];
//     Bsub = &B.elements[B.stride * BLOCK_SIZE * m + BLOCK_SIZE * block_col];

//     // Copy ELEMENTS OF  ASub and Bsub into shared memory
//     // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
//     // Notice: it does not need to be the element it requires to
//     //         compute its Cvalue, as long as all elements are 
//     //         collaboratively read. 

//     // Notice: every thread declares shared_A and shared_B in shared memory
//     //         even though a thread block has only one shared_A and one shared_B
//     __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

//     // Each thread copies just one element of shared_A and one element of shared_B
//     shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
//     shared_B[thread_row][thread_col] = Bsub[thread_row * B.stride + thread_col];

//     // Synchronize to ensure all elements are read
//     __syncthreads();

//     // Do an inproduct of one row of shared_A and one col of shared_B
//     // computing one Cvalue by accumulation
// #pragma unroll
//     for(int e=0; e<BLOCK_SIZE; ++e)
//        Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];

//     // Synchronize to ensure all Cvalues have been incremented
//     // before reading in the next shared_A AND shared_B BLOCKS
//     __syncthreads();
//   }

//   // Write Csub to GLOBAL memory.
//   // Each thread writes its own cell value.
//   Csub[thread_row * C.stride + thread_col] = Cvalue;
// }

