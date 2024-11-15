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

// By default, use coalesced writing
//#define NO_COALESCED_WRITE


#define NUM_THREADS (BLOCK_SIZE*BLOCK_SIZE)
#define NUM_TILE_ELEMS (FOOTPRINT_SIZE*FOOTPRINT_SIZE)
// For loading, we load rows (of size FOOTPRINT_SIZE) concurrently with NUM_THREADS threads 
#define LOAD_ROUNDS (NUM_TILE_ELEMS/NUM_THREADS)
// Number of tile rows loaded at once
#define CONCUR_ROWS (NUM_THREADS/FOOTPRINT_SIZE)

// Coalseced matrix multiplication kernel 
// Splits multiplication into three rounds, for each phase 
// 1. load A and B into shared memory (in coalescing-maximizing order)
// 2. compute C values (in naive order)
// 3. write C values to global memory (in coalescing-maximizing order)

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_col = threadIdx.x;
  int thread_row = threadIdx.y;
  
  // 'Consuective for the current thread id' in the order that maximizes coalescing
  int consec_id = thread_col + thread_row * BLOCK_SIZE;

  // Which tile the thread should load coalesced (different than the tile it calculates)
  int tile_load_row = (consec_id / FOOTPRINT_SIZE);
  int tile_load_col = (consec_id % FOOTPRINT_SIZE);
  
  // This thread working on chunk thread_row,thread_col 
  // Starts at element thread_row*2, thread_col*2
  int tile_calc_row = thread_row*2;
  int tile_calc_col = thread_col*2;

  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Output tile for this thread 
  float Cvalue[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  // For each tiling phase 
  for (int phase = 0; phase < (A.width / FOOTPRINT_SIZE); ++phase){
    // Get descriptors for Asub and Bsub
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * phase];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * phase + FOOTPRINT_SIZE * block_col];

    // 32x32 shared memory for A and B inputs
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

    // Do an inproduct of four elements from arow of shared_A and four elements from  col of shared_B
    // computing 4 Cvalues by accumulation
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
  

  // Write outputs to global memory
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];


  // Experimented with two options: writing directly to global memory or merging in shared memory then writing 
  #ifndef NO_COALESCED_WRITE
    // Option 1: coalesced--each thread writes its own value to shared memory, then writes coalesced to global memory
    // Option 1 experimentally is faster 
    __shared__ float shared_C[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // All threads write their values to shared memory
    shared_C[tile_calc_row][tile_calc_col] = Cvalue[0][0];
    shared_C[tile_calc_row][tile_calc_col+1] = Cvalue[0][1];
    shared_C[tile_calc_row+1][tile_calc_col] = Cvalue[1][0];
    shared_C[tile_calc_row+1][tile_calc_col+1] = Cvalue[1][1];

    __syncthreads();

    // All threads write coalesced to global memory
    // Require (FOOTPRINT_SIZE/BLOCK_SIZE)^2 iterations
    for (int ld_round = 0; ld_round < LOAD_ROUNDS; ld_round++){ 
      // Load each row using FOOTPRINT_SIZE consecutive threads
      // Need to map thread coordinate (16x16) onto tile load coordinate (32xLOAD_ROUNDS)
      int round_load_row = ld_round*CONCUR_ROWS + tile_load_row;
      int global_tile_index = round_load_row*A.stride + tile_load_col;
      Csub[global_tile_index] = shared_C[round_load_row][tile_load_col];
    }
  #else
    // Option 2: not coalesced--each thread writes its own 4 values directly to global memory 
    Csub[tile_calc_row * C.stride + tile_calc_col] = Cvalue[0][0];
    Csub[tile_calc_row * C.stride + (tile_calc_col + 1)] = Cvalue[0][1];
    Csub[(tile_calc_row + 1) * C.stride + tile_calc_col] = Cvalue[1][0];
    Csub[(tile_calc_row + 1) * C.stride + (tile_calc_col + 1)] = Cvalue[1][1];
  #endif

}
