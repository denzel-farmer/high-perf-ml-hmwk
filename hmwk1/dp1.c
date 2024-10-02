#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define FILL_VALUE 1 

// Two memory access per loop, N iterations of the loop
unsigned long dp_bytes_transfered(long N) {
    return 2*N*sizeof(float);
}

// One mult operation, one add operation per loop, N iterations of the loop
unsigned long dp_flops(long N) {
    return 2*N;
}

float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];
    return R;
}

double time_diff(const struct timespec *start, const struct timespec *end) {
    double diff;

    diff = end->tv_sec - start->tv_sec; 
    diff += (end->tv_nsec - start->tv_nsec) / 1e9;
    
    return diff; 
}


int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        fprintf(stderr, "Usage: ./dp1 <vector size> <measurement count>\n");
        return 1; 
    }
    
    unsigned long size = atol(argv[1]);
    unsigned long count = atol(argv[2]);
    
    if (count < 3) {
       fprintf(stderr, "Count must be at least 3\n");
       return 1; 
    }

    printf("Performing %lu measurements on vector of size %lu\n", count, size);
        
    float *vec1 = malloc(sizeof(float)*size);
    float *vec2 = malloc(sizeof(float)*size);

    for (unsigned long i = 0; i < size; i++) {
       vec1[i] = FILL_VALUE;
       vec2[i] = FILL_VALUE; 
    }
    
    struct timespec start, end;
    double total_duration = 0;
    volatile float product; 
    //double expected_product = FILL_VALUE*FILL_VALUE*size; 
    for (unsigned long j = 0; j < count; j++) { 
               
        clock_gettime(CLOCK_MONOTONIC, &start); 

        product = dp(size, vec1, vec2);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        product; // To avoid compiler warning
       /* 
        if (product != expected_product) {
            printf("Product (%f) does not match expected (%f)\n", product, expected_product);
        }
        */
        if (j > (count-1) / 2) 
            total_duration += time_diff(&start, &end);
    }
   
    printf("Result: %f\n", product);

    long num_measurements = (count-1) / 2; 
    // Calculate average duration with arithmetic mean  
    double average_duration = total_duration / num_measurements; 

    // Calculate bandwidth as (bytes transfered) / (duration) 
    double bandwidth = dp_bytes_transfered(size) / average_duration; 
    
    // Calculate throughput as (flops) / (duration) 
    double throughput = dp_flops(size) / average_duration;


    printf("N: %lu\t<T>: %.6f sec\tB: %.3f GB/sec\tF: %.3f GFLOP/sec\n",
            size, average_duration, bandwidth / 1e9, throughput / 1e9);
    return 0;
}
