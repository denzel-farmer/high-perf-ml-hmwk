#include <time.h>
#include <stdio.h>
#include <stdlib.h>



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

    printf("Performing %lu measurements on vector of size %lu\n", count, size);
        
    float *vec1 = malloc(sizeof(float)*size);
    float *vec2 = malloc(sizeof(float)*size);

    for (unsigned long i = 0; i < size; i++) {
       vec1[i] = 1.0;
       vec2[i] = 1.0; 
    }
    
    struct timespec start, end;
    double total_duration = 0;
    for (unsigned long j = 0; j < count; j++) { 
       
        clock_gettime(CLOCK_MONOTONIC, &start); 

        volatile float product = dp(size, vec1, vec2);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        product; // To avoid compiler warning

        total_duration += time_diff(&start, &end);
    
    }
    
    // Calculate average duration with arithmetic mean  
    double average_duration = total_duration / count; 

    // Calculate bandwidth as (bytes transfered) / (duration) 
    double bandwidth = dp_bytes_transfered(size) / average_duration; 
    
    // Calculate throughput as (flops) / (duration) 
    double throughput = dp_flops(size) / average_duration;


    printf("N: %lu\t<T>: %.6f sec\tB: %.3f GB/sec\tF: %.3f GFLOP/sec\n",
            size, average_duration, bandwidth / 1e9, throughput / 1e9);
    return 0;
}
