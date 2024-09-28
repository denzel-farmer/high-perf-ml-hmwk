
#include <stdio.h>
#include <stdlib.h>

float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j=0;j<N;j++)
        R += pA[j]*pB[j];
    return R;
}

int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        fprintf(stderr, "Usage: ./dp1 <vector size> <measurement count>\n");
        return 1; 
    }
    
    unsigned long size = atol(argv[1]);
    unsigned long count = atol(argv[2]);

    printf("Performing %lu measurements on vector of size %lu\n", count, size);
        
    float *vec1 = malloc(sizeof(float)*count);
    float *vec2 = malloc(sizeof(float)*count);

    for (unsigned long i = 0; i < count; i++) {
       vec1[i] = 1.0;
       vec2[i] = 1.0; 
    }

    float product = dp(count, vec1, vec2);

    printf("Result: %f\n", product);


    return 1;
}
