import sys as sys
import time as time 
import numpy as np

def dp_bytes_transfered(size):
    return 2*size*4

def dp_flops(size):
    return 2*size

def dp(N,A,B):
    R = 0.0;
    
    for j in range(0,N):
        R += A[j]*B[j]
    
    return R




if (len(sys.argv) != 3):
    print("Usage: python3 dp4.py <vector size> <measurement count>")
    exit(1);

size = int(sys.argv[1])
count = int(sys.argv[2])

print(f"Performing {count} measurements on vector of size {size}")

A = np.ones(size, dtype=np.float32)
B = np.ones(size, dtype=np.float32)

total_duration = 0

for _ in range(0, count):

    start = time.clock_gettime(time.CLOCK_MONOTONIC)

    product = dp(size,A,B)

    end = time.clock_gettime(time.CLOCK_MONOTONIC)
    
    total_duration += end - start


average_duration = total_duration / count
bandwidth = dp_bytes_transfered(size) / average_duration
throughput = dp_flops(size) / average_duration

print(f"N: {size}\t<T>: {average_duration:.6f} sec\tB: {bandwidth / 1e9:.3f} GB/sec\tF: {throughput / 1e9:.3f} GFLOP/sec")



