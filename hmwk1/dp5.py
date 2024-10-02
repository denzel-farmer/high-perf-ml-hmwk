import sys as sys
import time as time 
import numpy as np

FILL_VALUE = 1.0

def dp_bytes_transfered(size):
    return 2*size*4

def dp_flops(size):
    return 2*size

def dp(A,B):
    return np.dot(A,B)
       

if (len(sys.argv) != 3):
    print("Usage: python3 dp5.py <vector size> <measurement count>")
    exit(1);

size = int(sys.argv[1])
count = int(sys.argv[2])

print(f"Performing {count} measurements on vector of size {size}")

A = np.full(size, FILL_VALUE, dtype=np.float32)
B = np.full(size, FILL_VALUE, dtype=np.float32)

total_duration = 0
#expected_product = size*FILL_VALUE*FILL_VALUE
for i in range(0, count):

    start = time.clock_gettime(time.CLOCK_MONOTONIC)

    product = dp(A,B)

    end = time.clock_gettime(time.CLOCK_MONOTONIC)
    
#    if (product != expected_product):
#        print(f"Product ({product}) not equal to expected product ({expected_product})")
    
    if (i > (count-1) // 2):
        total_duration += end - start

num_measurements = (count-1) / 2

average_duration = total_duration / num_measurements
bandwidth = dp_bytes_transfered(size) / average_duration
throughput = dp_flops(size) / average_duration

print(f"N: {size}\t<T>: {average_duration:.6f} sec\tB: {bandwidth / 1e9:.3f} GB/sec\tF: {throughput / 1e9:.3f} GFLOP/sec")



