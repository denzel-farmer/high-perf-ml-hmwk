#ifndef PARTB_H
#define PARTB_H

#include <chrono>

struct alloc_sum_result {
    std::chrono::microseconds alloc_time;
    std::chrono::microseconds populate_time;
    std::chrono::microseconds calc_time;
    std::chrono::microseconds total_time;
    // Keep the last sum value and print it out to avoid optimizer skipping 
    float last_sum;
};

// Defined in utils.cpp
void print_results(alloc_sum_result result);
bool verify_sum_arrays(size_t count, float *input1, float *input2, float *output);
void run_cuda_tests(size_t elems);

// Defined in q1/q2/q3
alloc_sum_result time_sum_arrays(size_t count, size_t blocks, size_t threads);

#endif // PARTB_H
