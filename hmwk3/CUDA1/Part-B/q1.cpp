#include <iostream>
#include <sstream>
#include <chrono>

#include "partb.h"

using namespace std;
using namespace std::chrono;

void sum_arrays(size_t count, float *input1, float *input2, float *output) {
    for (size_t i = 0; i < count; i ++){
        output[i] = input1[i] + input2[i];
    }
}

alloc_sum_result time_sum_arrays(size_t count, size_t blocks, size_t threads) {
    alloc_sum_result result;
    result.alloc_time = microseconds(-1);
    result.populate_time = microseconds(-1);
    result.calc_time = microseconds(-1);
    result.total_time = microseconds(-1);

    if (blocks != 1 && threads != 1) {
        cout << "CPU only supports 1 block and 1 thread";
        return result;
    }

    cout << "Blocks=" << blocks << "\nThreads/block=" << threads << endl;

    auto t0 = high_resolution_clock::now();
    
    float *array1, *array2, *out;
    array1 = (float *)malloc(count * sizeof(float));
    if (!array1)
        return result;

    array2 = (float *)malloc(count * sizeof(float));
    if (!array2){
        free(array1);
        return result;
    }

    out = (float *)malloc(count * sizeof(float));
    if (!out){
        free(array1);
        free(array2);
        return result;
    }
    
    auto t1 = high_resolution_clock::now();
    result.alloc_time = duration_cast<microseconds>(t1 - t0);

    for (size_t i = 0; i < count; i++) {
        array1[i] = (float) i;
        array2[i] = (float) i;
    }

    auto t2 = high_resolution_clock::now();
    result.populate_time = duration_cast<microseconds>(t2 - t1);

    sum_arrays(count, array1, array2, out);

    auto t3 = high_resolution_clock::now();
    result.calc_time = duration_cast<microseconds>(t3-t2);

    // This is a bit pointless here
    verify_sum_arrays(count, array1, array2, out);

    result.last_sum = out[count-1];

    free(array1);
    free(array2);
    free(out);

    auto t4 = high_resolution_clock::now();
    result.total_time = duration_cast<microseconds>(t4-t0);
    return result;
}

int main(int const argc, char *argv[]) {

    if (argc != 2){
        cout << "Usage: " << argv[0] << " ELEMS\n";
        exit(1);
    }

    size_t elems;
    stringstream sstream(argv[1]);
    sstream >> elems;
    if (elems == 0) {
        cout << "Invalid element size\n";
        exit(1);
    }

    cout << "K=" << elems << "M" << endl;
    elems *= 1e6;
    auto result = time_sum_arrays(elems, 1, 1);
    print_results(result);
    write_to_csv("q1", "-", elems, result.calc_time);
}