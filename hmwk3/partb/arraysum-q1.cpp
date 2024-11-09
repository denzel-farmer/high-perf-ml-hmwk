#include <iostream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

struct alloc_sum_result {
    microseconds alloc_time;
    microseconds populate_time;
    microseconds calc_time;
    microseconds total_time;
    // Keep the last sum value and print it out to avoid optimizer skipping 
    float last_sum;
};

#define EPS ((float)1e-4)

bool verify_sum_arrays(size_t count, float *input1, float *input2, float *output) {
    for (size_t i = 0; i < count; i++) {
        float exp_sum = input1[i] + input2[i];
        if (abs(output[i] - exp_sum) > EPS)
            return false;
    }

    return true;

}

void sum_arrays(size_t count, float *input1, float *input2, float *output) {
    for (size_t i = 0; i < count; i ++){
        output[i] = input1[i] + input2[i];
    }
}

alloc_sum_result time_sum_arrays(size_t count) {
    alloc_sum_result result;
    auto t0 = high_resolution_clock::now();
    
    float *array1, *array2, *out;
    array1 = (float *)malloc(count * sizeof(float));
    array2 = (float *)malloc(count * sizeof(float));
    out = (float *)malloc(count * sizeof(float));
    
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

    cout << "Using array of size: " << elems << "M" << endl;
    
    auto result = time_sum_arrays(elems*(1e6));
    cout << "Total Time: " << duration_cast<milliseconds>(result.total_time).count() << "ms" << endl;
    cout << "Allocation Time: " << duration_cast<milliseconds>(result.alloc_time).count() << "ms" << endl;
    cout << "Populate Time: " << duration_cast<milliseconds>(result.populate_time).count() << "ms" << endl;
    cout << "Calculation Time: " << duration_cast<milliseconds>(result.calc_time).count() << "ms" << endl;
    cout << "Total Time: " << duration_cast<milliseconds>(result.total_time).count() << "ms" << endl;
    cout << "Last Sum Value: " << result.last_sum << endl;




}