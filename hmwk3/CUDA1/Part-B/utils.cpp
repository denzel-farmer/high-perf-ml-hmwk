#include "partb.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;



// Write to csv file of the format question,scenario,K,calculation time
#include <fstream>

void write_to_csv(const string& question, const string& scenario, int K, microseconds calc_time) {
    ofstream file;
    file.open("results.csv", ios::out | ios::app);

    if (file.tellp() == 0) {
        file << "question,scenario,K,calculation time\n";
    }

    file << question << "," << scenario << "," << K << "," << calc_time.count() << "\n";
    file.close();
}

void print_results(alloc_sum_result result) {
    // cout << "Total Time: " << duration_cast<milliseconds>(result.total_time).count() << "ms" << endl;
    // cout << "Allocation Time: " << duration_cast<milliseconds>(result.alloc_time).count() << "ms" << endl;
    // cout << "Populate Time: " << duration_cast<milliseconds>(result.populate_time).count() << "ms" << endl;
    cout << "Calculation Time: " << duration_cast<microseconds>(result.calc_time).count() << "us" << endl;
    // cout << "Total Time: " << duration_cast<milliseconds>(result.total_time).count() << "ms" << endl;
    cout << "Last Sum Value: " << result.last_sum << endl;
}

#define EPS ((float)1e-4)

bool verify_sum_arrays(size_t count, float *input1, float *input2, float *output) {
    for (size_t i = 0; i < count; i++) {
        float exp_sum = input1[i] + input2[i];
        if (abs(output[i] - exp_sum) > EPS) {
            cout << "Mismatch at index " << i << ": expected " << exp_sum << ", got " << output[i] << endl;
            return false;
        }
    }

    return true;

}

void run_cuda_tests(size_t elems, const string& question){
    elems *= 1e6;
    // Must be multiple of largest threads (256)
    elems = ((elems + 255) / 256) * 256;

    cout << "K=" << elems << endl;
  

    size_t blocks = 1;
    size_t threads = 1;
    //cout << "\nRunning test with " << blocks << " blocks/grid and " << threads << " threads per block\n";
    cout << "\nBlocks=" << blocks << "\nThreads/block=" << threads << endl;
    auto result = time_sum_arrays(elems, blocks, threads);
    print_results(result);
    write_to_csv(question, "1b-1t", elems, result.calc_time);
   

    blocks = 1;
    threads = 256;
   // cout << "\nRunning test with " << blocks << " blocks/grid and " << threads << " threads per block\n";
    cout << "\nBlocks=" << blocks << "\nThreads/block=" << threads << endl;
    result = time_sum_arrays(elems, blocks, threads);
    print_results(result);
    write_to_csv(question, "1b-256t", elems, result.calc_time);

    threads = 256;
    blocks = elems / 256;
    //cout << "\nRunning test with " << blocks << " blocks/grid and " << threads << " threads per block\n";
    cout << "\nBlocks=" << blocks << "\nThreads/block=" << threads << endl;
    result = time_sum_arrays(elems, blocks, threads);
    print_results(result);
    write_to_csv(question, "MANYb-256t", elems, result.calc_time);

}