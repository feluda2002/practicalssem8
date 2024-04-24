#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include <algorithm>

using namespace std;

// Sequential Bubble Sort
void sequential_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

// Parallel Bubble Sort
void parallel_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    bool swapped = true;

    while (swapped) {
        swapped = false;

        #pragma omp parallel for shared(swapped)
        for (int i = 0; i < n - 1; ++i) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    }
}

// Merge Sort Helper Function
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; ++i)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        } else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

// Parallel Merge Sort
void parallel_merge_sort(vector<int>& arr, int l, int r) {
    if (l >= r) {
        return;
    }

    int m = l + (r - l) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        parallel_merge_sort(arr, l, m);

        #pragma omp section
        parallel_merge_sort(arr, m + 1, r);
    }

    merge(arr, l, m, r);
}

// Sequential Merge Sort
void sequential_merge_sort(vector<int>& arr, int l, int r) {
    if (l >= r) {
        return;
    }

    int m = l + (r - l) / 2;

    sequential_merge_sort(arr, l, m);
    sequential_merge_sort(arr, m + 1, r);

    merge(arr, l, m, r);
}

int main() {
    const int N = 10000;
    vector<int> arr(N);

    // Generate random data
    srand(time(nullptr));
    for (int i = 0; i < N; ++i) {
        arr[i] = rand() % 10000;
    }

    // Measure performance of sequential Bubble Sort
    vector<int> arr_seq_bubble = arr;
    clock_t start = clock();
    sequential_bubble_sort(arr_seq_bubble);
    clock_t end = clock();
    double seq_time_bubble = double(end - start) / CLOCKS_PER_SEC;

    cout << "Sequential Bubble Sort Time: " << seq_time_bubble << " seconds" << endl;

    // Measure performance of parallel Bubble Sort
    vector<int> arr_par_bubble = arr;
    start = clock();
    parallel_bubble_sort(arr_par_bubble);
    end = clock();
    double par_time_bubble = double(end - start) / CLOCKS_PER_SEC;

    cout << "Parallel Bubble Sort Time: " << par_time_bubble << " seconds" << endl;

    // Measure performance of sequential Merge Sort
    vector<int> arr_seq_merge = arr;
    start = clock();
    sequential_merge_sort(arr_seq_merge, 0, N - 1);
    end = clock();
    double seq_time_merge = double(end - start) / CLOCKS_PER_SEC;

    cout << "Sequential Merge Sort Time: " << seq_time_merge << " seconds" << endl;

    // Measure performance of parallel Merge Sort
    vector<int> arr_par_merge = arr;
    start = clock();
    parallel_merge_sort(arr_par_merge, 0, N - 1);
    end = clock();
    double par_time_merge = double(end - start) / CLOCKS_PER_SEC;

    cout << "Parallel Merge Sort Time: " << par_time_merge << " seconds" << endl;

    return 0;
}

