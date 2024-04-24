#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    for (int i = 0; i < n - 1; i++) {
        swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
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

void parallelBubbleSort(vector<int>& arr) {
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

        #pragma omp barrier
    }
}

int main() {
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    vector<int> arrSequential = arr;
    vector<int> arrParallel = arr;

    cout << "Original array: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;

    // Sequential Bubble Sort
    clock_t startTime = clock();
    sequentialBubbleSort(arrSequential);
    clock_t endTime = clock();

    cout << "Sorted array using Sequential Bubble Sort: ";
    for (int num : arrSequential) {
        cout << num << " ";
    }
    cout << endl;

    cout << "Sequential Bubble Sort Time: " << static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC << " seconds" << endl;

    // Parallel Bubble Sort
    double startTimeParallel = omp_get_wtime();
    parallelBubbleSort(arrParallel);
    double endTimeParallel = omp_get_wtime();

    cout << "Sorted array using Parallel Bubble Sort: ";
    for (int num : arrParallel) {
        cout << num << " ";
    }
    cout << endl;

    cout << "Parallel Bubble Sort Time: " << endTimeParallel - startTimeParallel << " seconds" << endl;

    return 0;
}

