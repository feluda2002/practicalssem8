#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

void merge(vector<int>& arr, int low, int mid, int high) {
    int n1 = mid - low + 1;
    int n2 = high - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) {
        L[i] = arr[low + i];
    }
    for (int j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    int i = 0, j = 0, k = low;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void sequentialMergeSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = low + (high - low) / 2;
        sequentialMergeSort(arr, low, mid);
        sequentialMergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

void parallelMergeSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int mid = low + (high - low) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, low, mid);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, high);
        }

        merge(arr, low, mid, high);
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

    // Sequential Merge Sort
    clock_t startTime = clock();
    sequentialMergeSort(arrSequential, 0, arrSequential.size() - 1);
    clock_t endTime = clock();

    cout << "Sorted array using Sequential Merge Sort: ";
    for (int num : arrSequential) {
        cout << num << " ";
    }
    cout << endl;

    cout << "Sequential Merge Sort Time: " << static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC << " seconds" << endl;

    // Parallel Merge Sort
    double startTimeParallel = omp_get_wtime();
    parallelMergeSort(arrParallel, 0, arrParallel.size() - 1);
    double endTimeParallel = omp_get_wtime();

    cout << "Sorted array using Parallel Merge Sort: ";
    for (int num : arrParallel) {
        cout << num << " ";
    }
    cout << endl;

    cout << "Parallel Merge Sort Time: " << endTimeParallel - startTimeParallel << " seconds" << endl;

    return 0;
}

