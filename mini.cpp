#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    vector<int> arr = {5, 8, 3, 2, 9, 4, 6, 1, 7};

    // Size of the array
    int n = arr.size();
    int min_val = arr[0];
    int max_val = arr[0];
    int sum = 0;

    // Find minimum, maximum, and sum using parallel reduction
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum += arr[i];
    }

    double average = static_cast<double>(sum) / n;

    cout << "Minimum: " << min_val << endl;
    cout << "Maximum: " << max_val << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;

    return 0;
}

