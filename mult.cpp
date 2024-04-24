#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define N 4 // Matrix size

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    std::vector<int> a(N * N);
    std::vector<int> b(N * N);
    std::vector<int> c(N * N);

    int *d_a, *d_b, *d_c;

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copy host matrices to device
    cudaMemcpy(d_a, a.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(2, 2);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch matrix multiplication kernel
    matrixMultiplication<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result from device to host
    cudaMemcpy(c.data(), d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

