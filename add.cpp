#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAddition(int *a, int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int size = 1000000; // Size of vectors
    std::vector<int> a(size);
    std::vector<int> b(size);
    std::vector<int> c(size);

    int *d_a, *d_b, *d_c;

    // Initialize host vectors
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch vector addition kernel
    vectorAddition<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);

    // Copy result from device to host
    cudaMemcpy(c.data(), d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

