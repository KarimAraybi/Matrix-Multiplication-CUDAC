%%cu
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define M 10000
#define N 5000
#define K 9000

__global__ void MatrixMultiplicationBasic(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

int main() {

    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    srand(time(NULL));
    for (int i = 0; i < M * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N * K; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    auto start = std::chrono::steady_clock::now();
    MatrixMultiplicationBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);


    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution Time: " << duration.count() << " milliseconds" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
