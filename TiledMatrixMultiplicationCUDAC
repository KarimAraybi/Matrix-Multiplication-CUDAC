%%cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define M 10000
#define N 5000
#define K 9000
#define TILE_SIZE 32

__global__ void matrixMul(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int i = 0; i < (n + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        if (row < m && (i * TILE_SIZE + tx) < n)
            shared_A[ty][tx] = A[row * n + i * TILE_SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if ((i * TILE_SIZE + ty) < n && col < k)
            shared_B[ty][tx] = B[(i * TILE_SIZE + ty) * k + col];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j)
            sum += shared_A[ty][j] * shared_B[j][tx];

        __syncthreads();
    }

    if (row < m && col < k)
        C[row * k + col] = sum;
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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);


    auto start = std::chrono::steady_clock::now();
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

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
