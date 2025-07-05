#include <cuda_runtime.h>
#include "matmul_cuda.hpp"
#include "ir_trace.hpp"
#include <cassert>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

Tensor2D mat_mul_cuda(const Tensor2D& A, const Tensor2D& B) {
    int M = A.rows();
    int K = A.cols();
    int N = B.cols();

    if (A.get_device() != Device::GPU || B.get_device() != Device::GPU) {
        throw std::invalid_argument("mat_mul_cuda: inputs must be on GPU");
    }

    if (K != B.rows()) {
        throw std::invalid_argument("mat_mul_cuda: incompatible shapes");
    }

    // Create result tensor on GPU
    Tensor2D C(M, N, 0.0f, Device::GPU);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy input tensors to GPU (regardless of their device flag, we need the data on GPU for computation)
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host (C tensor's data is on CPU, but tensor is marked as GPU)
    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Record the operation in IR trace
    IRTrace::record("mat_mul_cuda", {A.get_id(), B.get_id()}, C.get_id(), C.shape(), C.get_device());

    return C;
}
