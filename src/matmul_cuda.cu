#include <cuda_runtime.h>
#include "matmul_cuda.hpp"
#include "ir_trace.hpp"
#include <cassert>
#include <iostream>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

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
    // Ensure CUDA is initialized
    CUDA_CHECK(cudaFree(0));
    
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
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy input tensors to GPU - both A and B are already on GPU, so we copy device-to-device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to the result tensor (which is also on GPU)
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Record the operation in IR trace
    IRTrace::record("mat_mul_cuda", {A.get_id(), B.get_id()}, C.get_id(), C.shape(), C.get_device());

    return C;
}
