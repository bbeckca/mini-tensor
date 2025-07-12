#include <cuda_runtime.h>
#include "matmul_cuda.hpp"
#include "ir_trace.hpp"
#include <cassert>
#include <iostream>

// Helper macro to check CUDA errors and throw exceptions on failure.
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

__global__ void matmul_batch_kernel(
    const float* A, const float* B, float* C,
    int BATCH, int M, int N, int K)
{
    int batch = blockIdx.z;
    int row   = blockIdx.y * blockDim.y + threadIdx.y;
    int col   = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a = A[batch * M * K + row * K + i];     // A[batch][row][i]
            float b = B[batch * K * N + i * N + col];     // B[batch][i][col]
            sum += a * b;
        }
        C[batch * M * N + row * N + col] = sum;          // C[batch][row][col]
    }
}

__global__ void add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

Tensor2D mat_mul_cuda(const Tensor2D& A, const Tensor2D& B) {
    // Initialize CUDA context.
    CUDA_CHECK(cudaFree(0));
    
    int M = A.rows();
    int K = A.cols();
    int N = B.cols();

    // Validate input tensors are on GPU.
    if (A.get_device() != Device::GPU || B.get_device() != Device::GPU) {
        throw std::invalid_argument("mat_mul_cuda: inputs must be on GPU");
    }

    // Check matrix multiplication compatibility.
    if (K != B.rows()) {
        throw std::invalid_argument("mat_mul_cuda: incompatible shapes");
    }

    // Create result tensor on GPU.
    Tensor2D C(M, N, 0.0f, Device::GPU);

    // Allocate temporary device memory for kernel execution.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy input tensors to temporary device memory.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Launch matrix multiplication kernel.
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for kernel launch errors and synchronize.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to the result tensor.
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up temporary device memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Record the operation in IR trace.
    IRTrace::record("mat_mul_cuda", {A.get_id(), B.get_id()}, C.get_id(), C.shape(), C.get_device());

    return C;
}

Tensor2D add_cuda(const Tensor2D& A, const Tensor2D& B) {
    // Initialize CUDA context.
    CUDA_CHECK(cudaFree(0));

    int M = A.rows();
    int N = A.cols();

    // Validate input tensors are on GPU.
    if (A.get_device() != Device::GPU || B.get_device() != Device::GPU)
        throw std::invalid_argument("add_cuda: inputs must be on GPU");

    // TODO: Add support for broadcasting.
    // Check matrix addition compatibility.
    if (M != B.rows() || N != B.cols())
        throw std::invalid_argument("add_cuda: incompatible shapes");

    // Create result tensor on GPU.
    Tensor2D C(M, N, 0.0f, Device::GPU);

    // Allocate temporary device memory for kernel execution.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy input tensors to temporary device memory.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * N * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Launch matrix addition kernel.
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N);

    // Check for kernel launch errors and synchronize.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to the result tensor.
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up temporary device memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Record the operation in IR trace.
    IRTrace::record("add_cuda", {A.get_id(), B.get_id()}, C.get_id(), C.shape(), C.get_device());

    return C;
}

Tensor3D bmm_cuda(const Tensor3D& A, const Tensor3D& B) {
    // Initialize CUDA context.
    CUDA_CHECK(cudaFree(0));

    int BATCH = A.batch_size();
    int M = A.rows();
    int K = A.cols();
    int N = B.cols();

    // Validate input tensors are on GPU.
    if (A.get_device() != Device::GPU || B.get_device() != Device::GPU)
        throw std::invalid_argument("bmm_cuda: inputs must be on GPU");

    // Check batch matrix multiplication compatibility.
    if (K != B.rows() || BATCH != B.batch_size())
        throw std::invalid_argument("bmm_cuda: incompatible shapes");

    // Create result tensor on GPU.
    Tensor3D C(BATCH, M, N, 0.0f, Device::GPU);

    // Allocate temporary device memory for kernel execution.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, BATCH * M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, BATCH * K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, BATCH * M * N * sizeof(float)));

    // Copy input tensors to temporary device memory.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), BATCH * M * K * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), BATCH * K * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Launch batched matrix multiplication kernel.
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, BATCH);
    matmul_batch_kernel<<<blocks, threads>>>(d_A, d_B, d_C, BATCH, M, N, K);

    // Check for kernel launch errors and synchronize.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to the result tensor.
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, BATCH * M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up temporary device memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Record the operation in IR trace.
    IRTrace::record("bmm_cuda", {A.get_id(), B.get_id()}, C.get_id(), C.shape(), C.get_device());

    return C;
}
