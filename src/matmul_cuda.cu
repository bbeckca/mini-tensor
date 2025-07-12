// TODO(bbeckca): Improvements for bmm_add_cuda
// - Support broadcasted bias shapes (e.g. [1, M, N] or [B, 1, N])
// - Skip GPU memcpy if input tensors already live on device (zero-copy)
// - Fuse with activation (e.g. bmm_add_relu) to reduce launch overhead
// - Benchmark fused vs unfused performance across shape configs

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

__global__ void add_kernel(
    const float* A, const float* B, float* C,
    int M, int N,
    int A_M, int A_N,
    int B_M, int B_N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int a_idx = (row % A_M) * A_N + (col % A_N);
        int b_idx = (row % B_M) * B_N + (col % B_N);
        int c_idx = row * N + col;

        C[c_idx] = A[a_idx] + B[b_idx];
    }
}

__global__ void add_batch_kernel(
    const float* A, const float* B, float* C,
    int M, int N,
    int A_B, int A_M, int A_N,
    int B_B, int B_M, int B_N)
{
    int b = blockIdx.z;                      // batch index
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    if (i < M && j < N) {
        // Compute broadcasted indices
        int a_b = b % A_B;
        int a_i = i % A_M;
        int a_j = j % A_N;

        int b_b = b % B_B;
        int b_i = i % B_M;
        int b_j = j % B_N;

        // Compute flattened offsets
        int a_offset = a_b * A_M * A_N + a_i * A_N + a_j;
        int b_offset = b_b * B_M * B_N + b_i * B_N + b_j;
        int c_offset = b * M * N + i * N + j;

        // Perform element-wise add
        C[c_offset] = A[a_offset] + B[b_offset];
    }
}

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

    if (batch < BATCH && row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a = A[batch * M * K + row * K + i];     // A[batch][row][i]
            float b = B[batch * K * N + i * N + col];     // B[batch][i][col]
            sum += a * b;
        }
        C[batch * M * N + row * N + col] = sum;          // C[batch][row][col]
    }
}

__global__ void bmm_add_kernel(
    const float* A,  // [B, M, K]
    const float* B,  // [B, K, N]
    const float* C,  // [B, M, N] bias
    float* D,        // [B, M, N] output
    int BATCH, int M, int N, int K) {

    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < BATCH && row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            int a_idx = batch * M * K + row * K + k;
            int b_idx = batch * K * N + k * N + col;
            sum += A[a_idx] * B[b_idx];
        }

        int c_idx = batch * M * N + row * N + col;
        D[c_idx] = sum + C[c_idx];
    }
}


Tensor2D add_cuda(const Tensor2D& A, const Tensor2D& B) {
    // Initialize CUDA context.
    CUDA_CHECK(cudaFree(0));

    auto [M, N] = A.infer_broadcast_shape(A.shape(), B.shape());

    // Validate input tensors are on GPU.
    if (A.get_device() != Device::GPU || B.get_device() != Device::GPU)
        throw std::invalid_argument("add_cuda: inputs must be on GPU");

    // Create result tensor on GPU.
    Tensor2D C(M, N, 0.0f, Device::GPU);

    auto [A_M, A_N] = A.shape();
    auto [B_M, B_N] = B.shape();

    // Allocate temporary device memory for kernel execution.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_M * A_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B_M * B_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy input tensors to temporary device memory.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A_M * A_N * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B_M * B_N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Launch matrix addition kernel.
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, A_M, A_N, B_M, B_N);

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


Tensor3D add_cuda(const Tensor3D& A, const Tensor3D& B) {
    // Initialize CUDA context.
    CUDA_CHECK(cudaFree(0));

    auto [C_B, C_M, C_N] = Tensor3D::infer_broadcast_shape_3d(A.shape(), B.shape());

    // Validate input tensors are on GPU.
    if (A.get_device() != Device::GPU || B.get_device() != Device::GPU)
        throw std::invalid_argument("add_cuda: inputs must be on GPU");

    // Create result tensor on GPU.
    Tensor3D C(C_B, C_M, C_N, 0.0f, Device::GPU);

    auto [A_B, A_M, A_N] = A.shape();
    auto [B_B, B_M, B_N] = B.shape();

    // Allocate temporary device memory for kernel execution.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_B * A_M * A_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B_B * B_M * B_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, C_B * C_M * C_N * sizeof(float)));

    // Copy input tensors to temporary device memory.
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A_B * A_M * A_N * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B_B * B_M * B_N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Launch matrix addition kernel.
    dim3 threads(16, 16);
    dim3 blocks((C_N + 15) / 16, (C_M + 15) / 16, C_B);
    add_batch_kernel<<<blocks, threads>>>(d_A, d_B, d_C, C_M, C_N, A_B, A_M, A_N, B_B, B_M, B_N);

    // Check for kernel launch errors and synchronize.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to the result tensor.
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, C_B * C_M * C_N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up temporary device memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Record the operation in IR trace.
    IRTrace::record("add_cuda", {A.get_id(), B.get_id()}, C.get_id(), C.shape(), C.get_device());

    return C;
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

    // Check matrix multiplication compatibility: (M × K) · (K × N)
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

    // Check matrix multiplication compatibility: (B × M × K) · (B × K × N)
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


Tensor3D bmm_add_cuda(const Tensor3D& input, const Tensor3D& weight, const Tensor3D& bias) {
    // Initialize CUDA context.
    CUDA_CHECK(cudaFree(0));

    int BATCH = input.batch_size();
    int M = input.rows();
    int K = input.cols();
    int N = weight.cols();
    
    // Ensure all inputs are on GPU
    if (input.get_device() != Device::GPU ||
        weight.get_device() != Device::GPU ||
        bias.get_device() != Device::GPU) {
        throw std::invalid_argument("bmm_add_cuda: all inputs must be on GPU");
    }
    
    // Check matrix multiplication compatibility: (B × M × K) · (B × K × N)
    if (weight.rows() != K || weight.batch_size() != BATCH) {
        throw std::invalid_argument("bmm_add_cuda: weight shape incompatible with input");
    }
    
    // Check that bias matches output shape exactly
    if (bias.shape() != std::make_tuple(BATCH, M, N)) {
        throw std::invalid_argument("bmm_add_cuda: bias must match output shape (B, M, N)");
    }    

    // Create result tensor on GPU.
    Tensor3D output(BATCH, M, N, 0.0f, Device::GPU);

    // Allocate temporary device memory for kernel execution.
    float *d_input, *d_weight, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, BATCH * M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, BATCH * K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, BATCH * M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH * M * N * sizeof(float)));

    // Copy input tensors to temporary device memory.
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), BATCH * M * K * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), BATCH * K * N * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), BATCH * M * N * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Launch batched matrix multiplication kernel.
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16, BATCH);
    bmm_add_kernel<<<blocks, threads>>>(d_input, d_weight, d_bias, d_output, BATCH, M, N, K);

    // Check for kernel launch errors and synchronize.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to the result tensor.
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, BATCH * M * N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Clean up temporary device memory.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));

    // Record the operation in IR trace.
    IRTrace::record("bmm_add_cuda", {input.get_id(), weight.get_id(), bias.get_id()}, output.get_id(), output.shape(), output.get_device());

    return output;
}