#include <iostream>
#include <chrono>
#include <cassert>
#include <Eigen/Core>
#include "../include/tensor2d.hpp"
#include "../include/tensor3d.hpp"
#include "../include/matmul_cuda.hpp"

void benchmark_matmul(size_t M, size_t K, size_t N) {
    std::cout << "Benchmarking matmul with shapes: (" << M << ", " << K << ") * (" << K << ", " << N << ")\n";

    Tensor2D A = Tensor2D(M, K, 1.0f);
    Tensor2D B = Tensor2D(K, N, 2.0f);

    // Manual matmul
    auto start = std::chrono::high_resolution_clock::now();
    Tensor2D C_manual = A.mat_mul(B);
    auto end = std::chrono::high_resolution_clock::now();
    auto manual_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Manual mat_mul: " << manual_us << " us\n";

    // Eigen matmul
    start = std::chrono::high_resolution_clock::now();
    Tensor2D C_eigen = A.mat_mul_eigen(B);
    end = std::chrono::high_resolution_clock::now();
    auto eigen_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Eigen mat_mul  : " << eigen_us << " us\n";

    std::cout << "Speedup        : " << static_cast<float>(manual_us) / eigen_us << "x\n\n";
}

void benchmark_batched_matmul(size_t batch, size_t M, size_t K, size_t N) {
    std::cout << "Benchmarking batched matmul with shapes: ("
              << batch << ", " << M << ", " << K << ") * ("
              << batch << ", " << K << ", " << N << ")\n";

    Tensor3D A(batch, M, K, 1.0f);
    Tensor3D B(batch, K, N, 2.0f);

    // Manual batched matmul
    auto start = std::chrono::high_resolution_clock::now();
    Tensor3D C_manual = A.mat_mul(B);
    auto end = std::chrono::high_resolution_clock::now();
    auto manual_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Manual batched_mat_mul: " << manual_us << " us\n";

    // Eigen batched matmul
    start = std::chrono::high_resolution_clock::now();
    Tensor3D C_eigen = A.mat_mul_eigen(B);
    end = std::chrono::high_resolution_clock::now();
    auto eigen_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Eigen batched_mat_mul  : " << eigen_us << " us\n";

    // Result
    std::cout << "Speedup                : " << static_cast<float>(manual_us) / eigen_us << "x\n\n";
}

void benchmark_batched_matmul_parallel(size_t batch, size_t M, size_t K, size_t N) {
    std::cout << "Benchmarking batched matmul parallel with shapes: ("
              << batch << ", " << M << ", " << K << ") * ("
              << batch << ", " << K << ", " << N << ")\n";

    Tensor3D A(batch, M, K, 1.0f);
    Tensor3D B(batch, K, N, 2.0f);

    // Eigen batched matmul
    auto start = std::chrono::high_resolution_clock::now();
    Tensor3D C_manual = A.mat_mul_eigen(B);
    auto end = std::chrono::high_resolution_clock::now();
    auto eigen_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Eigen batched_mat_mul  : " << eigen_us << " us\n";

    // Eigen batched matmul in parallel
    start = std::chrono::high_resolution_clock::now();
    Tensor3D C_eigen = A.mat_mul_eigen_parallel(B);
    end = std::chrono::high_resolution_clock::now();
    auto eigen_us_parallel = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Eigen batched_mat_mul in parallel  : " << eigen_us_parallel << " us\n";

    // Result
    std::cout << "Speedup                : " << static_cast<float>(eigen_us) / eigen_us_parallel << "x\n\n";
}

#ifdef USE_CUDA
void benchmark_matmul_cuda_vs_cpu(int N) {
    std::cout << "Benchmarking matmul cpu vs. gpu with shape: " << N << " x " << N << std::endl;

    // Create CPU tensors
    Tensor2D A_cpu = Tensor2D::from_random(N, N, Device::CPU);
    Tensor2D B_cpu = Tensor2D::from_random(N, N, Device::CPU);

    // CPU baseline
    auto cpu_start = std::chrono::high_resolution_clock::now();
    Tensor2D C_cpu = A_cpu.mat_mul_eigen(B_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // Manually copy to GPU tensors
    Tensor2D A_gpu(N, N, 0.0f, Device::GPU);
    Tensor2D B_gpu(N, N, 0.0f, Device::GPU);
    cudaMemcpy(A_gpu.data(), A_cpu.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu.data(), B_cpu.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // GPU matmul
    auto gpu_start = std::chrono::high_resolution_clock::now();
    Tensor2D C_gpu = mat_mul_cuda(A_gpu, B_gpu);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // Result
    double speedup = cpu_ms / gpu_ms;
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << "Speedup:  " << speedup << "x\n\n";
}

void benchmark_device_transfer(size_t N) {
    std::cout << "Benchmarking device transfer with shape: " << N << " x " << N << std::endl;

    // Create original CPU tensor with random data
    Tensor2D original = Tensor2D::from_random(N, N, Device::CPU);
    
    // Time CPU → GPU transfer
    auto cpu_to_gpu_start = std::chrono::high_resolution_clock::now();
    Tensor2D gpu_tensor = original.to(Device::GPU);
    auto cpu_to_gpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_to_gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_to_gpu_end - cpu_to_gpu_start).count();
    
    // Time GPU → CPU transfer
    auto gpu_to_cpu_start = std::chrono::high_resolution_clock::now();
    Tensor2D cpu_tensor = gpu_tensor.to(Device::CPU);
    auto gpu_to_cpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_to_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_to_cpu_end - gpu_to_cpu_start).count();
    
    // Verify shape and values match original
    assert(cpu_tensor.shape() == original.shape());
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            assert(cpu_tensor(i, j) == original(i, j));
        }
    }
    
    // Result
    std::cout << "CPU → GPU transfer: " << cpu_to_gpu_us << " us\n";
    std::cout << "GPU → CPU transfer: " << gpu_to_cpu_us << " us\n";
    std::cout << "Total roundtrip:    " << (cpu_to_gpu_us + gpu_to_cpu_us) << " us\n\n";
}

void benchmark_bmm_vs_bmm_cuda(size_t B, size_t M, size_t K, size_t N) {
    std::cout << "Benchmarking batched matmul CPU vs GPU with shapes: ("
              << B << ", " << M << ", " << K << ") * ("
              << B << ", " << K << ", " << N << ")\n";

    // Create CPU tensors once and reuse
    Tensor3D A_cpu = Tensor3D::from_random(B, M, K);
    Tensor3D B_cpu = Tensor3D::from_random(B, K, N);

    // CPU baseline (mat_mul_eigen_parallel)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    Tensor3D C_cpu = A_cpu.mat_mul_eigen_parallel(B_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // Create GPU tensors
    Tensor3D A_gpu = A_cpu.to(Device::GPU);
    Tensor3D B_gpu = B_cpu.to(Device::GPU);

    // GPU batched matmul
    auto gpu_start = std::chrono::high_resolution_clock::now();
    Tensor3D C_gpu = bmm_cuda(A_gpu, B_gpu);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // Result
    double speedup = cpu_ms / gpu_ms;
    std::cout << "CPU time (mat_mul_eigen_parallel): " << cpu_ms << " ms\n";
    std::cout << "GPU time (bmm_cuda):              " << gpu_ms << " ms\n";
    std::cout << "Speedup:                           " << speedup << "x\n\n";
}

#endif

int main() {
    benchmark_matmul(16, 16, 16);
    benchmark_matmul(128, 128, 128);
    benchmark_matmul(256, 512, 128);
    benchmark_matmul(512, 512, 512);
    benchmark_matmul(1024, 1024, 1024);
    std::cout << std::endl;

    benchmark_batched_matmul(8, 16, 16, 16);
    benchmark_batched_matmul(16, 64, 64, 64);
    benchmark_batched_matmul(32, 128, 128, 128);
    benchmark_batched_matmul(8, 256, 512, 128);
    benchmark_batched_matmul(4, 512, 512, 512);
    benchmark_batched_matmul(2, 1024, 1024, 1024);

    benchmark_batched_matmul_parallel(8, 16, 16, 16);
    benchmark_batched_matmul_parallel(16, 64, 64, 64);
    benchmark_batched_matmul_parallel(32, 128, 128, 128);
    benchmark_batched_matmul_parallel(8, 256, 512, 128);
    benchmark_batched_matmul_parallel(4, 512, 512, 512);
    benchmark_batched_matmul_parallel(2, 1024, 1024, 1024);
    std::cout << std::endl;

    #ifdef USE_CUDA
    benchmark_matmul_cuda_vs_cpu(512);
    benchmark_matmul_cuda_vs_cpu(1024);
    benchmark_device_transfer(512);
    benchmark_device_transfer(1024);
    benchmark_bmm_vs_bmm_cuda(8, 16, 16, 16);
    benchmark_bmm_vs_bmm_cuda(16, 64, 64, 64);
    benchmark_bmm_vs_bmm_cuda(32, 128, 128, 128);
    benchmark_bmm_vs_bmm_cuda(8, 256, 512, 128);
    benchmark_bmm_vs_bmm_cuda(4, 512, 512, 512);
    benchmark_bmm_vs_bmm_cuda(2, 1024, 1024, 1024);
    std::cout << std::endl;
    #endif
}
