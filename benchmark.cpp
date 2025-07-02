#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include "../include/tensor2d.hpp"
#include "../include/tensor3d.hpp"

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

    std::cout << "Speedup                : " << static_cast<float>(eigen_us) / eigen_us_parallel << "x\n\n";
}

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
}
