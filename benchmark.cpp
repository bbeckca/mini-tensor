#include <iostream>
#include <chrono>
#include "../include/tensor2d.hpp"

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

int main() {
    benchmark_matmul(16, 16, 16);
    benchmark_matmul(128, 128, 128);
    benchmark_matmul(256, 512, 128);
    benchmark_matmul(512, 512, 512);
    benchmark_matmul(1024, 1024, 1024);
}
