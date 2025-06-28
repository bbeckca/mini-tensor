#include <iostream>
#include <chrono>
#include "../include/tensor2d.hpp"

int main() {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};

    for (size_t dim : sizes) {
        Tensor2D A(dim, dim, 1.0f);
        Tensor2D B(dim, dim, 2.0f);

        auto start = std::chrono::high_resolution_clock::now();
        Tensor2D C = A.mat_mul(B);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "mat_mul(" << dim << "x" << dim << ") took "
                  << duration.count() << " ms" << std::endl;
    }

    return 0;
}
