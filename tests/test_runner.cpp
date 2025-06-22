#include "tensor2d.hpp"

int main() {
    Tensor2D t1(2, 3);       // 2 rows × 3 columns
    t1.fill(1.0f);
    // t1(0, 2) = 42.0f;
    t1.print();
    std::cout << std::endl;

    Tensor2D t2(2, 3);
    t2.fill(41.0f);
    // t2(1, 1) = 42.0f;
    t2.print();
    std::cout << std::endl;

    std::cout << "Adding tensors..." << std::endl;
    auto t3 = t1 + t2;
    t3.print();
    std::cout << std::endl;

    auto [rows, cols] = t3.shape();
    std::cout << "Shape of tensor: " << rows << " x " << cols << "\n";
    

    return 0;
}
