#include "tensor2d.hpp"

int main() {
    Tensor2D t(2, 3);       // 2 rows × 3 columns
    t.fill(1.0f);
    t(0, 2) = 5.5f;
    t.print();

    auto [rows, cols] = t.shape();
    std::cout << "Shape: " << rows << " x " << cols << "\n";

    return 0;
}
