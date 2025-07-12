#include "relu.hpp"

Tensor2D ReLU::forward(const Tensor2D& input) {
    return input.relu();
}

Tensor3D ReLU::forward(const Tensor3D& input) {
    return input.relu();
}