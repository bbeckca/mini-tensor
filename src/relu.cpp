#include "relu.hpp"

Tensor2D ReLU::forward(const Tensor2D& input) {
    return input.relu();
}