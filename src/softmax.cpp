#include "softmax.hpp"
#include <cmath>

Tensor2D Softmax::forward(const Tensor2D& input) {
    auto [rows, cols] = input.shape();
    Tensor2D output = Tensor2D(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        float max_val = input(i, 0);
        for (size_t j = 1; j < cols; ++j) {
            max_val = std::max(input(i, j), max_val);
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float exp_val = std::exp(input(i, j) - max_val);
            output(i, j) = exp_val;
            sum_exp += exp_val;
        }

        for (size_t j = 0; j < cols; ++j) {
            output(i, j) /= sum_exp;
        }
        
    }

    return output;
}