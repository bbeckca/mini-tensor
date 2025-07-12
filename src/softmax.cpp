#include "softmax.hpp"
#include "ir_trace.hpp"
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

    IRTrace::record("softmax", {input.get_id()}, output.get_id(), output.shape(), output.get_device());
    return output;
}

Tensor3D Softmax::forward(const Tensor3D& input) {
    auto [B, M, N] = input.shape();
    Tensor3D output = Tensor3D(B, M, N);

    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < M; ++i) {
            float max_val = input(b, i, 0);
            for (size_t j = 1; j < N; ++j) {
                max_val = std::max(input(b, i, j), max_val);
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < N; ++j) {
                float exp_val = std::exp(input(b, i, j) - max_val);
                output(b, i, j) = exp_val;
                sum_exp += exp_val;
            }

            for (size_t j = 0; j < N; ++j) {
                output(b, i, j) /= sum_exp;
            }
        }
    }
    IRTrace::record("softmax", {input.get_id()}, output.get_id(), output.shape(), output.get_device());
    return output;
}