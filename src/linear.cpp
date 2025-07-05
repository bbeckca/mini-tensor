#include "linear.hpp"
#include "ir_trace.hpp"
#include <stdexcept>

Linear::Linear(size_t in_dim, size_t out_dim)
    : weights(in_dim, out_dim, 0.1f), bias(1, out_dim, 0.0f) {}

void Linear::set_weights(const Tensor2D& new_weights) {
    if (new_weights.shape() != weights.shape()) {
        throw std::invalid_argument("Shape mismatch in set_weights");
    }
    weights.copy_from(new_weights);
}


void Linear::set_bias(const Tensor2D& new_bias) {
    if (new_bias.shape() != bias.shape()) {
        throw std::invalid_argument("Shape mismatch in set_bias");
    }
    bias.copy_from(new_bias);
}

Tensor2D Linear::get_weights() const {
    return weights;
}

Tensor2D Linear::get_bias() const {
    return bias;
}

Tensor2D Linear::forward(const Tensor2D& input) {
    Tensor2D result = input.mat_mul(weights) + bias;
    IRTrace::record("linear", {input.get_id(), weights.get_id(), bias.get_id()}, result.get_id(), result.shape(), result.get_device());
    return result;
}
