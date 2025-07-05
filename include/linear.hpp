#pragma once
#include "module.hpp"

class Linear : public Module {
private:
    Tensor2D weights;
    Tensor2D bias;

public:
    Linear(size_t in_dim, size_t out_dim);
    void set_weights(const Tensor2D& w);
    void set_bias(const Tensor2D& b);
    Tensor2D get_weights() const;
    Tensor2D get_bias() const;
    Tensor2D forward(const Tensor2D& input) override;
};
