#pragma once
#include "tensor2d.hpp"
#include <vector>

class Tensor3D {
private:
    std::vector<Tensor2D> batches_;

public:
    Tensor3D(size_t batch, size_t rows, size_t cols, float val = 0.0f) {
        for (size_t i = 0; i < batch; ++i) {
            batches_.emplace_back(rows, cols, val);
        }
    }

    Tensor2D& operator[](size_t i) { return batches_[i]; }
    const Tensor2D& operator[](size_t i) const { return batches_[i]; }
    size_t batch_size() const { return batches_.size(); }
    size_t rows() const { return batches_.empty() ? 0 : batches_[0].rows(); }
    size_t cols() const { return batches_.empty() ? 0 : batches_[0].cols(); }
};
