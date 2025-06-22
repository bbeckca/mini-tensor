#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>

class Tensor2D {
public:
    Tensor2D(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

    float& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[row * cols_ + col];
    }

    const float& operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[row * cols_ + col];
    }

    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    void print() const {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

    std::pair<size_t, size_t> shape() const {
        return {rows_, cols_};
    }

private:
    size_t rows_, cols_;
    std::vector<float> data_;
};
