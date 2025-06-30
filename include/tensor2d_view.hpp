#pragma once
#include "tensor2d.hpp"

class Tensor2DView {

private:
    Tensor2D& tensor_;
    size_t row_start;
    size_t row_end;
    size_t col_start;
    size_t col_end;

public:
    Tensor2DView(Tensor2D& tensor, size_t row_start, size_t row_end, size_t col_start, size_t col_end)
        : tensor_(tensor), row_start(row_start), row_end(row_end), col_start(col_start), col_end(col_end) {}

    size_t rows() const { return row_end - row_start; }
    size_t cols() const { return col_end - col_start; }

    std::pair<size_t, size_t> shape() const {
        return {rows(), cols()};
    }

    float& operator()(size_t row, size_t col) {
        if (row >= rows() || col >= cols()) {
            throw std::out_of_range("Index out of bounds");
        }
        return tensor_(row_start + row, col_start + col);
    }

    const float& operator()(size_t row, size_t col) const {
        if (row >= rows() || col >= cols()) {
            throw std::out_of_range("Index out of bounds");
        }
        return tensor_(row_start + row, col_start + col);
    }

    bool is_empty() const {
        return rows() == 0 || cols() == 0;
    }

    void print() const {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "\n";
        }
    }

};