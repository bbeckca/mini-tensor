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

    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }

    Tensor2D operator+(const Tensor2D& other) {
        auto [output_rows, output_cols] = infer_broadcast_shape(shape(), other.shape());

        Tensor2D result = Tensor2D(output_rows, output_cols);

        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                float a_val = (*this)(i % this->rows_, j % this->cols_);
                float b_val = other(i % other.rows_, j % other.cols_);
                result(i, j) = a_val + b_val;
            }
        }
        return result;
    }

    Tensor2D& operator+=(const Tensor2D& other) {
        if ((rows_ != other.rows_ && other.rows_ != 1) ||
            (cols_ != other.cols_ && other.cols_ != 1)) {
            throw std::invalid_argument("Shape mismatch in operator+=");
        }
        for (size_t i = 0; i < rows_; ++i) {
            size_t i_other = (other.rows_ == 1) ? 0 : i;
            for (size_t j = 0; j < cols_; ++j) {
                size_t j_other = (other.cols_ == 1) ? 0 : j;
                (*this)(i, j) += other(i_other, j_other);
            }
        }
        return *this;
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


    void reshape(size_t rows, size_t cols) {
        if (rows * cols != rows_ * cols_) {
            throw std::invalid_argument("New shape must match the total number of elements");
        }
        rows_ = rows;
        cols_ = cols;
    }

    std::pair<size_t, size_t> infer_broadcast_shape(std::pair<size_t, size_t> shape1, std::pair<size_t, size_t> shape2) const {
        if (shape1.first != shape2.first && shape1.first != 1 && shape2.first != 1) {
            throw std::invalid_argument("Shape mismatch in infer_broadcast_shape for rows");
        }
        if (shape1.second != shape2.second && shape1.second != 1 && shape2.second != 1) {
            throw std::invalid_argument("Shape mismatch in infer_broadcast_shape for cols");
        }
        return {std::max(shape1.first, shape2.first), std::max(shape1.second, shape2.second)};
    }

private:
    size_t rows_, cols_;
    std::vector<float> data_;
};
