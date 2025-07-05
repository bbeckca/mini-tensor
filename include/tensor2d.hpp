#pragma once
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include "id_utils.hpp"
#include "ir_trace.hpp"
#include "device.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class Tensor2D {

private:
    size_t rows_, cols_;
    float* data_;
    std::string id_;
    Device device_;

public:
    Tensor2D(size_t rows, size_t cols, float val = 0.0f, Device device = Device::CPU)
        : rows_(rows), cols_(cols), device_(device), id_(TensorID::generate()) {
        size_t size = rows * cols;
        if (device == Device::CPU) {
            data_ = new float[size];
            std::fill(data_, data_ + size, val);
        } else {
            #ifdef USE_CUDA
            cudaError_t err = cudaMalloc(&data_, size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
            }
            if (val != 0.0f) {
                float* tmp = new float[size];
                std::fill(tmp, tmp + size, val);
                err = cudaMemcpy(data_, tmp, size * sizeof(float), cudaMemcpyHostToDevice);
                delete[] tmp;
                if (err != cudaSuccess) {
                    cudaFree(data_);
                    throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
                }
            }
            #else
            throw std::runtime_error("CUDA support not enabled — recompile with -DUSE_CUDA");
            #endif
        }
    }

    Tensor2D(const Tensor2D& other)
        : rows_(other.rows_), cols_(other.cols_), device_(other.device_), id_(TensorID::generate()), data_(nullptr) {
        size_t size = rows_ * cols_;
        if (device_ == Device::CPU) {
            data_ = new float[size];
            std::copy(other.data_, other.data_ + size, data_);
        } else {
            #ifdef USE_CUDA
            cudaError_t err = cudaMalloc(&data_, size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed in copy constructor: " + std::string(cudaGetErrorString(err)));
            }
            err = cudaMemcpy(data_, other.data_, size * sizeof(float), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(data_);
                throw std::runtime_error("CUDA memcpy failed in copy constructor: " + std::string(cudaGetErrorString(err)));
            }
            #endif
        }
    }

    Tensor2D& operator=(const Tensor2D& other) {
        if (this != &other) {
            if (data_) {
                if (device_ == Device::CPU) {
                    delete[] data_;
                            } else {
                #ifdef USE_CUDA
                cudaError_t err = cudaFree(data_);
                if (err != cudaSuccess) {
                    std::cerr << "Warning: CUDA free failed in assignment: " << cudaGetErrorString(err) << std::endl;
                }
                #endif
            }
        }
        rows_ = other.rows_;
        cols_ = other.cols_;
        device_ = other.device_;
        id_ = TensorID::generate();
        size_t size = rows_ * cols_;
        if (device_ == Device::CPU) {
            data_ = new float[size];
            std::copy(other.data_, other.data_ + size, data_);
        } else {
            #ifdef USE_CUDA
            cudaError_t err = cudaMalloc(&data_, size * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA malloc failed in assignment: " + std::string(cudaGetErrorString(err)));
            }
            err = cudaMemcpy(data_, other.data_, size * sizeof(float), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(data_);
                throw std::runtime_error("CUDA memcpy failed in assignment: " + std::string(cudaGetErrorString(err)));
            }
            #endif
        }
        }
        return *this;
    }

    float* data() { return data_; }
    const float* data() const { return data_; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const std::string& get_id() const { return id_; }
    Device get_device() const { return device_; }

    float& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Index out of bounds");
        }
        if (device_ == Device::GPU) {
            throw std::runtime_error("Cannot access GPU tensor data directly from CPU code. Use to(Device::CPU) first.");
        }
        return data_[row * cols_ + col];
    }

    const float& operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Index out of bounds");
        }
        if (device_ == Device::GPU) {
            throw std::runtime_error("Cannot access GPU tensor data directly from CPU code. Use to(Device::CPU) first.");
        }
        return data_[row * cols_ + col];
    }

    float& operator[](size_t index) { 
        if (index >= rows_ * cols_) {
            throw std::out_of_range("Index out of bounds in operator[]");
        }
        if (device_ == Device::GPU) {
            throw std::runtime_error("Cannot access GPU tensor data directly from CPU code. Use to(Device::CPU) first.");
        }
        return data_[index]; 
    }
    const float& operator[](size_t index) const { 
        if (index >= rows_ * cols_) {
            throw std::out_of_range("Index out of bounds in operator[]");
        }
        if (device_ == Device::GPU) {
            throw std::runtime_error("Cannot access GPU tensor data directly from CPU code. Use to(Device::CPU) first.");
        }
        return data_[index]; 
    }


    #ifdef USE_CUDA
    Tensor2D to(Device target) const {
        if (device_ == target) return *this;

        Tensor2D result(rows_, cols_, 0.0f, target);
        size_t bytes = rows_ * cols_ * sizeof(float);
        cudaError_t err;
        if (target == Device::GPU) {
            err = cudaMemcpy(result.data_, data_, bytes, cudaMemcpyHostToDevice);
        } else {
            err = cudaMemcpy(result.data_, data_, bytes, cudaMemcpyDeviceToHost);
        }
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed in to(): " + std::string(cudaGetErrorString(err)));
        }
        return result;
    }
    #endif

    void fill(float val) {
        size_t size = rows_ * cols_;
        if (device_ == Device::CPU) {
            std::fill(data_, data_ + size, val);
        }
        else {
            #ifdef USE_CUDA
            float* tmp = new float[size];
            std::fill(tmp, tmp + size, val);
            cudaError_t err = cudaMemcpy(data_, tmp, size * sizeof(float), cudaMemcpyHostToDevice);
            delete[] tmp;
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed in fill(): " + std::string(cudaGetErrorString(err)));
            }
            #endif
        }
    }

    static Tensor2D from_vector(size_t rows, size_t cols, const std::vector<float>& data, Device device = Device::CPU) {
        if (data.size() != rows * cols) {
            throw std::invalid_argument("Data size must match the number of elements");
        }

        Tensor2D result(rows, cols, 0.0f, device);
        size_t bytes = rows * cols * sizeof(float);

        if (device == Device::CPU) {
            std::memcpy(result.data(), data.data(), bytes);
        } else {
            #ifdef USE_CUDA
            cudaError_t err = cudaMemcpy(result.data(), data.data(), bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed in from_vector: " + std::string(cudaGetErrorString(err)));
            }
            #else
            throw std::runtime_error("CUDA support not enabled — recompile with -DUSE_CUDA");
            #endif
        }

        return result;
    }

    static Tensor2D from_random(size_t rows, size_t cols, Device device=Device::CPU) {
        Tensor2D result(rows, cols, 0.0f, device);
        if (device == Device::CPU) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = std::rand() / (float)RAND_MAX;
                }
            }
        } else {
            #ifdef USE_CUDA
            // Generate random values on CPU first, then copy to GPU
            std::vector<float> cpu_data(rows * cols);
            for (size_t i = 0; i < rows * cols; ++i) {
                cpu_data[i] = std::rand() / (float)RAND_MAX;
            }
            cudaError_t err = cudaMemcpy(result.data(), cpu_data.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed in from_random: " + std::string(cudaGetErrorString(err)));
            }
            #endif
        }
        return result;
    }

    void copy_from(const Tensor2D& other) {
        if (this == &other) return;
        if (rows_ != other.rows_ || cols_ != other.cols_ || device_ != other.device_) {
            throw std::invalid_argument("copy_from shape or device mismatch");
        }

        size_t bytes = rows_ * cols_ * sizeof(float);
        if (device_ == Device::CPU) {
            std::memcpy(data_, other.data_, bytes);
        } else {
            #ifdef USE_CUDA
            cudaError_t err = cudaMemcpy(data_, other.data_, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed in copy_from: " + std::string(cudaGetErrorString(err)));
            }
            #else
            throw std::runtime_error("CUDA not enabled");
            #endif
        }
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
        IRTrace::record("reshape", {this->get_id()}, this->get_id(), {rows, cols}, device_);
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

    Tensor2D expand(size_t rows, size_t cols) const {
        auto [output_rows, output_cols] = infer_broadcast_shape(shape(), {rows, cols});
        Tensor2D result = Tensor2D(output_rows, output_cols);
        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                result(i, j) = (*this)(i % rows_, j % cols_);
            }
        }
        return result;
    }

    bool operator==(const Tensor2D& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        size_t size = rows_ * cols_;
        
        if (device_ == Device::CPU && other.device_ == Device::CPU) {
            for (size_t i = 0; i < size; ++i) {
                if (data_[i] != other.data_[i]) return false;
            }
        } else {
            #ifdef USE_CUDA
            // Copy both tensors to CPU for comparison
            std::vector<float> this_data(size);
            std::vector<float> other_data(size);
            
            if (device_ == Device::GPU) {
                cudaMemcpy(this_data.data(), data_, size * sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                std::copy(data_, data_ + size, this_data.begin());
            }
            
            if (other.device_ == Device::GPU) {
                cudaMemcpy(other_data.data(), other.data_, size * sizeof(float), cudaMemcpyDeviceToHost);
            } else {
                std::copy(other.data_, other.data_ + size, other_data.begin());
            }
            
            for (size_t i = 0; i < size; ++i) {
                if (this_data[i] != other_data[i]) return false;
            }
            #else
            throw std::runtime_error("CUDA support not enabled for GPU tensor comparison");
            #endif
        }
        return true;
    }

    bool operator!=(const Tensor2D& other) const {
        return !(*this == other);
    }

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
        IRTrace::record("operator+", {this->get_id(), other.get_id()}, result.get_id(), result.shape(), device_);
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

    Tensor2D operator+(float scalar) {
        Tensor2D result = Tensor2D(rows_, cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
        return result;
    }

    Tensor2D& operator+=(float scalar) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) += scalar;
            }
        }
        return *this;
    }    

    Tensor2D operator-(const Tensor2D& other) {
        auto [output_rows, output_cols] = infer_broadcast_shape(shape(), other.shape());

        Tensor2D result = Tensor2D(output_rows, output_cols);

        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                float a_val = (*this)(i % this->rows_, j % this->cols_);
                float b_val = other(i % other.rows_, j % other.cols_);
                result(i, j) = a_val - b_val;
            }
        }
        IRTrace::record("operator-", {this->get_id(), other.get_id()}, result.get_id(), result.shape(), device_);
        return result;
    }

    Tensor2D& operator-=(const Tensor2D& other) {
        if ((rows_ != other.rows_ && other.rows_ != 1) ||
            (cols_ != other.cols_ && other.cols_ != 1)) {
            throw std::invalid_argument("Shape mismatch in operator-=");
        }
        for (size_t i = 0; i < rows_; ++i) {
            size_t i_other = (other.rows_ == 1) ? 0 : i;
            for (size_t j = 0; j < cols_; ++j) {
                size_t j_other = (other.cols_ == 1) ? 0 : j;
                (*this)(i, j) -= other(i_other, j_other);
            }
        }
        return *this;
    }

    Tensor2D operator-(float scalar) {
        Tensor2D result = Tensor2D(rows_, cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) - scalar;
            }
        }
        return result;
    }

    Tensor2D& operator-=(float scalar) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) -= scalar;
            }
        }
        return *this;
    }    

    Tensor2D operator*(const Tensor2D& other) {
        auto [output_rows, output_cols] = infer_broadcast_shape(shape(), other.shape());

        Tensor2D result = Tensor2D(output_rows, output_cols);

        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                float a_val = (*this)(i % this->rows_, j % this->cols_);
                float b_val = other(i % other.rows_, j % other.cols_);
                result(i, j) = a_val * b_val;
            }
        }
        IRTrace::record("operator*", {this->get_id(), other.get_id()}, result.get_id(), result.shape(), device_);
        return result;
    }

    Tensor2D& operator*=(const Tensor2D& other) {
        if ((rows_ != other.rows_ && other.rows_ != 1) ||
            (cols_ != other.cols_ && other.cols_ != 1)) {
            throw std::invalid_argument("Shape mismatch in operator*=");
        }
        for (size_t i = 0; i < rows_; ++i) {
            size_t i_other = (other.rows_ == 1) ? 0 : i;
            for (size_t j = 0; j < cols_; ++j) {
                size_t j_other = (other.cols_ == 1) ? 0 : j;
                (*this)(i, j) *= other(i_other, j_other);
            }
        }
        return *this;
    }

    Tensor2D operator*(float scalar) {
        Tensor2D result = Tensor2D(rows_, cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    }

    Tensor2D& operator*=(float scalar) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) *= scalar;
            }
        }
        return *this;
    }

    Tensor2D operator/(const Tensor2D& other) {
        auto [output_rows, output_cols] = infer_broadcast_shape(shape(), other.shape());

        Tensor2D result = Tensor2D(output_rows, output_cols);

        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                float a_val = (*this)(i % this->rows_, j % this->cols_);
                float b_val = other(i % other.rows_, j % other.cols_);
                result(i, j) = a_val / b_val;
            }
        }
        IRTrace::record("operator/", {this->get_id(), other.get_id()}, result.get_id(), result.shape(), device_);
        return result;
    }

    Tensor2D& operator/=(const Tensor2D& other) {
        if ((rows_ != other.rows_ && other.rows_ != 1) ||
            (cols_ != other.cols_ && other.cols_ != 1)) {
            throw std::invalid_argument("Shape mismatch in operator/=");
        }
        for (size_t i = 0; i < rows_; ++i) {
            size_t i_other = (other.rows_ == 1) ? 0 : i;
            for (size_t j = 0; j < cols_; ++j) {
                size_t j_other = (other.cols_ == 1) ? 0 : j;
                (*this)(i, j) /= other(i_other, j_other);
            }
        }
        return *this;
    }

    Tensor2D operator/(float scalar) {
        Tensor2D result = Tensor2D(rows_, cols_);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
        return result;
    }

    Tensor2D& operator/=(float scalar) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) /= scalar;
            }
        }
        return *this;
    }    

    Tensor2D unary_op(std::function<float(float)> fn) const {
        Tensor2D result = Tensor2D(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result(i, j) = fn((*this)(i, j));
            }
        }
        return result;
    }

    void unary_op(std::function<float(float)> fn) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) = fn((*this)(i, j));
            }
        }
    }

    Tensor2D relu() const {
        Tensor2D result = unary_op([](float x) {return std::max(x, 0.0f); });
        IRTrace::record("relu", {this->get_id()}, result.get_id(), result.shape(), device_);
        return result;
    }

    void relu_in_place() {
        unary_op([](float x) { return std::max(x, 0.0f); });
    }

    Tensor2D negate() const {
        return unary_op([](float x) { return -x; });
    }

    void negate_in_place() {
        unary_op([](float x) { return -x; });
    }

    Tensor2D abs() const {
        return unary_op([](float x) { return std::abs(x); });
    }

    void abs_in_place() {
        unary_op([](float x) { return std::abs(x); });
    }

    float sum() const {
        float sum = 0.0f;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                sum += (*this)(i, j);
            }
        }
        return sum;
    }

    float mean() const {
        return sum() / (rows_ * cols_);
    }

    float max() const {
        float max = (*this)(0, 0);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if ((*this)(i, j) > max) {
                    max = (*this)(i, j);
                }
            }
        }
        return max;
    }

    Tensor2D mat_mul(const Tensor2D& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Shape mismatch in mat_mul");
        }

        size_t output_rows = rows_, output_cols = other.cols_;
        Tensor2D result = Tensor2D(output_rows, output_cols);

        for (size_t i = 0; i < output_rows; ++i) {
            for (size_t j = 0; j < output_cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        IRTrace::record("mat_mul", {this->get_id(), other.get_id()}, result.get_id(), result.shape(), device_);
        return result;
    }

    Tensor2D mat_mul_eigen(const Tensor2D& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Shape mismatch in mat_mul_eigen");
        }

        Tensor2D result(rows_, other.cols_);

        using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        Eigen::Map<const MatrixRM> A(this->data(), rows_, cols_);
        Eigen::Map<const MatrixRM> B(other.data(), other.rows_, other.cols_);
        Eigen::Map<MatrixRM> C(result.data(), result.rows_, result.cols_);

        C.noalias() = A * B;

        return result;
    }


    std::pair<size_t, size_t> arg_max() const {
        float max = (*this)(0, 0);
        std::pair<size_t, size_t> max_index = {0, 0};
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if ((*this)(i, j) > max) {
                    max = (*this)(i, j);
                    max_index = {i, j};
                }
            }
        }
        return max_index;
    }  

    ~Tensor2D() {
        if (data_) {
            if (device_ == Device::CPU) {
                delete[] data_;
            }
            else {
                #ifdef USE_CUDA
                cudaError_t err = cudaFree(data_);
                if (err != cudaSuccess) {
                    std::cerr << "Warning: CUDA free failed: " << cudaGetErrorString(err) << std::endl;
                }
                #endif
            }
        }
    }

};