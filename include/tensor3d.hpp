#pragma once
#include "tensor2d.hpp"
#include <vector>
#include <Eigen/Core>

class Tensor3D {
private:
    float* data_;
    size_t B_, M_, N_;
    Device device_;

public:
    Tensor3D(size_t B, size_t M, size_t N, float val = 0.0f, Device device = Device::CPU)
        : B_(B), M_(M), N_(N), device_(device) {
        size_t size = B * M * N;

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
                    throw std::runtime_error("CUDA memset failed: " + std::string(cudaGetErrorString(err)));
                }
            } else {
                cudaMemset(data_, 0, size * sizeof(float));
            }
        #else
            throw std::runtime_error("CUDA support not enabled — recompile with -DUSE_CUDA");
        #endif
        }
    }

    ~Tensor3D() {
        if (device_ == Device::CPU) {
            delete[] data_;
        } else {
        #ifdef USE_CUDA
            cudaFree(data_);
        #else
            std::cerr << "Warning: CUDA support not enabled, cannot free GPU memory in Tensor3D destructor." << std::endl;
        #endif
        }
    }

    Tensor3D(const Tensor3D&) = delete;
    Tensor3D& operator=(const Tensor3D&) = delete;
    Tensor3D(Tensor3D&&) = default;
    Tensor3D& operator=(Tensor3D&&) = default;


    size_t batch_size() const { return B_; }
    size_t rows() const { return M_; }
    size_t cols() const { return N_; }

    Tensor2D operator[](size_t index) const {
        return slice_batch(index);  // returns a view (non-owning)
    }

    Tensor3D mat_mul(const Tensor3D& other) {
        if (this->cols() != other.rows()) {
            throw std::invalid_argument("Shape mismatch in mat_mul");
        }
        Tensor3D result(this->batch_size(), this->rows(), other.cols());
        for (size_t i = 0; i < this->batch_size(); ++i) {
            Tensor2D A = (*this)[i];
            Tensor2D B = other[i];
            Tensor2D C = A.mat_mul(B);
            std::memcpy(result.data_ + i * C.rows() * C.cols(), C.data(), C.rows() * C.cols() * sizeof(float));
        }
        return result;
    }

    Tensor3D mat_mul_eigen(const Tensor3D& other) {
        if (this->cols() != other.rows()) {
            throw std::invalid_argument("Shape mismatch in mat_mul_eigen");
        }
        Tensor3D result(this->batch_size(), this->rows(), other.cols());
        for (size_t i = 0; i < this->batch_size(); ++i) {
            Tensor2D A = (*this)[i];
            Tensor2D B = other[i];
            Tensor2D C = A.mat_mul_eigen(B);
            std::memcpy(result.data_ + i * C.rows() * C.cols(), C.data(), C.rows() * C.cols() * sizeof(float));
        }   
        return result;
    }

    Tensor3D mat_mul_eigen_parallel(const Tensor3D& other) {
        if (this->cols() != other.rows()) {
            throw std::invalid_argument("Shape mismatch in mat_mul_eigen");
        }
        Eigen::setNbThreads(1);
        Tensor3D result(this->batch_size(), this->rows(), other.cols());
        #pragma omp parallel for
        for (size_t i = 0; i < this->batch_size(); ++i) {
            Tensor2D A = (*this)[i];
            Tensor2D B = other[i];
            Tensor2D C = A.mat_mul_eigen(B);
            std::memcpy(result.data_ + i * C.rows() * C.cols(), C.data(), C.rows() * C.cols() * sizeof(float));
        }
        return result;
    }

    static Tensor3D from_random(size_t batch, size_t rows, size_t cols) {
        Tensor3D result(batch, rows, cols);
        for (size_t i = 0; i < batch; ++i) {
            std::memcpy(result.data_ + i * rows * cols, result[i].data(), rows * cols * sizeof(float));
        }
        return result;
    }

    Tensor2D slice_batch(size_t b) const {
        if (device_ != Device::CPU)
            throw std::runtime_error("slice_batch only supports CPU for now");

        if (b >= B_)
            throw std::out_of_range("Batch index out of bounds");

        float* ptr = data_ + b * M_ * N_;
        return Tensor2D(M_, N_, ptr, Device::CPU, /*owns_data=*/false);
    }

    void set_batch(size_t b, const Tensor2D& t) {
        if (device_ != Device::CPU)
            throw std::runtime_error("set_batch only supports CPU for now");

        if (b >= B_)
            throw std::out_of_range("Batch index out of bounds");

        if (t.rows() != M_ || t.cols() != N_)
            throw std::invalid_argument("Tensor2D shape mismatch in set_batch");

        float* ptr = data_ + b * M_ * N_;
        std::memcpy(ptr, t.data(), M_ * N_ * sizeof(float));
    }

};
