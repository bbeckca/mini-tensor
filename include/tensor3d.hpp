#pragma once
#include "tensor2d.hpp"
#include <vector>
#include <Eigen/Core>

class Tensor3D {
private:
    float* data_;
    size_t B_, M_, N_;
    Device device_;
    std::string id_;
    bool owns_data_;

public:
    Tensor3D(size_t B, size_t M, size_t N, float val = 0.0f, Device device = Device::CPU)
        : B_(B), M_(M), N_(N), device_(device), id_(TensorID::generate()), owns_data_(true) {
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
        if (!owns_data_) return;

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

    Tensor3D(size_t B, size_t M, size_t N, float* external_data, Device device, bool owns_data)
        : B_(B), M_(M), N_(N), data_(external_data), device_(device), id_(TensorID::generate()), owns_data_(owns_data) {}

    Tensor3D(const Tensor3D& other)
        : B_(other.B_), M_(other.M_), N_(other.N_), device_(other.device_), id_(TensorID::generate()), data_(nullptr), owns_data_(true) {
        size_t size = B_ * M_ * N_;
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
            #else
            throw std::runtime_error("CUDA support not enabled");
            #endif
        }
    }

    Tensor3D& operator=(const Tensor3D& other) {
        if (this != &other) {
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
            
            B_ = other.B_;
            M_ = other.M_;
            N_ = other.N_;
            device_ = other.device_;
            id_ = TensorID::generate();
            owns_data_ = true;
            
            size_t size = B_ * M_ * N_;
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
                #else
                throw std::runtime_error("CUDA support not enabled");
                #endif
            }
        }
        return *this;
    }

    size_t batch_size() const { return B_; }
    size_t rows() const { return M_; }
    size_t cols() const { return N_; }
    float* data() { return data_; }
    const float* data() const { return data_; }
    Device get_device() const { return device_; }
    const std::string& get_id() const { return id_; }
    bool is_view() const { return !owns_data_; }
    std::tuple<size_t, size_t, size_t> shape() const { return std::make_tuple(B_, M_, N_); }

    Tensor2D operator[](size_t index) const {
        return slice_batch(index);  // returns a view (non-owning)
    }

    Tensor3D operator+(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        Tensor3D result(B, M, N);
        for_each_broadcasted_3d(*this, other, result, [](float a, float b) { return a + b; });
        return result;
    }

    Tensor3D operator+(const Tensor2D& other) {
        Tensor3D result(B_, M_, N_);
        for_each_broadcasted_3d_2d(*this, other, result, [](float a, float b) { return a + b; });
        return result;
    }

    Tensor3D& operator+=(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        for_each_broadcasted_3d(*this, other, *this, [](float a, float b) { return a + b; });
        return *this;
    }

    Tensor3D operator-(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        Tensor3D result(B, M, N);
        for_each_broadcasted_3d(*this, other, result, [](float a, float b) { return a - b; });
        return result;
    }

    Tensor3D operator-(const Tensor2D& other) {
        Tensor3D result(B_, M_, N_);
        for_each_broadcasted_3d_2d(*this, other, result, [](float a, float b) { return a - b; });
        return result;
    }

    Tensor3D& operator-=(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        for_each_broadcasted_3d(*this, other, *this, [](float a, float b) { return a - b; });
        return *this;
    }

    Tensor3D operator*(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        Tensor3D result(B, M, N);
        for_each_broadcasted_3d(*this, other, result, [](float a, float b) { return a * b; });
        return result;
    }

    Tensor3D operator*(const Tensor2D& other) {
        Tensor3D result(B_, M_, N_);
        for_each_broadcasted_3d_2d(*this, other, result, [](float a, float b) { return a * b; });
        return result;
    }

    Tensor3D& operator*=(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        for_each_broadcasted_3d(*this, other, *this, [](float a, float b) { return a * b; });
        return *this;
    }

    Tensor3D operator/(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        Tensor3D result(B, M, N);
        for_each_broadcasted_3d(*this, other, result, [](float a, float b) { return a / b; });
        return result;
    }

    Tensor3D operator/(const Tensor2D& other) {
        Tensor3D result(B_, M_, N_);
        for_each_broadcasted_3d_2d(*this, other, result, [](float a, float b) { return a / b; });
        return result;
    }

    Tensor3D& operator/=(const Tensor3D& other) {
        auto [B, M, N] = infer_broadcast_shape_3d(this->shape(), other.shape());
        for_each_broadcasted_3d(*this, other, *this, [](float a, float b) { return a / b; });
        return *this;
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

    static Tensor3D from_vector(size_t batch, size_t rows, size_t cols, const std::vector<float>& data, Device device = Device::CPU) {
        if (data.size() != batch * rows * cols) {
            throw std::invalid_argument("Data size must match the number of elements");
        }

        Tensor3D result(batch, rows, cols, 0.0f, device);
        size_t bytes = batch * rows * cols * sizeof(float);

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

    #ifdef USE_CUDA
    Tensor3D to(Device target) const {
        if (device_ == target) return *this;

        Tensor3D result(B_, M_, N_, 0.0f, target);
        size_t bytes = B_ * M_ * N_ * sizeof(float);
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

    float& operator()(size_t b, size_t i, size_t j) {
        if (device_ != Device::CPU)
            throw std::runtime_error("operator() only supports CPU tensors");
        if (b >= B_ || i >= M_ || j >= N_)
            throw std::out_of_range("Tensor3D index out of bounds");
        return data_[b * M_ * N_ + i * N_ + j];
    }

    const float& operator()(size_t b, size_t i, size_t j) const {
        if (device_ != Device::CPU)
            throw std::runtime_error("operator() only supports CPU tensors");
        if (b >= B_ || i >= M_ || j >= N_)
            throw std::out_of_range("Tensor3D index out of bounds");
        return data_[b * M_ * N_ + i * N_ + j];
    }



    static std::tuple<size_t, size_t, size_t> infer_broadcast_shape_3d(
        const std::tuple<size_t, size_t, size_t>& shape1,
        const std::tuple<size_t, size_t, size_t>& shape2) {

        auto broadcast_dim = [](size_t a, size_t b) -> size_t {
            if (a == b) return a;
            if (a == 1) return b;
            if (b == 1) return a;
            throw std::invalid_argument("Cannot broadcast: incompatible shapes");
        };

        return {
            broadcast_dim(std::get<0>(shape1), std::get<0>(shape2)),  // B
            broadcast_dim(std::get<1>(shape1), std::get<1>(shape2)),  // M
            broadcast_dim(std::get<2>(shape1), std::get<2>(shape2))   // N
        };
    }


    template <typename F>
    static void for_each_broadcasted_3d(
        const Tensor3D& A,
        const Tensor3D& B,
        Tensor3D& out,
        F op) {

        auto [B1, M1, N1] = A.shape();
        auto [B2, M2, N2] = B.shape();
        auto [BO, MO, NO] = out.shape();

        if (BO != B1 && BO != B2) throw std::invalid_argument("Batch size mismatch in broadcasting");
        if (MO != M1 && MO != M2) throw std::invalid_argument("Row mismatch in broadcasting");
        if (NO != N1 && NO != N2) throw std::invalid_argument("Column mismatch in broadcasting");

        for (size_t b = 0; b < BO; ++b) {
            for (size_t i = 0; i < MO; ++i) {
                for (size_t j = 0; j < NO; ++j) {
                    float a = A(b % B1, i % M1, j % N1);
                    float b_val = B(b % B2, i % M2, j % N2);
                    out(b, i, j) = op(a, b_val);
                }
            }
        }
    }

    template <typename F>
    static void for_each_broadcasted_3d_2d(
        const Tensor3D& A,
        const Tensor2D& B,
        Tensor3D& out,
        F op) {

        auto [B1, M1, N1] = A.shape();
        auto [M2, N2] = B.shape();
        auto [BO, MO, NO] = out.shape();

        if (BO != B1 || MO != M1 || NO != N1) {
            throw std::invalid_argument("Output tensor shape does not match broadcasted shape of Tensor3D input.");
        }

        if ((M2 != 1 && M2 != M1) || (N2 != 1 && N2 != N1)) {
            throw std::invalid_argument(
                "Tensor2D shape is not broadcast-compatible with Tensor3D. "
                "Expected (1, N), (M, 1), (M, N), or (1, 1). Got (" +
                std::to_string(M2) + ", " + std::to_string(N2) + ")");
        }

        for (size_t b = 0; b < B1; ++b) {
            for (size_t i = 0; i < M1; ++i) {
                for (size_t j = 0; j < N1; ++j) {
                    float a = A(b, i, j);
                    float b_val = B(i % M2, j % N2);
                    out(b, i, j) = op(a, b_val);
                }
            }
        }
    }
};
