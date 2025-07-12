#pragma once

#include "tensor2d.hpp"
#include "tensor3d.hpp"

#ifdef USE_CUDA
Tensor2D add_cuda(const Tensor2D& A, const Tensor2D& B);
Tensor3D add_cuda(const Tensor3D& A, const Tensor3D& B);
Tensor2D mat_mul_cuda(const Tensor2D& A, const Tensor2D& B);
Tensor3D bmm_cuda(const Tensor3D& A, const Tensor3D& B);

#endif