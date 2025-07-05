#pragma once

#include "tensor2d.hpp"

#ifdef USE_CUDA
Tensor2D mat_mul_cuda(const Tensor2D& A, const Tensor2D& B);
#endif