#pragma once
#include "tensor2d.hpp"
#include "tensor3d.hpp"

class Module {
public:
    virtual Tensor2D forward(const Tensor2D& input) = 0;
    virtual Tensor3D forward(const Tensor3D& input) = 0;
    virtual ~Module() = default;
};
