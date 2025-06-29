#pragma once
#include "tensor2d.hpp"

class Module {
public:
    virtual Tensor2D forward(const Tensor2D& input) = 0;
    virtual ~Module() = default;
};
