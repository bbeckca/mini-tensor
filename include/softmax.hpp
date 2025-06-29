#pragma once
#include "module.hpp"

class Softmax : public Module {
public:
    Tensor2D forward(const Tensor2D& input) override;
};