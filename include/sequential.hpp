#pragma once
#include <memory>
#include <vector>
#include "module.hpp"

class Sequential : public Module {
private:
    std::vector<std::unique_ptr<Module>> layers;

public:
    void add(std::unique_ptr<Module> layer);
    Tensor2D forward(const Tensor2D& input) override;
    Tensor3D forward(const Tensor3D& input) override;
};