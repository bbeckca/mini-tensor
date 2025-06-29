#include "sequential.hpp"

void Sequential::add(std::unique_ptr<Module> layer) {
    layers.push_back(std::move(layer));
}

Tensor2D Sequential::forward(const Tensor2D& input) {
    Tensor2D x = input;
    for (auto& layer : layers) {
        x = layer->forward(x);
    }
    return x;
}