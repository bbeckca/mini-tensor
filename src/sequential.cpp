#include <memory>
#include "sequential.hpp"
#include "ir_trace.hpp"

void Sequential::add(std::unique_ptr<Module> layer) {
    layers.push_back(std::move(layer));
}

Tensor2D Sequential::forward(const Tensor2D& input) {
    Tensor2D x = input;
    std::vector<std::string> inputs;
    for (auto& layer : layers) {
        inputs.push_back(x.get_id());
        x = layer->forward(x);
        
    }
    IRTrace::record("sequential", inputs, x.get_id(), x.shape(), x.get_device());
    return x;
}

Tensor3D Sequential::forward(const Tensor3D& input) {
    Tensor3D x = input;
    std::vector<std::string> inputs;
    for (auto& layer : layers) {
        inputs.push_back(x.get_id());
        x = layer->forward(x);
    }
    IRTrace::record("sequential", inputs, x.get_id(), x.shape(), x.get_device());
    return x;
}