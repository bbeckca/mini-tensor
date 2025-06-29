#include "sequential.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "tensor2d.hpp"

int main() {
    // Define model
    Sequential model;
    auto linear = std::make_unique<Linear>(4, 2);
    linear->set_weights(Tensor2D::from_vector(4, 2, {
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
        0.7, 0.8
    }));
    linear->set_bias(Tensor2D::from_vector(1, 2, {0.5, 0.5}));
    model.add(std::move(linear));
    model.add(std::make_unique<ReLU>());

    // Run forward pass
    Tensor2D input = Tensor2D::from_vector(1, 4, {1, 2, 3, 4});
    Tensor2D output = model.forward(input);

    // Print result
    std::cout << "Output:" << std::endl;
    output.print();
    return 0;
}
