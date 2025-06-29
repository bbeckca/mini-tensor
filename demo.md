# Tensor2D Demo and Feature Documentation
_Last updated: June 28, 2025_

## Table of Contents
- [Constructors](#constructors)
- [Basic Operations](#basic-operations)
- [Arithmetic Operations](#arithmetic-operations)
- [Matrix Operations](#matrix-operations)
- [Neural Network Modules](#neural-network-modules)
- [Reduction Operations](#reduction-operations)
- [Element-wise Functions](#element-wise-functions)
- [Shape and Reshaping](#shape-and-reshaping)
- [Broadcasting](#broadcasting)
- [Comparison Operations](#comparison-operations)
- [Building and Running](#building-and-running)

## Constructors

### Basic Constructor
```cpp
// Create a 3x4 tensor filled with zeros
Tensor2D tensor(3, 4);

// Create a 2x2 tensor filled with a specific value
Tensor2D tensor(2, 2, 42.0f);
```

### From Vector
```cpp
// Create tensor from existing data
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
Tensor2D tensor = Tensor2D::from_vector(2, 2, data);
```

## Basic Operations

### Element Access
```cpp
// Access elements using (row, col) syntax
float value = tensor(0, 1);
tensor(1, 2) = 5.5f;

// Access elements using linear indexing
float value = tensor[3];
tensor[5] = 10.0f;
```

### Fill Operations
```cpp
// Fill entire tensor with a value
tensor.fill(42.0f);
```

### Printing
```cpp
// Print tensor contents
tensor.print();
```

## Arithmetic Operations

### Addition
```cpp
// Tensor + Tensor (with broadcasting)
Tensor2D a(2, 2, 1.0f);
Tensor2D b(2, 2, 2.0f);
Tensor2D result = a + b;  // result = 3.0f everywhere

// Tensor + Scalar
Tensor2D result = a + 5.0f;  // adds 5.0f to every element

// In-place addition
a += b;
a += 3.0f;
```

### Subtraction
```cpp
// Tensor - Tensor (with broadcasting)
Tensor2D result = a - b;

// Tensor - Scalar
Tensor2D result = a - 2.0f;

// In-place subtraction
a -= b;
a -= 1.0f;
```

### Multiplication
```cpp
// Element-wise multiplication (with broadcasting)
Tensor2D result = a * b;

// Tensor * Scalar
Tensor2D result = a * 3.0f;

// In-place multiplication
a *= b;
a *= 2.0f;
```

### Division
```cpp
// Element-wise division (with broadcasting)
Tensor2D result = a / b;

// Tensor / Scalar
Tensor2D result = a / 2.0f;

// In-place division
a /= b;
a /= 4.0f;
```

## Matrix Operations

### Matrix Multiplication
```cpp
// Matrix multiplication (not element-wise)
Tensor2D a(2, 3);  // 2x3 matrix
Tensor2D b(3, 2);  // 3x2 matrix
Tensor2D result = a.mat_mul(b);  // Result is 2x2

// Identity matrix multiplication
Tensor2D identity(2, 2);
identity(0, 0) = 1.0f;
identity(1, 1) = 1.0f;
Tensor2D result = a.mat_mul(identity);  // a unchanged
```

## Neural Network Modules

The library provides a modular neural network framework with a base `Module` class and several layer implementations.

### Base Module Class
```cpp
#include "module.hpp"

// All neural network layers inherit from Module
class Module {
public:
    virtual Tensor2D forward(const Tensor2D& input) = 0;
    virtual ~Module() = default;
};
```

### Linear Layer
```cpp
#include "linear.hpp"

// Create a linear layer with input and output dimensions
Linear linear(4, 2);  // 4 input features, 2 output features

// Set weights and bias (optional - default values are provided)
linear.set_weights(Tensor2D::from_vector(4, 2, {
    0.1f, 0.2f,
    0.3f, 0.4f,
    0.5f, 0.6f,
    0.7f, 0.8f
}));
linear.set_bias(Tensor2D::from_vector(1, 2, {0.5f, 0.5f}));

// Forward pass: output = input * weights + bias
Tensor2D input = Tensor2D::from_vector(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
Tensor2D output = linear.forward(input);
```

### ReLU Layer
```cpp
#include "relu.hpp"

// Create a ReLU activation layer
ReLU relu;

// Forward pass: applies ReLU function (max(0, x)) to all elements
Tensor2D input = Tensor2D::from_vector(1, 3, {-1.0f, 0.0f, 2.0f});
Tensor2D output = relu.forward(input);  // Result: [0.0f, 0.0f, 2.0f]
```

### Softmax Layer
```cpp
#include "softmax.hpp"

// Create a Softmax activation layer
Softmax softmax;

// Forward pass: applies softmax function to each row independently
// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the row
Tensor2D input = Tensor2D::from_vector(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
Tensor2D output = softmax.forward(input);
// Result: probabilities that sum to 1.0 for each row

// For multiple rows, softmax is applied independently to each row
Tensor2D batch_input = Tensor2D::from_vector(2, 3, {
    1.0f, 2.0f, 3.0f,  // First row
    4.0f, 5.0f, 6.0f   // Second row
});
Tensor2D batch_output = softmax.forward(batch_input);
// Each row will have probabilities that sum to 1.0
```

### Sequential Container
```cpp
#include "sequential.hpp"

// Create a sequential model to chain layers together
Sequential model;

// Add layers to the model
auto linear1 = std::make_unique<Linear>(4, 2);
linear1->set_weights(Tensor2D::from_vector(4, 2, {
    0.1f, 0.2f,
    0.3f, 0.4f,
    0.5f, 0.6f,
    0.7f, 0.8f
}));
linear1->set_bias(Tensor2D::from_vector(1, 2, {0.5f, 0.5f}));

model.add(std::move(linear1));
model.add(std::make_unique<ReLU>());

// Forward pass through the entire model
Tensor2D input = Tensor2D::from_vector(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
Tensor2D output = model.forward(input);
```

### Complete Neural Network Example
```cpp
#include "sequential.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include "tensor2d.hpp"
#include <iostream>

int main() {
    // Define a neural network: Linear -> ReLU -> Linear -> Softmax
    Sequential model;
    
    // First layer: 4 inputs -> 3 outputs
    auto linear1 = std::make_unique<Linear>(4, 3);
    linear1->set_weights(Tensor2D::from_vector(4, 3, {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f
    }));
    linear1->set_bias(Tensor2D::from_vector(1, 3, {0.1f, 0.2f, 0.3f}));
    
    // Second layer: 3 inputs -> 3 outputs
    auto linear2 = std::make_unique<Linear>(3, 3);
    linear2->set_weights(Tensor2D::from_vector(3, 3, {
        0.5f, 0.6f, 0.7f,
        0.8f, 0.9f, 1.0f,
        1.1f, 1.2f, 1.3f
    }));
    linear2->set_bias(Tensor2D::from_vector(1, 3, {0.1f, 0.1f, 0.1f}));
    
    // Build the model
    model.add(std::move(linear1));
    model.add(std::make_unique<ReLU>());
    model.add(std::move(linear2));
    model.add(std::make_unique<Softmax>());
    
    // Run forward pass
    Tensor2D input = Tensor2D::from_vector(1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D output = model.forward(input);
    
    std::cout << "Input:" << std::endl;
    input.print();
    
    std::cout << "Output (probabilities):" << std::endl;
    output.print();
    
    // Verify that probabilities sum to 1.0
    float sum = output.sum();
    std::cout << "Sum of probabilities: " << sum << std::endl;
    
    return 0;
}
```

### Building and Running Neural Network Examples
```bash
# Build and run the forward pass example
g++ -std=c++17 -Iinclude -o build/forward_pass \
examples/forward_pass.cpp src/tensor2d.cpp src/linear.cpp src/relu.cpp src/softmax.cpp src/sequential.cpp && ./build/forward_pass
```

## Reduction Operations

### Sum
```cpp
// Sum all elements
float total = tensor.sum();
```

### Mean
```cpp
// Calculate mean of all elements
float average = tensor.mean();
```

### Maximum
```cpp
// Find maximum value
float max_val = tensor.max();

// Find position of maximum value
std::pair<size_t, size_t> max_pos = tensor.arg_max();
```

## Element-wise Functions

### ReLU (Rectified Linear Unit)
```cpp
// Apply ReLU function (max(0, x))
Tensor2D result = tensor.relu();

// In-place ReLU
tensor.relu_in_place();
```

### Negation
```cpp
// Negate all elements
Tensor2D result = tensor.negate();

// In-place negation
tensor.negate_in_place();
```

### Absolute Value
```cpp
// Take absolute value of all elements
Tensor2D result = tensor.abs();

// In-place absolute value
tensor.abs_in_place();
```

### Custom Element-wise Operations
```cpp
// Apply custom function to all elements
Tensor2D result = tensor.unary_op([](float x) { return x * x; });

// In-place custom operation
tensor.unary_op([](float x) { return std::sqrt(x); });
```

## Shape and Reshaping

### Get Shape
```cpp
// Get tensor dimensions
auto [rows, cols] = tensor.shape();
// or
std::pair<size_t, size_t> dims = tensor.shape();
```

### Reshape
```cpp
// Reshape tensor (must preserve total number of elements)
tensor.reshape(4, 1);  // From 2x2 to 4x1
```

## Broadcasting

### Automatic Broadcasting
Tensor2D supports NumPy-style broadcasting for arithmetic operations:

```cpp
// Broadcasting with different shapes
Tensor2D a(1, 3);  // Shape: 1x3
Tensor2D b(2, 1);  // Shape: 2x1
Tensor2D result = a + b;  // Result: 2x3

// Broadcasting rules:
// - Dimensions are aligned from the right
// - Missing dimensions are treated as 1
// - Dimensions of size 1 are broadcast to match larger dimensions
```

### Manual Broadcasting
```cpp
// Expand tensor to specific shape
Tensor2D expanded = tensor.expand(3, 4);
```

## Comparison Operations

### Equality
```cpp
// Check if two tensors are equal
bool are_equal = (a == b);

// Check if tensors are not equal
bool are_different = (a != b);
```

## Error Handling

The library provides comprehensive error checking:

```cpp
// Out of bounds access throws std::out_of_range
try {
    float value = tensor(10, 10);  // Will throw if tensor is smaller
} catch (const std::out_of_range& e) {
    std::cout << "Index out of bounds: " << e.what() << std::endl;
}

// Shape mismatch throws std::invalid_argument
try {
    Tensor2D result = a.mat_mul(b);  // Will throw if shapes incompatible
} catch (const std::invalid_argument& e) {
    std::cout << "Shape mismatch: " << e.what() << std::endl;
}
```

## Building and Running

### Build and Run Tests
```bash
g++ -std=c++17 -Iinclude -o build/test_runner \
tests/test_runner.cpp src/tensor2d.cpp src/linear.cpp src/relu.cpp src/softmax.cpp src/sequential.cpp && ./build/test_runner
```

This command does the following:
- `g++`: Invokes the C++ compiler.
- `-std=c++17`: Uses the C++17 standard.
- `-Iinclude`: Tells the compiler to look for header files in the `include` directory.
- `-o build/test_runner`: Specifies the output executable name and location.
- `tests/test_runner.cpp src/tensor2d.cpp`: The source files to compile.
- `&& ./build/test_runner`: Runs the compiled program if the build was successful.

### Build and Run Benchmark
```bash
g++ -std=c++17 -Iinclude -o build/benchmark benchmark.cpp src/tensor2d.cpp && ./build/benchmark
```

This command does the following:
- `g++`: Invokes the C++ compiler.
- `-std=c++17`: Uses the C++17 standard.
- `-Iinclude`: Tells the compiler to look for header files in the `include` directory.
- `-o build/benchmark`: Specifies the output executable name and location.
- `benchmark.cpp src/tensor2d.cpp`: The source files to compile.
- `&& ./build/benchmark`: Runs the compiled program if the build was successful.

## Complete Example

```cpp
#include "tensor2d.hpp"
#include <iostream>

int main() {
    // Create tensors
    Tensor2D a = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D b = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
    
    std::cout << "Tensor A:" << std::endl;
    a.print();
    
    std::cout << "Tensor B:" << std::endl;
    b.print();
    
    // Arithmetic operations
    Tensor2D sum = a + b;
    Tensor2D product = a * b;
    Tensor2D scaled = a * 2.0f;
    
    std::cout << "A + B:" << std::endl;
    sum.print();
    
    std::cout << "A * B (element-wise):" << std::endl;
    product.print();
    
    // Matrix multiplication
    Tensor2D matmul_result = a.mat_mul(b);
    std::cout << "A @ B (matrix multiplication):" << std::endl;
    matmul_result.print();
    
    // Reductions
    std::cout << "Sum of A: " << a.sum() << std::endl;
    std::cout << "Mean of A: " << a.mean() << std::endl;
    std::cout << "Max of A: " << a.max() << std::endl;
    
    // Element-wise functions
    Tensor2D relu_result = a.relu();
    std::cout << "ReLU(A):" << std::endl;
    relu_result.print();
    
    return 0;
}
```

## Performance Notes

- The library uses contiguous memory layout for efficient cache access
- Matrix multiplication uses naive O(n³) algorithm - suitable for small matrices
- Broadcasting operations create new tensors rather than views
- In-place operations are available for better performance when possible

## Limitations

- Only supports 2D tensors
- Limited to float data type
- No GPU acceleration
- No automatic differentiation
- Broadcasting creates new tensors (no view semantics) 