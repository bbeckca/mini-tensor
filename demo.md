# Tensor2D and Tensor3D Demo and Feature Documentation
_Last updated: July 4, 2025_

## Table of Contents
- [Constructors](#constructors)
- [Basic Operations](#basic-operations)
- [Tensor2DView and Slicing](#tensor2dview-and-slicing)
- [Arithmetic Operations](#arithmetic-operations)
- [Matrix Operations](#matrix-operations)
- [Tensor3D and Batched Matrix Operations](#tensor3d-and-batched-matrix-operations)
- [Fused vs Unfused CUDA Benchmark: bmm_add_cuda](#fused-vs-unfused-cuda-benchmark-bmm_add_cuda)
- [Neural Network Modules](#neural-network-modules)
- [Reduction Operations](#reduction-operations)
- [Element-wise Functions](#element-wise-functions)
- [Shape and Reshaping](#shape-and-reshaping)
- [Broadcasting](#broadcasting)
- [Comparison Operations](#comparison-operations)
- [Error Handling](#error-handling)
- [IR Trace and Tensor IDs](#ir-trace-and-tensor-ids)
- [Building and Running](#building-and-running)

## Constructors

### Basic Constructor
```cpp
// Create a 3x4 tensor filled with zeros on the default device (CPU)
Tensor2D tensor(3, 4);

// Create a 2x2 tensor filled with a specific value on the default device (CPU)
Tensor2D tensor(2, 2, 42.0f);

// Create a 2x2 tensor on a specific device (CPU or GPU)
Tensor2D tensor(2, 2, 0.0f, Device::GPU); // On GPU
```

- Each `Tensor2D` contains device information, indicating whether it resides on the CPU or GPU. By default, tensors are created on the CPU unless specified otherwise.

### From Vector
```cpp
// Create tensor from existing data (on CPU)
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
Tensor2D tensor = Tensor2D::from_vector(2, 2, data);

// Create tensor from vector on a specific device
Tensor2D tensor = Tensor2D::from_vector(2, 2, data, Device::GPU);
```

### From Random
```cpp
// Create a 3x4 tensor with random values in [0, 1)
Tensor2D tensor = Tensor2D::from_random(3, 4);
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
// Print tensor contents (includes device information)
tensor.print();
```

### Device Access
```cpp
// Get the device of a tensor
Device device = tensor.get_device();
std::cout << "Tensor is on device: " << to_string(device) << std::endl;
```

## Tensor2DView and Slicing

`Tensor2DView` provides a non-owning view (slice) into a subregion of a `Tensor2D` without copying data. Modifying the view updates the original tensor.

### Creating a View (Slice)
```cpp
#include "tensor2d_view.hpp"

Tensor2D base(4, 4);
// Fill base tensor for demonstration
for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j)
        base(i, j) = i * 4 + j;

// Create a view of rows 1-2 and cols 1-2 (exclusive of end)
Tensor2DView view(base, 1, 3, 1, 3);  // 2x2 view of the center
```

### Accessing and Modifying Elements
```cpp
float value = view(0, 1);      // Access element at (0,1) in the view
view(0, 0) = 42.0f;            // Modify element in the view (also updates base tensor)
```

### Shape and Properties
```cpp
auto [rows, cols] = view.shape();  // Get view dimensions
bool empty = view.is_empty();      // Check if view is empty
```

### Printing a View
```cpp
view.print();  // Print the contents of the view
```

### Example: View Reflects Base Tensor Changes
```cpp
// Changing the view updates the base tensor
view(0, 0) = 99.0f;
std::cout << base(1, 1) << std::endl;  // Prints 99.0
```

### Out-of-Bounds Handling
Accessing outside the view's shape throws `std::out_of_range`.

### Notes
- Views do not own data; they reference the original tensor.
- Slicing is defined by `[row_start, row_end)` and `[col_start, col_end)` (end-exclusive).
- Modifying a view modifies the base tensor.

## Arithmetic Operations

### Broadcasting Overview

`Tensor2D` supports NumPy-style broadcasting for all arithmetic operations (`+`, `-`, `*`, `/`). When operating on tensors of different shapes, dimensions of size 1 are automatically expanded to match the other tensor, following standard broadcasting rules.

Broadcasting allows operations between tensors of different, but compatible, shapes.  
- If a dimension is `1` in one tensor and `N` in the other, the size-1 dimension is **stretched**.  
- If shapes are incompatible, an exception is thrown.

The same broadcasting rules apply for -, *, and / operators.

### Example: Broadcasting with Addition

```cpp
Tensor2D a(1, 3);         // Shape: 1x3
a.fill(1.0f);

Tensor2D b(2, 1);         // Shape: 2x1
b.fill(2.0f);

Tensor2D result = a + b;  // Broadcasted to 2x3, all elements = 3.0f
```

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

### Eigen Matrix Multiplication
The library also provides an Eigen-based matrix multiplication implementation for comparison:

```cpp
// Eigen-based matrix multiplication
Tensor2D result = a.mat_mul_eigen(b);
```

#### Performance Comparison
Benchmark results comparing manual implementation vs Eigen (all CPU benchmarks use `mat_mul_eigen()` and `mat_mul_eigen_parallel()`):

| Matrix Size | Manual (μs) | Eigen (μs) | Speedup |
|-------------|-------------|------------|---------|
| 16×16       | 48          | 154        | 0.31x   |
| 128×128     | 16,960      | 3,437      | 4.93x   |
| 256×512     | 92,871      | 19,086     | 4.87x   |
| 512×512     | 691,489     | 149,118    | 4.64x   |
| 1024×1024   | 5,633,871   | 1,214,421  | 4.64x   |

**Key observations:**
- For small matrices (16×16), the manual implementation is faster due to lower overhead
- For larger matrices (128×128 and above), Eigen provides significant speedup (~4.6-4.9x)
- Eigen's optimized BLAS implementation scales better with matrix size
- The manual implementation uses a naive O(n³) algorithm, while Eigen uses optimized algorithms

#### GPU vs CPU Performance (Eigen-based CPU benchmarks)

| Shape             | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------------|---------------|----------------|---------|
| 512 × 512         | 859.59        | 1.20           | 714.93×  |
| 1024 × 1024       | 6912.91       | 10.32          | 669.61×  |

## Tensor3D and Batched Matrix Operations

`Tensor3D` provides support for batched 2D tensors and batched matrix multiplication, enabling efficient operations on a stack of matrices (e.g., for mini-batch neural network computations). The implementation uses a contiguous `float*` buffer for GPU compatibility and efficient memory access, improving compatibility with batched GPU matmul via `bmm_cuda()`.

### Tensor3D API

```cpp
#include "tensor3d.hpp"

// Construct a batch of 2D tensors (e.g., 8 matrices of shape 16x16, filled with 1.0f)
Tensor3D batch(8, 16, 16, 1.0f);

// Construct a batch of 2D tensors with random values in [0, 1)
Tensor3D random_batch = Tensor3D::from_random(8, 16, 16);

// Initialize from explicit data
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
Tensor3D explicit_batch = Tensor3D::from_vector(2, 2, 2, data);

// Access a specific matrix in the batch (returns a non-owning view)
Tensor2D mat = batch[0];  // Delegates to slice_batch(0)

// Get batch size and shape
size_t batch_size = batch.batch_size();
size_t rows = batch.rows();
size_t cols = batch.cols();

// Batched matrix multiplication (manual, Eigen, and CUDA)
Tensor3D A(8, M, K, 1.0f);
Tensor3D B(8, K, N, 2.0f);
Tensor3D C_manual = A.mat_mul(B);                    // Manual implementation
Tensor3D C_eigen  = A.mat_mul_eigen(B);              // Eigen-accelerated
Tensor3D C_cuda   = bmm_cuda(A.to(Device::GPU), B.to(Device::GPU));  // CUDA-accelerated
```

### Batched Matrix Multiplication Benchmarks

| Batch × M × K × N         | Manual (μs) | Eigen (μs) | Eigen Parallel (μs) | Speedup | Speedup (Parallel) |
|---------------------------|-------------|------------|---------------------|---------|--------------------|
| 8 × 16 × 16 × 16          | 259         | 79         | 129                 | 3.28x   | 2.01x              |
| 16 × 64 × 64 × 64         | 29,490      | 2,233      | 923                 | 13.21x  | 31.95x             |
| 32 × 128 × 128 × 128      | 461,121     | 21,028     | 8,985               | 21.93x  | 51.33x             |
| 8 × 256 × 512 × 128       | 904,623     | 29,978     | 18,751              | 30.18x  | 48.26x             |
| 4 × 512 × 512 × 512       | 3,636,260   | 88,458     | 142,519             | 41.11x  | 25.52x             |
| 2 × 1024 × 1024 × 1024    | 14,824,380  | 260,751    | 1,110,492           | 56.85x  | 13.35x             |

**Key observations:**
- For small batches/matrices, manual implementation can be competitive.
- For larger batches and matrices, Eigen provides significant speedup (13x–57x), and parallelization can further improve performance, especially for medium-to-large batch sizes.
- For very large matrices, parallelization may have diminishing returns due to thread overhead.
- Batched matmul is essential for deep learning and large-scale computations.

### GPU vs CPU Batched Matrix Multiplication (Eigen Parallel CPU benchmarks)

| Batch × M × K × N         | CPU Time (ms) | GPU Time (ms) | Speedup |
|---------------------------|---------------|----------------|---------|
| 8 × 16 × 16 × 16          | 0.129         | 0.036         | 3.6×    |
| 16 × 64 × 64 × 64         | 0.923         | 0.003         | 292×    |
| 32 × 128 × 128 × 128      | 8.985         | 0.027         | 332×    |
| 8 × 256 × 512 × 128       | 18.751        | 0.040         | 474×    |
| 4 × 512 × 512 × 512       | 142.519       | 0.236         | 603×    |
| 2 × 1024 × 1024 × 1024    | 1,110.492     | 1.607         | 691×    |

## Fused vs Unfused CUDA Benchmark: bmm_add_cuda

The `bmm_add_cuda` function implements a fused kernel that performs batched matrix multiplication followed by bias addition in a single CUDA kernel. This eliminates the need for separate memory operations and can provide performance benefits by reducing memory bandwidth usage and kernel launch overhead.

### Benchmark Results

The following benchmarks compare the performance of the fused `bmm_add_cuda` kernel against the unfused approach (separate `bmm_cuda` + `add` operations):

| Batch × M × K × N         | Unfused (ms) | Fused (ms) | Speedup |
|---------------------------|--------------|------------|---------|
| 8 × 16 × 16 × 16          | 0.129338     | 0.055994   | 2.31×   |
| 16 × 64 × 64 × 64         | 0.131719     | 0.275237   | 0.48×   |
| 32 × 128 × 128 × 128      | 1.79135      | 1.25413    | 1.43×   |
| 8 × 256 × 512 × 128       | 1.72189      | 1.39071    | 1.24×   |
| 4 × 512 × 512 × 512       | 4.58485      | 3.86199    | 1.19×   |
| 2 × 1024 × 1024 × 1024    | 13.0954      | 12.1351    | 1.08×   |

**Key observations:**
- For small matrices (8×16×16), the fused kernel provides significant speedup (~2.3x) due to reduced kernel launch overhead
- For medium matrices (16×64×64), the unfused approach is actually faster, likely due to better cache utilization in separate kernels
- For larger matrices, the fused kernel shows consistent but modest speedup (1.1-1.4x)
- The performance benefit of fusion varies with matrix size and batch dimensions

### Example: Batched Matrix Multiplication (with Parallel Eigen and CUDA)
```cpp
#include "tensor3d.hpp"
#include "matmul_cuda.hpp"
#include <iostream>

int main() {
    Tensor3D A(8, 16, 16, 1.0f);
    Tensor3D B(8, 16, 16, 2.0f);
    Tensor3D C = A.mat_mul(B); // Manual batched matmul
    Tensor3D D = A.mat_mul_eigen(B); // Eigen batched matmul
    Tensor3D E = A.mat_mul_eigen_parallel(B); // Eigen batched matmul (parallel)
    
    #ifdef USE_CUDA
    Tensor3D A_gpu = A.to(Device::GPU);
    Tensor3D B_gpu = B.to(Device::GPU);
    Tensor3D F = bmm_cuda(A_gpu, B_gpu); // CUDA batched matmul
    std::cout << "F[0](0,0): " << F.to(Device::CPU)[0](0,0) << std::endl;  // [0] delegates to slice_batch(0)
    #endif
    
    std::cout << "C[0](0,0): " << C[0](0,0) << std::endl;  // [0] delegates to slice_batch(0)
    std::cout << "D[0](0,0): " << D[0](0,0) << std::endl;  // [0] delegates to slice_batch(0)
    std::cout << "E[0](0,0): " << E[0](0,0) << std::endl;  // [0] delegates to slice_batch(0)
    return 0;
}
```

### Testing and Building
- Tensor3D is tested in `tests/test_runner.cpp` (see [README](README.md) for test commands).
- Most implementation is in the header (`include/tensor3d.hpp`). The `.cpp` file is provided for completeness.
- See [README](README.md) for build/test/benchmark instructions including Tensor3D.

### Advanced Tensor3D Operations

```cpp
// Create a non-owning view of a specific batch
Tensor2D batch_view = tensor3d.slice_batch(2);  // View of batch index 2

// Copy data into a specific batch slice
Tensor2D new_data = Tensor2D::from_random(16, 16);
tensor3d.set_batch(1, new_data);  // Copy new_data into batch index 1

// Note: slice_batch() and set_batch() are currently CPU-only operations
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
g++ -std=c++17 -Iinclude -Ithird_party/eigen -o build/forward_pass \
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

## IR Trace and Tensor IDs

### IR Trace

All major operations on `Tensor2D`—such as arithmetic, `relu`, `matmul`, and `reshape`—are automatically recorded in a global IR trace. Each entry logs the operation name, input tensor IDs, output tensor ID, shape, and device. This trace is useful for debugging and introspection.

#### Printing the IR Trace

Call `IRTrace::print()` after performing tensor operations to display the trace:

```cpp
#include "tensor2d.hpp"
#include "relu.hpp"
#include "ir_trace.hpp"

Tensor2D a = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}); // tensor_186
Tensor2D b = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f}); // tensor_187
Tensor2D c = a.mat_mul(b); // tensor_188
Tensor2D d = c.relu();      // tensor_189

IRTrace::print();
```

**Output:**
```text
Printing IRTrace:
[0] Operation: mat_mul
    Inputs : tensor_186, tensor_187
    Output : tensor_188
    Shape  : 2 x 2
    Device : CPU
[1] Operation: relu
    Inputs : tensor_188
    Output : tensor_189
    Shape  : 2 x 2
    Device : CPU
```

#### Arithmetic Operations IR Trace

The IR trace captures all arithmetic operations (`+`, `-`, `*`, `/`):

```cpp
#include "tensor2d.hpp"
#include "ir_trace.hpp"

TensorID::reset();
IRTrace::reset();

Tensor2D a = Tensor2D::from_random(2, 2);
Tensor2D b = Tensor2D::from_random(2, 2);
Tensor2D c = a + b;  // Addition
Tensor2D d = a - b;  // Subtraction
Tensor2D e = a * b;  // Multiplication
Tensor2D f = a / b;  // Division

IRTrace::print();
```

**Output:**
```text
Printing IRTrace:
[0] Operation: operator+
    Inputs : tensor_0, tensor_1
    Output : tensor_2
    Shape  : 2 x 2
    Device : CPU
[1] Operation: operator-
    Inputs : tensor_0, tensor_1
    Output : tensor_3
    Shape  : 2 x 2
    Device : CPU
[2] Operation: operator*
    Inputs : tensor_0, tensor_1
    Output : tensor_4
    Shape  : 2 x 2
    Device : CPU
[3] Operation: operator/
    Inputs : tensor_0, tensor_1
    Output : tensor_5
    Shape  : 2 x 2
    Device : CPU
```

#### Linear Layer IR Trace

The Linear layer's forward pass generates multiple operations:

```cpp
#include "linear.hpp"
#include "ir_trace.hpp"

TensorID::reset();
IRTrace::reset();

Linear linear(3, 2);
linear.set_weights(Tensor2D::from_random(3, 2));
linear.set_bias(Tensor2D::from_random(1, 2));
Tensor2D input = Tensor2D::from_random(1, 3);
Tensor2D output = linear.forward(input);

IRTrace::print();
```

**Output:**
```text
Printing IRTrace:
[0] Operation: mat_mul
    Inputs : tensor_4, tensor_2
    Output : tensor_5
    Shape  : 1 x 2
    Device : CPU
[1] Operation: operator+
    Inputs : tensor_5, tensor_3
    Output : tensor_6
    Shape  : 1 x 2
    Device : CPU
[2] Operation: linear
    Inputs : tensor_4, tensor_2, tensor_3
    Output : tensor_6
    Shape  : 1 x 2
    Device : CPU
```

The Linear layer generates three trace entries:
1. **mat_mul**: Matrix multiplication of input with weights
2. **operator+**: Addition of bias to the result
3. **linear**: Composite operation recording all inputs and output

**Note**: The IR trace records shapes as `std::variant<std::pair<size_t, size_t>, std::tuple<size_t, size_t, size_t>>` to support both 2D and 3D tensors.

#### Sequential Model IR Trace

A complete Sequential model demonstrates the full computation graph:

```cpp
#include "sequential.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "ir_trace.hpp"
#include <iostream>

int main() {
    TensorID::reset();
    IRTrace::reset();

    // Build a Sequential model: Linear -> ReLU -> Linear
    Sequential model;
    model.add(std::make_unique<Linear>(3, 4));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(4, 2));

    // Run forward pass
    Tensor2D input = Tensor2D::from_random(1, 3);
    Tensor2D output = model.forward(input);

    // Print the complete IR trace
    std::cout << "Complete computation graph:" << std::endl;
    IRTrace::print();
    
    return 0;
}
```

**Output:**
```text
Complete computation graph:
Printing IRTrace:
[0] Operation: mat_mul
    Inputs : tensor_4, tensor_2
    Output : tensor_5
    Shape  : 1 x 4
    Device : CPU
[1] Operation: operator+
    Inputs : tensor_5, tensor_3
    Output : tensor_6
    Shape  : 1 x 4
    Device : CPU
[2] Operation: linear
    Inputs : tensor_4, tensor_2, tensor_3
    Output : tensor_6
    Shape  : 1 x 4
    Device : CPU
[3] Operation: relu
    Inputs : tensor_6
    Output : tensor_7
    Shape  : 1 x 4
    Device : CPU
[4] Operation: mat_mul
    Inputs : tensor_7, tensor_8
    Output : tensor_9
    Shape  : 1 x 2
    Device : CPU
[5] Operation: operator+
    Inputs : tensor_9, tensor_10
    Output : tensor_11
    Shape  : 1 x 2
    Device : CPU
[6] Operation: linear
    Inputs : tensor_7, tensor_8, tensor_10
    Output : tensor_11
    Shape  : 1 x 2
    Device : CPU
[7] Operation: sequential
    Inputs : tensor_4, tensor_2, tensor_3, tensor_8, tensor_10
    Output : tensor_11
    Shape  : 1 x 2
    Device : CPU
```

This trace shows the complete computation graph:
- **Operations 0-2**: First Linear layer (mat_mul + bias + composite)
- **Operation 3**: ReLU activation
- **Operations 4-6**: Second Linear layer (mat_mul + bias + composite)
- **Operation 7**: Sequential composite operation

- The IR trace records every major operation performed on `Tensor2D` objects.
- Each entry shows the operation, input tensor IDs, output tensor ID, shape, and device in a readable, indented format.
- Use the trace to debug or inspect the computation graph of your tensor code.

### Tensor IDs

Every `Tensor2D` is assigned a unique string ID at construction. This ID is used in the IR trace and can be accessed programmatically:

```cpp
std::string id = tensor.get_id();
```

## Building and Running

### Build Instructions

The library supports both CPU-only and CUDA-enabled builds. CUDA builds require an NVIDIA GPU (e.g., T4 on GCP) and CUDA toolkit.

#### Required Flags for CUDA Builds:
- `-DUSE_CUDA`: Enables CUDA support and GPU benchmarks
- `-I/usr/local/cuda/include`: Includes CUDA headers
- `-L/usr/local/cuda/lib64 -lcudart -lcublas`: Links CUDA runtime and cuBLAS libraries

### Running Tests

#### CPU-only
```bash
g++-15 -fopenmp -std=c++17 -Iinclude -Ithird_party/eigen -o build/test_runner \
  tests/test_runner.cpp src/tensor2d.cpp src/tensor3d.cpp src/tensor2d_view.cpp src/linear.cpp src/relu.cpp src/softmax.cpp src/sequential.cpp \
  && ./build/test_runner
```

This command does the following:
- `g++-15`: Invokes the C++15 compiler.
- `-fopenmp`: Enables OpenMP for parallelization.
- `-std=c++17`: Uses the C++17 standard.
- `-Iinclude`: Tells the compiler to look for header files in the `include` directory.
- `-Ithird_party/eigen`: Tells the compiler to look for Eigen header files.
- `-o build/test_runner`: Specifies the output executable name and location.
- `tests/test_runner.cpp src/tensor2d.cpp src/tensor3d.cpp src/tensor2d_view.cpp`: The source files to compile.
- `&& ./build/test_runner`: Runs the compiled program if the build was successful.

#### GPU-enabled (CUDA)
*Requires an NVIDIA GPU (e.g., T4 on GCP) and CUDA toolkit*

```bash
# Compile CUDA matmul kernel
nvcc --expt-relaxed-constexpr -std=c++17 \
    -Iinclude -Ithird_party/eigen \
    -c src/matmul_cuda.cu -o build/matmul_cuda.o

# Build test runner with CUDA support
g++ -std=c++17 -Iinclude -Ithird_party/eigen -I/usr/local/cuda/include -DUSE_CUDA \
    tests/test_runner.cpp \
    src/tensor2d.cpp src/tensor3d.cpp src/linear.cpp \
    src/relu.cpp src/sequential.cpp src/softmax.cpp src/tensor2d_view.cpp \
    build/matmul_cuda.o \
    -o build/test_runner \
    -L/usr/local/cuda/lib64 -lcudart -lcublas

# Run tests
./build/test_runner
```

### Running Benchmarks

#### CPU-only
```bash
g++ -std=c++17 -Iinclude -Ithird_party/eigen -o build/benchmark benchmark.cpp src/tensor2d.cpp src/tensor3d.cpp && ./build/benchmark
```

This command does the following:
- `g++`: Invokes the C++ compiler.
- `-std=c++17`: Uses the C++17 standard.
- `-Iinclude`: Tells the compiler to look for header files in the `include` directory.
- `-Ithird_party/eigen`: Tells the compiler to look for Eigen header files.
- `-o build/benchmark`: Specifies the output executable name and location.
- `benchmark.cpp src/tensor2d.cpp src/tensor3d.cpp`: The source files to compile.
- `&& ./build/benchmark`: Runs the compiled program if the build was successful.

#### GPU-enabled (CUDA)
*Requires an NVIDIA GPU (e.g., T4 on GCP) and CUDA toolkit*

```bash
# Compile CUDA matmul kernel
nvcc --expt-relaxed-constexpr -std=c++17 -Iinclude -Ithird_party/eigen \
    -c src/matmul_cuda.cu -o build/matmul_cuda.o

# Build benchmark binary
g++ -std=c++17 -Iinclude -Ithird_party/eigen -I/usr/local/cuda/include -DUSE_CUDA \
    benchmark.cpp src/tensor2d.cpp src/tensor3d.cpp build/matmul_cuda.o \
    -o build/benchmark \
    -L/usr/local/cuda/lib64 -lcudart -lcublas

# Run benchmarks
./build/benchmark
```

This command does the following:
- `nvcc`: NVIDIA CUDA compiler to compile the CUDA kernel
- `g++`: C++ compiler with `-DUSE_CUDA` flag to enable GPU benchmarks
- `-L/usr/local/cuda/lib64 -lcudart -lcublas`: Links against CUDA runtime and cuBLAS libraries
- The resulting executable will run both CPU and GPU benchmarks when available

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
- Manual matrix multiplication uses a naive O(n³) algorithm—suitable only for small matrices
- Eigen-based matrix multiplication (including parallel) uses highly optimized algorithms (cache blocking, SIMD, and multi-threading), making it efficient for both small and large matrices
- Broadcasting operations create new tensors rather than views
- In-place operations are available for better performance when possible

## CUDA Support

The library supports GPU acceleration through CUDA for matrix multiplication operations with comprehensive device memory management.

### Architecture Update

**Tensor2D** now uses `float*` instead of `std::vector<float>` to support device memory:

- **Device Memory Support**: Raw pointers enable direct CUDA memory allocation and management
- **Runtime Safety**: All CPU-side operations validate device type to prevent invalid GPU memory access
- **Explicit Transfer**: `to(Device::CPU)` and `to(Device::GPU)` enable safe device transfer
- **Device-Aware Copy**: `copy_from`, assignment operator, and copy constructor handle device memory correctly

### Runtime Safety

Runtime safety checks (e.g., in `operator()`, `operator[]`) only apply to CPU-side access. GPU-side kernel code accesses memory directly via raw `float*` without validation logic.

### Memory Semantics

```cpp
// Device transfer
Tensor2D cpu_tensor = Tensor2D::from_random(1024, 1024, Device::CPU);
Tensor2D gpu_tensor = cpu_tensor.to(Device::GPU);

// Memory copy with validation
Tensor2D source = Tensor2D::from_random(2, 3, Device::CPU);
Tensor2D dest = Tensor2D(2, 3, 0.0f, Device::CPU);
dest.copy_from(source);  // Validates shape and device compatibility

// Deep copy semantics
Tensor2D original = Tensor2D::from_random(512, 512, Device::GPU);
Tensor2D copy(original);  // Proper device allocation and copy
```

### GPU Operations

```cpp
#include "tensor2d.hpp"
#include "matmul_cuda.hpp"

// GPU-accelerated matrix multiplication
Tensor2D A = Tensor2D::from_random(1024, 1024, Device::GPU);
Tensor2D B = Tensor2D::from_random(1024, 1024, Device::GPU);
Tensor2D C = mat_mul_cuda(A, B);  // Runs on GPU

// GPU-accelerated batched matrix multiplication
Tensor3D batch_A = Tensor3D::from_random(8, 256, 512, Device::GPU);
Tensor3D batch_B = Tensor3D::from_random(8, 512, 128, Device::GPU);
Tensor3D batch_C = bmm_cuda(batch_A, batch_B);  // Runs on GPU
```

The `matmul_cuda.cu` file defines both `mat_mul_cuda()` (for Tensor2D) and `bmm_cuda()` (for Tensor3D) CUDA kernels.

### Performance Benchmarks

On an NVIDIA T4 instance (GCP):

#### Matrix Multiplication Performance

| Shape             | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------------|---------------|----------------|---------|
| 512 × 512         | 2287.25       | 1864.76        | 1.23×    |
| 1024 × 1024       | 23455.9       | 18.49          | 1268.24× |

#### Device Transfer Performance

| Shape             | CPU → GPU (us) | GPU → CPU (us) | Roundtrip (us) |
|------------------|----------------|----------------|----------------|
| 512 × 512         | 463            | 1374           | 1837           |
| 1024 × 1024       | 1225           | 15726          | 16951          |

**Note**: GPU → CPU transfer is significantly slower due to PCIe bandwidth limits.

### Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- Compile with `-DUSE_CUDA` flag and link against CUDA libraries

## Limitations

- Only supports 2D, 3D tensors
- Limited to float data type
- No automatic differentiation
- Broadcasting creates new tensors (no view semantics)

## Advanced Details

<details>
<summary>CUDA Memory Management Implementation</summary>

### Memory Allocation Strategy

The library uses different memory allocation strategies for CPU and GPU:

```cpp
// CPU allocation (standard new/delete)
if (device == Device::CPU) {
    data_ = new float[size];
    std::fill(data_, data_ + size, val);
}

// GPU allocation (CUDA malloc/free)
if (device == Device::GPU) {
    cudaError_t err = cudaMalloc(&data_, size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed");
    }
}
```

### Device Transfer Implementation

The `to(Device)` method implements safe device transfer:

```cpp
Tensor2D to(Device target) const {
    if (device_ == target) return *this;
    
    Tensor2D result(rows_, cols_, 0.0f, target);
    size_t bytes = rows_ * cols_ * sizeof(float);
    
    if (target == Device::GPU) {
        cudaMemcpy(result.data_, data_, bytes, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(result.data_, data_, bytes, cudaMemcpyDeviceToHost);
    }
    
    return result;
}
```

### Runtime Safety Checks

All CPU-side memory access operations include device validation:

```cpp
float& operator()(size_t row, size_t col) {
    if (device_ == Device::GPU) {
        throw std::runtime_error("Cannot access GPU tensor data directly from CPU code");
    }
    // ... rest of implementation
}
```

**Note**: GPU-side kernel code accesses memory directly via raw `float*` without validation logic.

</details>

<details>
<summary>IR Trace for GPU Operations</summary>

GPU operations are tracked in the IR trace just like CPU operations:

```cpp
#include "tensor2d.hpp"
#include "matmul_cuda.hpp"
#include "ir_trace.hpp"

TensorID::reset();
IRTrace::reset();

Tensor2D A = Tensor2D::from_random(2, 2, Device::GPU);
Tensor2D B = Tensor2D::from_random(2, 2, Device::GPU);
Tensor2D C = mat_mul_cuda(A, B);

IRTrace::print();
```

**Output:**
```text
Printing IRTrace:
[0] Operation: mat_mul_cuda
    Inputs : tensor_0, tensor_1
    Output : tensor_2
    Shape  : 2 x 2
    Device : GPU
```

Batched GPU operations are also tracked:

```cpp
#include "tensor3d.hpp"
#include "matmul_cuda.hpp"
#include "ir_trace.hpp"

TensorID::reset();
IRTrace::reset();

Tensor3D A = Tensor3D::from_random(4, 8, 8, Device::GPU);
Tensor3D B = Tensor3D::from_random(4, 8, 8, Device::GPU);
Tensor3D C = bmm_cuda(A, B);

IRTrace::print();
```

**Output:**
```text
Printing IRTrace:
[0] Operation: bmm_cuda
    Inputs : tensor_0, tensor_1
    Output : tensor_2
    Shape  : 4 x 8 x 8
    Device : GPU
```

</details> 