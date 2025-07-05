# Mini Tensor

A simple C++ tensor library with PyTorch-style modules and forward pass support — for learning how neural nets work under the hood.

## Project Structure

```
mini-tensor/
├── include/
│   ├── tensor2d.hpp         # Tensor2D public interface
│   ├── tensor3d.hpp         # Tensor3D public interface
│   ├── module.hpp           # Base Module class
│   ├── linear.hpp           # Linear layer
│   ├── relu.hpp             # ReLU activation layer
│   ├── softmax.hpp          # Softmax activation layer
│   ├── sequential.hpp       # Sequential container
│   ├── tensor2d_view.hpp    # Tensor2DView public interface
│   ├── ir_trace.hpp         # IR tracing system to log tensor operations
│   ├── matmul_cuda.hpp      # Header for CUDA-based matmul
│   └── device.hpp           # Device enumeration (CPU/GPU)
├── src/
│   ├── tensor2d.cpp         # Tensor2D implementation
│   ├── tensor3d.cpp         # Tensor3D implementation
│   ├── linear.cpp           # Linear layer implementation
│   ├── relu.cpp             # ReLU layer implementation
│   ├── softmax.cpp          # Softmax layer implementation
│   ├── sequential.cpp       # Sequential container implementation
│   └── matmul_cuda.cu       # CUDA kernel for matrix multiplication
├── examples/
│   └── forward_pass.cpp     # Neural network example
├── tests/
│   └── test_runner.cpp      # Test file
├── benchmark.cpp            # Performance benchmarks for matrix multiplication
├── build/                   # Used for build artifacts
├── .gitignore
└── README.md
```

## Quick Start

### Run Tests

#### CPU-only
```bash
g++ -std=c++17 -Iinclude -Ithird_party/eigen \
    tests/test_runner.cpp \
    src/tensor2d.cpp src/tensor3d.cpp src/linear.cpp \
    src/relu.cpp src/sequential.cpp src/softmax.cpp src/tensor2d_view.cpp \
    -o build/test_runner

./build/test_runner
```

#### GPU-enabled (CUDA)
*Requires an NVIDIA GPU (e.g., T4 on GCP) and CUDA toolkit*

```bash
nvcc --expt-relaxed-constexpr -std=c++17 \
    -Iinclude -Ithird_party/eigen \
    -c src/matmul_cuda.cu -o build/matmul_cuda.o

g++ -std=c++17 -Iinclude -Ithird_party/eigen -DUSE_CUDA \
    tests/test_runner.cpp \
    src/tensor2d.cpp src/tensor3d.cpp src/linear.cpp \
    src/relu.cpp src/sequential.cpp src/softmax.cpp src/tensor2d_view.cpp \
    build/matmul_cuda.o \
    -o build/test_runner \
    -L/usr/local/cuda/lib64 -lcudart -lcublas

./build/test_runner
```

### Run Neural Network Example
```bash
g++ -std=c++17 -Iinclude -Ithird_party/eigen -o build/forward_pass \
examples/forward_pass.cpp src/tensor2d.cpp src/tensor3d.cpp src/tensor2d_view.cpp src/linear.cpp src/relu.cpp src/softmax.cpp src/sequential.cpp && ./build/forward_pass
```

### Run Benchmarks
```bash
g++ -std=c++17 -Iinclude -Ithird_party/eigen -o build/benchmark benchmark.cpp src/tensor2d.cpp src/tensor3d.cpp && ./build/benchmark
```

## Features

- **2D Tensor Operations**: Element-wise arithmetic, broadcasting, matrix multiplication
- **3D Tensor Operations**: Batched tensors, batched matrix multiplication
- **Neural Network Modules**: Linear layers, ReLU activation, Softmax activation, Sequential containers
- **Forward Pass**: Run input through neural network models
- **Performance**: Contiguous memory layout for efficient cache access; matmul benchmarks included
- **IR Trace**: All Tensor2D operations are tracked in a global IR trace for debugging and introspection
- **Unique Tensor IDs**: Every Tensor2D instance is assigned a unique ID for traceability
- **GPU Acceleration**: CUDA-based matrix multiplication

## GPU Acceleration

Tensor2D supports both `Device::CPU` and `Device::GPU` device types. The `mat_mul_cuda()` function dispatches to a custom CUDA kernel for GPU-based matrix multiplication, providing significant speedup for large matrices.

```cpp
// GPU-accelerated matrix multiplication
Tensor2D A = Tensor2D::from_random(1024, 1024, Device::GPU);
Tensor2D B = Tensor2D::from_random(1024, 1024, Device::GPU);
Tensor2D C = mat_mul_cuda(A, B);  // Runs on GPU
```

## IR Trace

The library automatically tracks all major operations in a global IR trace, including:
- **Arithmetic operators**: `+`, `-`, `*`, `/`
- **Matrix operations**: `mat_mul`, `mat_mul_eigen`, `mat_mul_eigen_parallel`, `mat_mul_cuda`
- **Element-wise functions**: `abs`, `neg`, `relu`
- **Neural network modules**: `Linear`, `Softmax`, `Sequential`

### Example IR Trace Output

```cpp
#include "tensor2d.hpp"
#include "linear.hpp"
#include "ir_trace.hpp"

TensorID::reset();
IRTrace::reset();

Tensor2D a = Tensor2D::from_random(2, 2);
Tensor2D b = Tensor2D::from_random(2, 2);
Tensor2D c = a + b;  // Addition
Linear linear(2, 2);
Tensor2D output = linear.forward(c);

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
[1] Operation: mat_mul
    Inputs : tensor_2, tensor_3
    Output : tensor_4
    Shape  : 2 x 2
    Device : CPU
[2] Operation: operator+
    Inputs : tensor_4, tensor_5
    Output : tensor_6
    Shape  : 2 x 2
    Device : CPU
[3] Operation: linear
    Inputs : tensor_2, tensor_3, tensor_5
    Output : tensor_6
    Shape  : 2 x 2
    Device : CPU
```

📖 Full API docs and usage examples → [See demo.md](demo.md)