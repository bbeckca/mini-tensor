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
│   └── sequential.hpp       # Sequential container
├── src/
│   ├── tensor2d.cpp         # Tensor2D implementation
│   ├── tensor3d.cpp         # Tensor2D implementation
│   ├── linear.cpp           # Linear layer implementation
│   ├── relu.cpp             # ReLU layer implementation
│   ├── softmax.cpp          # Softmax layer implementation
│   └── sequential.cpp       # Sequential container implementation
├── examples/
│   └── forward_pass.cpp     # Neural network example
├── tests/
│   └── test_runner.cpp      # Test file
├── build/                   # Used for build artifacts
├── .gitignore
└── README.md
```

## Quick Start

### Run Tests
```bash
# If you want OpenMP support (recommended for parallel features), use Homebrew GCC:
# Install with: brew install gcc
# Then use the versioned g++ (e.g., g++-15) as below:
g++-15 -fopenmp -std=c++17 -Iinclude -Ithird_party/eigen -o build/test_runner \
  tests/test_runner.cpp src/tensor2d.cpp src/tensor3d.cpp src/tensor2d_view.cpp src/linear.cpp src/relu.cpp src/softmax.cpp src/sequential.cpp \
  && ./build/test_runner
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

📖 Full API docs and usage examples → [See demo.md](demo.md)