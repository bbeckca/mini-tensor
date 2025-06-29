# Mini Tensor

A lightweight 2D tensor library in C++ with PyTorch-style modules and forward pass support, to explore neural network internals and performance from first principles

## Project Structure

```
mini-tensor/
├── include/
│   ├── tensor2d.hpp         # Tensor2D public interface
│   ├── module.hpp           # Base Module class
│   ├── linear.hpp           # Linear layer
│   ├── relu.hpp             # ReLU activation layer
│   └── sequential.hpp       # Sequential container
├── src/
│   ├── tensor2d.cpp         # Tensor2D implementation
│   ├── linear.cpp           # Linear layer implementation
│   ├── relu.cpp             # ReLU layer implementation
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
g++ -std=c++17 -Iinclude -o build/test_runner \
tests/test_runner.cpp src/tensor2d.cpp src/linear.cpp src/relu.cpp src/sequential.cpp && ./build/test_runner
```

### Run Neural Network Example
```bash
g++ -std=c++17 -Iinclude -o build/forward_pass \
examples/forward_pass.cpp src/tensor2d.cpp src/linear.cpp src/relu.cpp src/sequential.cpp && ./build/forward_pass
```

## Features

- **2D Tensor Operations**: Element-wise arithmetic, broadcasting, matrix multiplication
- **Neural Network Modules**: Linear layers, ReLU activation, Sequential containers
- **Forward Pass**: Run input through neural network models
- **Performance**: Contiguous memory layout for efficient cache access

📖 Full API docs and usage examples → [See demo.md](demo.md)

