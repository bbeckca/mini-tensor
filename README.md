# Mini Tensor

A lightweight 2D tensor library in C++, built to explore PyTorch-style ops, broadcasting, and performance profiling.

## Project Structure

```
mini-tensor/
├── include/
│   └── tensor2d.hpp         # Public interface
├── src/
│   └── tensor2d.cpp         # todo: split public interface
├── tests/
│   └── test_runner.cpp      # Test file
├── build/                   # Used for build artifacts
├── .gitignore
└── README.md
```

## Quick Start

```bash
g++ -std=c++17 -Iinclude -o build/test_runner tests/test_runner.cpp src/tensor2d.cpp && ./build/test_runner

📖 Full API docs and usage examples → [See demo.md](demo.md)

