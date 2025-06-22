# mini-tensor

A lightweight 2D Tensor library in C++.

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

## How to Build and Run Tests

To compile and run the test file:

```bash
g++ -std=c++17 -Iinclude -o build/test_runner tests/test_runner.cpp src/tensor2d.cpp && ./build/test_runner
```

This command does the following:
- `g++`: Invokes the C++ compiler.
- `-std=c++17`: Uses the C++17 standard.
- `-Iinclude`: Tells the compiler to look for header files in the `include` directory.
- `-o build/test_runner`: Specifies the output executable name and location.
- `tests/test_runner.cpp src/tensor2d.cpp`: The source files to compile.
- `&& ./build/test_runner`: Runs the compiled program if the build was successful.
