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

### Run Neural Network Example
```bash
g++ -std=c++17 -Iinclude -Ithird_party/eigen -o build/forward_pass \
examples/forward_pass.cpp src/tensor2d.cpp src/tensor3d.cpp src/tensor2d_view.cpp src/linear.cpp src/relu.cpp src/softmax.cpp src/sequential.cpp && ./build/forward_pass
```

### Run Benchmarks

#### CPU-only
```bash
g++ -std=c++17 -Iinclude -Ithird_party/eigen -o build/benchmark benchmark.cpp src/tensor2d.cpp src/tensor3d.cpp && ./build/benchmark
```

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

## Features

- **2D Tensor Operations**: Element-wise arithmetic, broadcasting, matrix multiplication
- **3D Tensor Operations**: Batched tensors with contiguous memory layout, batched matrix multiplication
- **Neural Network Modules**: Linear layers, ReLU activation, Softmax activation, Sequential containers
- **Forward Pass**: Run input through neural network models
- **Performance**: Contiguous memory layout for efficient cache access; matmul benchmarks included
- **IR Trace**: All Tensor2D operations are tracked in a global IR trace for debugging and introspection
- **Unique Tensor IDs**: Every Tensor2D instance is assigned a unique ID for traceability
- **CUDA Support**: GPU acceleration with device memory management and CUDA kernels for matrix multiplication

## CUDA Support

Tensor2D supports both `Device::CPU` and `Device::GPU` device types with CUDA acceleration for matrix multiplication operations.

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

// GPU-accelerated operations
Tensor2D A = Tensor2D::from_random(1024, 1024, Device::GPU);
Tensor2D B = Tensor2D::from_random(1024, 1024, Device::GPU);
Tensor2D C = mat_mul_cuda(A, B);  // CUDA kernel execution

// Batched operations
Tensor3D batch_A = Tensor3D::from_random(8, 256, 512, Device::GPU);
Tensor3D batch_B = Tensor3D::from_random(8, 512, 128, Device::GPU);
Tensor3D batch_C = bmm_cuda(batch_A, batch_B);  // Batched CUDA kernel execution
```

### Performance Benchmarks

On an NVIDIA T4 instance (GCP) (CPU benchmarks use `mat_mul_eigen()` and `mat_mul_eigen_parallel()`):

#### Matrix Multiplication Performance

| Shape             | CPU Time (ms) | GPU Time (ms) | Speedup |
|------------------|---------------|----------------|---------|
| 512 × 512         | 859.59        | 1.20           | 714.93×  |
| 1024 × 1024       | 6912.91       | 10.32          | 669.61×  |

#### Batched Matrix Multiplication Performance

| Batch × M × K × N         | CPU Time (ms) | GPU Time (ms) | Speedup |
|---------------------------|---------------|----------------|---------|
| 8 × 16 × 16 × 16          | 0.129         | 0.036         | 3.6×    |
| 16 × 64 × 64 × 64         | 0.923         | 0.003         | 292×    |
| 32 × 128 × 128 × 128      | 8.985         | 0.027         | 332×    |
| 8 × 256 × 512 × 128       | 18.751        | 0.040         | 474×    |
| 4 × 512 × 512 × 512       | 142.519       | 0.236         | 603×    |
| 2 × 1024 × 1024 × 1024    | 1,110.492     | 1.607         | 691×    |

#### Device Transfer Performance

| Shape             | CPU → GPU (us) | GPU → CPU (us) | Roundtrip (us) |
|------------------|----------------|----------------|----------------|
| 512 × 512         | 463            | 1374           | 1837           |
| 1024 × 1024       | 1225           | 15726          | 16951          |

**Note**: GPU → CPU transfer is significantly slower due to PCIe bandwidth limits.


## GPU Development via rsync

To sync only source and test files to your remote machine:

```bash
REMOTE_HOST=your-user@your-remote-ip bash scripts/sync_to_remote.sh
```

**Edit `scripts/sync_to_remote.sh` to point to your own GPU box.**

The sync script uses a `.rsync-filter` file to include only essential files:
- `include/` - Header files
- `src/` - Source files  
- `tests/` - Test files
- `scripts/` - Build scripts
- `third_party/eigen/Eigen/` - Core Eigen headers
- `third_party/eigen/unsupported/Eigen/CXX11/Tensor/` - Tensor support headers


## IR Trace

The library automatically tracks all major operations in a global IR trace, including:
- **Arithmetic operators**: `+`, `-`, `*`, `/`
- **Matrix operations**: `mat_mul`, `mat_mul_eigen`, `mat_mul_eigen_parallel`, `mat_mul_cuda`, `bmm_cuda`
- **Element-wise functions**: `abs`, `neg`, `relu`
- **Neural network modules**: `Linear`, `Softmax`, `Sequential`

The IR trace records tensor shapes as `std::variant<std::pair<size_t, size_t>, std::tuple<size_t, size_t, size_t>>` to support both 2D and 3D tensors.

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

📖 **Full API documentation and detailed usage examples** → [See demo.md](demo.md)

The demo.md file contains comprehensive documentation including:
- Complete API reference with code examples
- Detailed CUDA implementation examples  
- Advanced memory management details
- IR trace examples and debugging
- Neural network module usage
- Performance optimization guidelines