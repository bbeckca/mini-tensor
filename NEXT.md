Planned Extensions
This file tracks completed and upcoming features for the tensor library.

## Tensor Slicing and Views (Completed)
- Implemented `Tensor2DView` for non-owning, in-place slicing
- Create views using `Tensor2DView(tensor, row_start, row_end, col_start, col_end)`
- View reflects and updates the base tensor
- Includes shape helpers, accessors, and bounds checking
- Tested and documented

## Broadcasting (Completed)
- Arithmetic operations (`+`, `-`, `*`, `/`) support shape broadcasting
- Handles row and column expansion using `infer_broadcast_shape()`
- Throws an error on incompatible shapes
- Documented with examples and tests

## Forward Pass and Modules (Completed)
- `Module` base class with `forward()` interface
- Includes `Linear`, `ReLU`, `Softmax`, and `Sequential` implementations
- Supports chained execution via `Sequential::forward()`
- Verified with working model in `forward_pass.cpp`

## MatMul Optimization (Completed)
- Integrated Eigen for faster `matmul()` implementation
- Replaced nested loops with `Eigen::Map` operations
- Added performance benchmarks and correctness tests
- Optional: compile-time toggle for raw vs Eigen backend

## Tensor3D and Batched Operations (Completed)
- Implemented `Tensor3D` with contiguous memory layout for GPU compatibility
- Batched matrix multiplication with manual, Eigen, and parallel implementations
- Device memory management with CPU/GPU support
- Broadcasting helper functions `for_each_broadcasted_3d()` and `infer_broadcast_shape_3d()`
- Comprehensive testing and benchmarking

## CUDA Support (Completed)
- GPU acceleration for matrix multiplication operations
- Device memory management with `to(Device)` transfer methods
- Runtime safety checks for CPU-side operations
- CUDA kernels for `mat_mul_cuda()` and `bmm_cuda()`
- Performance benchmarks showing significant speedup

## IR Trace and Tensor IDs (Completed)
- Global IR trace system for tracking all tensor operations
- Unique tensor ID generation and tracking
- Support for both 2D and 3D tensor shapes in trace records
- Comprehensive operation logging for debugging and introspection

## Next: Refactor Tensor3D Operations to Use Broadcast Helper (Completed)
- Convert `-`, `*`, `/` and their in-place versions to use `for_each_broadcasted_3d()`
— Test with shape mismatch, same-shape, and Tensor3D <-> Tensor2D broadcasting
- Replace manual loop implementations with broadcast helper for consistency

## Extend Module Support to Tensor3D (Completed)
- Add `Linear::forward(const Tensor3D&)` for batched linear layers
- Add `ReLU::forward(const Tensor3D&)` for batched activation
- Add `Sequential::forward(const Tensor3D&)` for batched model execution
- Add tests for batched inputs (2–3 examples)

## Set Up GPU Dev Flow with rsync
- Configure local-to-T4 rsync workflow
- Use this instead of pushing commits just to test CUDA
- Streamline GPU development and testing process

## Add CUDA Support for Tensor3D Operations
- `add_cuda()` for same-shape Tensor3D addition
- `bmm_add_cuda()` (fused bmm + bias kernel)
- Benchmark: `bmm_cuda()` + CPU add vs. `bmm_add_cuda()`

## (Optional but Useful)
- Extend IRTrace to record broadcasted shapes
- Benchmark CPU broadcasted ops vs. fused-loop versions (to measure perf gains)
- Tensor3D + float, Tensor3D - float, etc.