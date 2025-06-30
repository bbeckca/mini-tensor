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

## MatMul Optimization (Next)
- Integrate Eigen for faster `matmul()` implementation
- Replace nested loops with `Eigen::Map` operations
- Add performance benchmarks and correctness tests
- Optional: compile-time toggle for raw vs Eigen backend

## Export to Simple IR
- Trace operations during forward pass
- Export a minimal IR (e.g., JSON or list of operations)
- Enables inspection and potential future transformations
