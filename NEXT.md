# Mini Tensor – Next Steps

Short term
- print() or to_string() for console output (row-major, optional formatting)
- Reductions: sum(), mean(), max() for scalar aggregation
- operator== for test assertions (shape + value equality)

Stretch Goals (July)
- matmul() or operator* for matrix multiplication (dim check + shape logic)
- Views/slicing: row(i), submatrix(i, j, h, w)
- Initializers: Tensor2D::zeros(), from_vector(...) for test ergonomics

Backlog 
- expand() method for explicit broadcast-aware shape expansion  
  - Signature: Tensor2D expand(size_t new_rows, size_t new_cols) const  
  - Enables broadcasting outside of binary ops  
  - Lays foundation for views or derived tensor types
