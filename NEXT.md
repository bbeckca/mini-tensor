## Planned Extensions

The project is expanding to support basic model execution, better broadcasting, and sub-tensor access. Upcoming features include:

### Forward Pass and Modules

* Add a base `Module` class with a `forward()` method
* Create `Linear` and `ReLU` layers
* Add a `Sequential` container to run layers in order
* Support running input through a simple model

### Broadcasting Improvements

* Add a helper to check compatible shapes
* Use a shared function for `+`, `-`, `*`, and `/`
* Make all arithmetic operations handle broadcasting consistently

### Tensor Slicing and Views

* Add support for getting sub-tensors without copying data
* Let users slice tensors using row and column ranges
* Start with read-only access, later allow updates

### Export to Simple IR Format

* Create a way to record what operations are run
* Each layer can add to the recorded list
* Allow exporting the model as a list or file for inspection
