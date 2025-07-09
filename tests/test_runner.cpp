#include "linear.hpp"
#include "relu.hpp"
#include "sequential.hpp"
#include "softmax.hpp"
#include "tensor2d.hpp"
#include "tensor2d_view.hpp"
#include "tensor3d.hpp"
#include "tensor3d.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include "ir_trace.hpp"
#include "matmul_cuda.hpp"


bool shapes_equal(const std::variant<std::pair<size_t, size_t>, std::tuple<size_t, size_t, size_t>>& variant_shape, 
                  const std::pair<size_t, size_t>& pair_shape) {
    return std::visit([&pair_shape](const auto& shape) -> bool {
        if constexpr (std::is_same_v<decltype(shape), const std::pair<size_t, size_t>&>) {
            return shape == pair_shape;
        } else {
            return false;
        }
    }, variant_shape);
}

void test_constructor_with_default_value() {
    std::cout << "Running test: constructor with default value... ";
    Tensor2D t(2, 3, 42.0f);
    for (size_t i = 0; i < t.shape().first; ++i) {
        for (size_t j = 0; j < t.shape().second; ++j) {
            assert(t(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_fill_and_operator_parentheses() {
    std::cout << "Running test: fill() and operator()... ";
    Tensor2D t(2, 3);
    t.fill(5.5f);

    for (size_t i = 0; i < t.shape().first; ++i) {
        for (size_t j = 0; j < t.shape().second; ++j) {
            assert(t(i, j) == 5.5f);
        }
    }

    t(1, 2) = 10.1f;
    assert(t(1, 2) == 10.1f);
    std::cout << "PASSED" << std::endl;
}

void test_from_vector() {
    std::cout << "Running test: from_vector()... ";
    std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor2D t = Tensor2D::from_vector(2, 3, data);
    for (size_t i = 0; i < t.shape().first; ++i) {
        for (size_t j = 0; j < t.shape().second; ++j) {
            assert(t(i, j) == data[i * t.shape().second + j]);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_from_random() {
    std::cout << "Running test: from_random()... ";
    size_t rows = 2, cols = 3;
    Tensor2D t = Tensor2D::from_random(rows, cols);
    assert(t.shape().first == rows);
    assert(t.shape().second == cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            assert(t(i, j) >= 0.0f && t(i, j) <= 1.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_tensor_device() {
    std::cout << "Running test: tensor device... ";
    size_t rows = 2, cols = 3;
    Tensor2D t1(rows, cols, 0.0f, Device::CPU);
    assert(t1.get_device() == Device::CPU);

    #ifdef USE_CUDA
    Tensor2D t2(rows, cols, 0.0f, Device::GPU);
    assert(t2.get_device() == Device::GPU);

    Tensor2D t3 = Tensor2D::from_vector(rows, cols, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Device::GPU);
    assert(t3.get_device() == Device::GPU);

    Tensor2D t4 = Tensor2D::from_random(rows, cols, Device::GPU);
    assert(t4.get_device() == Device::GPU);    
    std::cout << "PASSED" << std::endl;
    #endif
}

void test_shape() {
    std::cout << "Running test: shape()... ";
    Tensor2D t1(4, 5);
    auto shape1 = t1.shape();
    assert(shape1.first == 4);
    assert(shape1.second == 5);

    Tensor2D t2(100, 200);
    auto [rows, cols] = t2.shape();
    assert(rows == 100);
    assert(cols == 200);
    std::cout << "PASSED" << std::endl;
}

void test_reshape() {
    std::cout << "Running test: reshape()... ";
    Tensor2D t1(2, 2);
    t1.reshape(4, 1);
    assert(t1.shape().first == 4);
    assert(t1.shape().second == 1);
    std::cout << "PASSED" << std::endl;
}

void test_infer_broadcast_shape() {
    std::cout << "Running test: infer_broadcast_shape()... ";
    Tensor2D t1(2, 1);
    Tensor2D t2(1, 5);
    auto shape = t1.infer_broadcast_shape(t1.shape(), t2.shape());
    assert(shape.first == 2);
    assert(shape.second == 5);
    std::cout << "PASSED" << std::endl;
}

void test_row_wise_expand() {
    std::cout << "Running test: row-wise expand()... ";
    size_t rows = 1, cols = 3;
    Tensor2D t1 = Tensor2D::from_vector(rows, cols, {1.0f, 2.0f, 3.0f});
    Tensor2D t2 = t1.expand(2, 3);
    assert(t2.shape().first == 2);
    assert(t2.shape().second == 3);
    for (size_t i = 0; i < t2.shape().first; ++i) {
        for (size_t j = 0; j < t2.shape().second; ++j) {
            assert(t1(i % rows, j % cols) == t2(i, j));
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_column_wise_expand() {
    std::cout << "Running test: column-wise expand()... ";
    size_t rows = 1, cols = 3;
    Tensor2D t1 = Tensor2D::from_vector(rows, cols, {1.0f, 2.0f, 3.0f});
    Tensor2D t2 = t1.expand(2, 3);
    assert(t2.shape().first == 2);
    assert(t2.shape().second == 3);
    for (size_t i = 0; i < t2.shape().first; ++i) {
        for (size_t j = 0; j < t2.shape().second; ++j) {
            assert(t1(i % rows, j % cols) == t2(i, j));
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_expand_with_incompatible_shapes() {
    std::cout << "Running test: expand() with incompatible shapes... ";
    Tensor2D t1(2, 2);
    t1.fill(42.0f);
    bool exception_thrown = false;
    try {
        Tensor2D t2 = t1.expand(3, 3);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_tensor_equality() {
    std::cout << "Running test: operator==... ";
    Tensor2D t1(2, 2, 42.0f);
    Tensor2D t2(2, 2, 42.0f);
    assert(t1 == t2);
    std::cout << "PASSED" << std::endl;
}

void test_tensor_inequality() {
    std::cout << "Running test: operator!=... ";
    Tensor2D t1(2, 2, 42.0f);
    Tensor2D t2(2, 2, 42.0f);
    t2(0, 0) = 1.0f;
    assert(t1 != t2);
    std::cout << "PASSED" << std::endl;
}

void test_addition_with_same_shapes() {
    std::cout << "Running test: operator+... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);

    Tensor2D t2(2, 2);
    t2.fill(41.0f);

    Tensor2D t3 = t1 + t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_addition_with_compatible_broadcast_shapes() {
    std::cout << "Running test: operator+ with broadcast compatible shapes... ";
    Tensor2D t1(1, 5);
    t1.fill(1.0f);
    Tensor2D t2(3, 1);
    t2.fill(41.0f);

    Tensor2D t3 = t1 + t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_addition_with_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator+ with incompatible broadcast shapes... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);

    Tensor2D t2(3, 2);
    t2.fill(41.0f);

    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1 + t2;
    } catch (const std::invalid_argument& e) { 
        exception_thrown = true;
    }
    assert(exception_thrown);
        std::cout << "PASSED" << std::endl;
}

void test_addition_in_place_with_same_shapes() {
    std::cout << "Running test: operator+=... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);
    Tensor2D t2(2, 2);
    t2.fill(41.0f);

    t1 += t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }

    std::cout << "PASSED" << std::endl;
}

void test_addition_in_place_compatible_broadcast_shapes() {
    std::cout << "Running test: operator+= with compatible broadcast shapes... ";
    Tensor2D t1(3, 5);
    t1.fill(1.0f);
    Tensor2D t2(3, 1);
    t2.fill(41.0f);

    t1 += t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_addition_in_place_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator+= with incompatible broadcast shapes... ";
    Tensor2D t1(2, 3);
    Tensor2D t2(3, 2);
    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1 + t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}


void test_addition_with_scalar() {
    std::cout << "Running test: operator+ with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);

    Tensor2D t2 = t1 + 41.0f;

    for (size_t i = 0; i < t2.shape().first; ++i) {
        for (size_t j = 0; j < t2.shape().second; ++j) {
            assert(t2(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_addition_with_scalar_in_place() {
    std::cout << "Running test: operator+= with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);

    t1 += 41.0f;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_with_same_shapes() {
    std::cout << "Running test: operator-... ";
    Tensor2D t1(2, 2);
    t1.fill(43.0f);

    Tensor2D t2(2, 2);
    t2.fill(1.0f);

    Tensor2D t3 = t1 - t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_with_compatible_broadcast_shapes() {
    std::cout << "Running test: operator- with broadcast compatible shapes... ";
    Tensor2D t1(1, 5);
    t1.fill(43.0f);
    Tensor2D t2(3, 1);
    t2.fill(1.0f);

    Tensor2D t3 = t1 - t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_with_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator- with incompatible broadcast shapes... ";
    Tensor2D t1(2, 2);
    t1.fill(43.0f);

    Tensor2D t2(3, 2);
    t2.fill(1.0f);

    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1 - t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_in_place_with_same_shapes() {
    std::cout << "Running test: operator-=... ";
    Tensor2D t1(2, 2);
    t1.fill(43.0f);
    Tensor2D t2(2, 2);
    t2.fill(1.0f);

    t1 -= t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }

    std::cout << "PASSED" << std::endl;
}

void test_subtraction_in_place_compatible_broadcast_shapes() {
    std::cout << "Running test: operator-= with compatible broadcast shapes... ";
    Tensor2D t1(3, 5);
    t1.fill(43.0f);
    Tensor2D t2(3, 1);
    t2.fill(1.0f);

    t1 -= t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_in_place_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator-= with incompatible broadcast shapes... ";
    Tensor2D t1(2, 3);
    Tensor2D t2(3, 2);
    bool exception_thrown = false;
    try {
        t1 -= t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_with_scalar() {
    std::cout << "Running test: operator- with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(43.0f);

    Tensor2D t2 = t1 - 1.0f;

    for (size_t i = 0; i < t2.shape().first; ++i) {
        for (size_t j = 0; j < t2.shape().second; ++j) {
            assert(t2(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_subtraction_with_scalar_in_place() {
    std::cout << "Running test: operator-= with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(43.0f);

    t1 -= 1.0f;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}


void test_relu() {
    std::cout << "Running test: relu()... ";
    Tensor2D t1(1, 3);
    t1(0, 0) = -1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = -3.0f;

    Tensor2D t2 = t1.relu();
    assert(t2(0, 0) == 0.0f);
    assert(t2(0, 1) == 2.0f);
    assert(t2(0, 2) == 0.0f);
    std::cout << "PASSED" << std::endl;
}

void test_relu_in_place() {
    std::cout << "Running test: relu() in place... ";
    Tensor2D t1(1, 3);
    t1(0, 0) = -1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = -3.0f;

    t1.relu_in_place();
    assert(t1(0, 0) == 0.0f);
    assert(t1(0, 1) == 2.0f);
    assert(t1(0, 2) == 0.0f);
    std::cout << "PASSED" << std::endl;
}

void test_negate() {
    std::cout << "Running test: negate()... ";
    Tensor2D t1(1, 3);
    t1(0, 0) = -1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = -3.0f;

    Tensor2D t2 = t1.negate();
    assert(t2(0, 0) == 1.0f);
    assert(t2(0, 1) == -2.0f);
    assert(t2(0, 2) == 3.0f);
    std::cout << "PASSED" << std::endl;
}

void test_negate_in_place() {
    std::cout << "Running test: negate() in place... ";
    Tensor2D t1(1, 3);
    t1(0, 0) = -1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = -3.0f;

    t1.negate_in_place();
    assert(t1(0, 0) == 1.0f);
    assert(t1(0, 1) == -2.0f);
    assert(t1(0, 2) == 3.0f);
    std::cout << "PASSED" << std::endl;
}

void test_abs() {
    std::cout << "Running test: abs()... ";
    Tensor2D t1(1, 3);
    t1(0, 0) = -1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = -3.0f;

    Tensor2D t2 = t1.abs();
    assert(t2(0, 0) == 1.0f);
    assert(t2(0, 1) == 2.0f);
    assert(t2(0, 2) == 3.0f);
    std::cout << "PASSED" << std::endl;
}

void test_abs_in_place() {
    std::cout << "Running test: abs() in place... ";
    Tensor2D t1(1, 3);
    t1(0, 0) = -1.0f;
    t1(0, 1) = 2.0f;
    t1(0, 2) = -3.0f;

    t1.abs_in_place();
    assert(t1(0, 0) == 1.0f);
    assert(t1(0, 1) == 2.0f);
    assert(t1(0, 2) == 3.0f);
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_with_same_shapes() {
    std::cout << "Running test: operator*... ";
    Tensor2D t1(2, 2);
    t1.fill(6.0f);

    Tensor2D t2(2, 2);
    t2.fill(7.0f);

    Tensor2D t3 = t1 * t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_with_compatible_broadcast_shapes() {
    std::cout << "Running test: operator* with broadcast compatible shapes... ";
    Tensor2D t1(1, 5);
    t1.fill(6.0f);
    Tensor2D t2(3, 1);
    t2.fill(7.0f);

    Tensor2D t3 = t1 * t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_with_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator* with incompatible broadcast shapes... ";
    Tensor2D t1(2, 2);
    t1.fill(6.0f);

    Tensor2D t2(3, 2);
    t2.fill(7.0f);

    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1 * t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_in_place_with_same_shapes() {
    std::cout << "Running test: operator*=... ";
    Tensor2D t1(2, 2);
    t1.fill(6.0f);
    Tensor2D t2(2, 2);
    t2.fill(7.0f);

    t1 *= t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_in_place_compatible_broadcast_shapes() {
    std::cout << "Running test: operator*= with compatible broadcast shapes... ";
    Tensor2D t1(3, 5);
    t1.fill(6.0f);
    Tensor2D t2(3, 1);
    t2.fill(7.0f);

    t1 *= t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_in_place_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator*= with incompatible broadcast shapes... ";
    Tensor2D t1(2, 3);
    Tensor2D t2(3, 2);
    bool exception_thrown = false;
    try {
        t1 *= t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_with_scalar() {
    std::cout << "Running test: operator* with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(6.0f);

    Tensor2D t2 = t1 * 7.0f;

    for (size_t i = 0; i < t2.shape().first; ++i) {
        for (size_t j = 0; j < t2.shape().second; ++j) {
            assert(t2(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_multiplication_with_scalar_in_place() {
    std::cout << "Running test: operator*= with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(6.0f);

    t1 *= 7.0f;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_division_with_same_shapes() {
    std::cout << "Running test: operator/... ";
    Tensor2D t1(2, 2);
    t1.fill(84.0f);

    Tensor2D t2(2, 2);
    t2.fill(2.0f);

    Tensor2D t3 = t1 / t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_division_with_compatible_broadcast_shapes() {
    std::cout << "Running test: operator/ with broadcast compatible shapes... ";
    Tensor2D t1(1, 5);
    t1.fill(84.0f);
    Tensor2D t2(3, 1);
    t2.fill(2.0f);

    Tensor2D t3 = t1 / t2;

    for (size_t i = 0; i < t3.shape().first; ++i) {
        for (size_t j = 0; j < t3.shape().second; ++j) {
            assert(t3(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_division_with_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator/ with incompatible broadcast shapes... ";
    Tensor2D t1(2, 2);
    t1.fill(84.0f);

    Tensor2D t2(3, 2);
    t2.fill(2.0f);

    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1 / t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_division_in_place_with_same_shapes() {
    std::cout << "Running test: operator/=... ";
    Tensor2D t1(2, 2);
    t1.fill(84.0f);
    Tensor2D t2(2, 2);
    t2.fill(2.0f);

    t1 /= t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_division_in_place_compatible_broadcast_shapes() {
    std::cout << "Running test: operator/= with compatible broadcast shapes... ";
    Tensor2D t1(3, 5);
    t1.fill(84.0f);
    Tensor2D t2(3, 1);
    t2.fill(2.0f);

    t1 /= t2;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_division_in_place_incompatible_broadcast_shapes() {
    std::cout << "Running test: operator/= with incompatible broadcast shapes... ";
    Tensor2D t1(2, 3);
    Tensor2D t2(3, 2);
    bool exception_thrown = false;
    try {
        t1 /= t2;
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_division_with_scalar() {
    std::cout << "Running test: operator/ with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(84.0f);

    Tensor2D t2 = t1 / 2.0f;

    for (size_t i = 0; i < t2.shape().first; ++i) {
        for (size_t j = 0; j < t2.shape().second; ++j) {
            assert(t2(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_division_with_scalar_in_place() {
    std::cout << "Running test: operator/= with scalar... ";
    Tensor2D t1(2, 2);
    t1.fill(84.0f);

    t1 /= 2.0f;

    for (size_t i = 0; i < t1.shape().first; ++i) {
        for (size_t j = 0; j < t1.shape().second; ++j) {
            assert(t1(i, j) == 42.0f);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_sum() {
    std::cout << "Running test: sum()... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);

    float sum = t1.sum();
    assert(sum == 4.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mean() {
    std::cout << "Running test: mean()... ";
    Tensor2D t1(2, 2);
    t1.fill(42.0f);

    float mean = t1.mean();
    assert(mean == 42.0f);
    std::cout << "PASSED" << std::endl;
}

void test_max() {
    std::cout << "Running test: max()... ";
    Tensor2D t1(2, 2);
    t1.fill(1.0f);
    t1(0, 0) = 42.0f;

    float max = t1.max();
    assert(max == 42.0f);
    std::cout << "PASSED" << std::endl;
}

void test_arg_max() {
    std::cout << "Running test: arg_max()... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 42.0f, 4.0f});
    std::pair<size_t, size_t> arg_max = t1.arg_max();
    assert(arg_max.first == 1);
    assert(arg_max.second == 0);
    std::cout << "PASSED" << std::endl;
}

void test_arg_max_with_multiple_max_values() {
    std::cout << "Running test: arg_max() with multiple max values... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 42.0f, 42.0f});
    std::pair<size_t, size_t> arg_max = t1.arg_max();
    assert(arg_max.first == 1);
    assert(arg_max.second == 0);
    std::cout << "PASSED" << std::endl;
}

void test_arg_max_with_all_equal() {
    std::cout << "Running test: arg_max() with all equal values... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 1.0f, 1.0f, 1.0f});
    std::pair<size_t, size_t> arg_max = t1.arg_max();
    assert(arg_max.first == 0);
    assert(arg_max.second == 0);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_with_same_shapes() {
    std::cout << "Running test: mat_mul() with same shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor2D t3 = t1.mat_mul(t2);
    assert(t3(0, 0) == 19.0f);
    assert(t3(0, 1) == 22.0f);
    assert(t3(1, 0) == 43.0f);
    assert(t3(1, 1) == 50.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_with_compatible_shapes() {
    std::cout << "Running test: mat_mul() with compatible shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor2D t2 = Tensor2D::from_vector(3, 1, {1.0f, 0.0f, -1.0f});
    Tensor2D t3 = t1.mat_mul(t2);
    assert(t3(0, 0) == -2.0f);
    assert(t3(1, 0) == -2.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_with_identity_matrix() {
    std::cout << "Running test: mat_mul() with identity matrix... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor2D t3 = t1.mat_mul(t2);
    assert(t3(0, 0) == 1.0f);
    assert(t3(0, 1) == 2.0f);
    assert(t3(1, 0) == 3.0f);
    assert(t3(1, 1) == 4.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_with_incompatible_shapes() {
    std::cout << "Running test: mat_mul() with incompatible shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D t2 = Tensor2D::from_vector(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1.mat_mul(t2);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_eigen_with_same_shapes() {
    std::cout << "Running test: mat_mul_eigen() with same shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor2D t3 = t1.mat_mul_eigen(t2);
    assert(t3(0, 0) == 19.0f);
    assert(t3(0, 1) == 22.0f);
    assert(t3(1, 0) == 43.0f);
    assert(t3(1, 1) == 50.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_eigen_with_compatible_shapes() {
    std::cout << "Running test: mat_mul_eigen() with compatible shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor2D t2 = Tensor2D::from_vector(3, 1, {1.0f, 0.0f, -1.0f});
    Tensor2D t3 = t1.mat_mul_eigen(t2);
    assert(t3(0, 0) == -2.0f);
    assert(t3(1, 0) == -2.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_eigen_with_identity_matrix() {
    std::cout << "Running test: mat_mul_eigen() with identity matrix... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor2D t3 = t1.mat_mul_eigen(t2);
    assert(t3(0, 0) == 1.0f);
    assert(t3(0, 1) == 2.0f);
    assert(t3(1, 0) == 3.0f);
    assert(t3(1, 1) == 4.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_eigen_with_incompatible_shapes() {
    std::cout << "Running test: mat_mul_eigen() with incompatible shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor2D t2 = Tensor2D::from_vector(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    bool exception_thrown = false;
    try {
        Tensor2D t3 = t1.mat_mul_eigen(t2);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

#ifdef USE_CUDA
void test_mat_mul_cuda_with_same_shapes() {
    std::cout << "Running test: mat_mul_cuda() with same shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, Device::GPU);
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f}, Device::GPU);
    Tensor2D t3 = mat_mul_cuda(t1, t2);
    assert(t3.get_device() == Device::GPU);
    Tensor2D t3_cpu = t3.to(Device::CPU);
    assert(t3_cpu(0, 0) == 19.0f);
    assert(t3_cpu(0, 1) == 22.0f);
    assert(t3_cpu(1, 0) == 43.0f);
    assert(t3_cpu(1, 1) == 50.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_cuda_with_compatible_shapes() {
    std::cout << "Running test: mat_mul_cuda() with compatible shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Device::GPU);
    Tensor2D t2 = Tensor2D::from_vector(3, 1, {1.0f, 0.0f, -1.0f}, Device::GPU);
    Tensor2D t3 = mat_mul_cuda(t1, t2);
    assert(t3.get_device() == Device::GPU);
    Tensor2D t3_cpu = t3.to(Device::CPU);
    assert(t3_cpu(0, 0) == -2.0f);
    assert(t3_cpu(1, 0) == -2.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_cuda_with_identity_matrix() {
    std::cout << "Running test: mat_mul_cuda() with identity matrix... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, Device::GPU);
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}, Device::GPU);
    Tensor2D t3 = mat_mul_cuda(t1, t2);
    assert(t3.get_device() == Device::GPU);
    Tensor2D t3_cpu = t3.to(Device::CPU);
    assert(t3_cpu(0, 0) == 1.0f);
    assert(t3_cpu(0, 1) == 2.0f);
    assert(t3_cpu(1, 0) == 3.0f);
    assert(t3_cpu(1, 1) == 4.0f);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_cuda_with_incompatible_shapes() {
    std::cout << "Running test: mat_mul_cuda() with incompatible shapes... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, Device::GPU);
    Tensor2D t2 = Tensor2D::from_vector(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Device::GPU);
    bool exception_thrown = false;
    try {
        Tensor2D t3 = mat_mul_cuda(t1, t2);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_mat_mul_cuda_with_incompatible_device() {
    std::cout << "Running test: mat_mul_cuda() with incompatible device... ";
    Tensor2D t1 = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, Device::CPU);
    Tensor2D t2 = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f}, Device::GPU);
    bool exception_thrown = false;
    try {
        Tensor2D t3 = mat_mul_cuda(t1, t2);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_bmm_cuda_with_same_shapes() {
    std::cout << "Running test: bmm_cuda() with same shapes... ";
    Tensor3D t1 = Tensor3D::from_vector(2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Device::GPU);
    Tensor3D t2 = Tensor3D::from_vector(2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Device::GPU);
    
    Tensor3D t3 = bmm_cuda(t1, t2);
    assert(t3.get_device() == Device::GPU);
    Tensor3D t3_cpu = t3.to(Device::CPU);
    
    assert(t3_cpu[0](0, 0) == 7.0f);
    assert(t3_cpu[0](0, 1) == 10.0f);
    assert(t3_cpu[0](1, 0) == 15.0f);
    assert(t3_cpu[0](1, 1) == 22.0f);
    assert(t3_cpu[1](0, 0) == 67.0f);
    assert(t3_cpu[1](0, 1) == 78.0f);
    assert(t3_cpu[1](1, 0) == 91.0f);
    assert(t3_cpu[1](1, 1) == 106.0f);
    std::cout << "PASSED" << std::endl;
}
#endif

void test_linear_forward() {
    std::cout << "Running test: Linear forward()... ";
    Linear linear(2, 2);
    linear.set_weights(Tensor2D::from_vector(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}));
    linear.set_bias(Tensor2D::from_vector(1, 2, {1.0f, 2.0f}));
    Tensor2D input = Tensor2D::from_vector(1, 2, {3.0f, 4.0f});
    Tensor2D output = linear.forward(input);
    assert(output(0, 0) == 4.0f);
    assert(output(0, 1) == 6.0f);
    std::cout << "PASSED" << std::endl;
}

void test_relu_forward() {
    std::cout << "Running test: ReLU forward()... ";
    ReLU relu;
    Tensor2D input = Tensor2D::from_vector(1, 4, {-1.0f, 0.0f, 2.0f, 3.0f});
    Tensor2D output = relu.forward(input);
    assert(output(0, 0) == 0.0f);
    assert(output(0, 1) == 0.0f);
    assert(output(0, 2) == 2.0f);
    assert(output(0, 3) == 3.0f);
    std::cout << "PASSED" << std::endl;
}

void test_sequential_forward() {
    std::cout << "Running test: Sequential forward()... ";
    Sequential sequential;
    
    auto linear = std::make_unique<Linear>(2, 2);
    linear->set_weights(Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}));
    linear->set_bias(Tensor2D::from_vector(1, 2, {0.0f, 0.0f}));

    sequential.add(std::move(linear));
    sequential.add(std::make_unique<ReLU>());

    Tensor2D input = Tensor2D::from_vector(1, 2, {1.0f, 1.0f});
    Tensor2D output = sequential.forward(input);

    assert(output(0, 0) == 4.0f);
    assert(output(0, 1) == 6.0f);
    std::cout << "PASSED" << std::endl;
}

void test_softmax_forward() {
    std::cout << "Running test: Softmax forward()... ";
    Softmax softmax;

    Tensor2D input = Tensor2D::from_vector(2, 3, {
        1.0f, 2.0f, 3.0f,    // row 0 (deterministic)
        0.0f, 0.0f, 0.0f     // row 1 (uniform)
    });

    Tensor2D output = softmax.forward(input);
    assert(output.shape() == std::make_pair(2UL, 3UL));

    // Check relative ordering in row 0: softmax(3) > softmax(2) > softmax(1)
    assert(output(0, 2) > output(0, 1));
    assert(output(0, 1) > output(0, 0));

    // Check uniformity in row 1: softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
    float expected = 1.0f / 3.0f;
    for (size_t j = 0; j < 3; ++j) {
        assert(std::abs(output(1, j) - expected) < 1e-4f);
    }

    // Check that each row sums to ~1
    for (size_t i = 0; i < 2; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            row_sum += output(i, j);
        }
        assert(std::abs(row_sum - 1.0f) < 1e-4f);
    }

    std::cout << "PASSED" << std::endl;
}

void test_tensor2d_view() {
    std::cout << "Running test: Tensor2DView... ";
    size_t rows = 4, cols = 4;
    Tensor2D base(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            base(i, j) = i * cols + j;
        }
    }
    
    Tensor2DView view(base, 1, 3, 1, 3);
    assert(view.rows() == 2);
    assert(view.cols() == 2);
    for (size_t i = 0; i < view.rows(); ++i) {
        for (size_t j = 0; j < view.cols(); ++j) {
            assert(view(i, j) == base(i + 1, j + 1));
        }
    }

    view(0, 0) = 42.0f;
    assert(base(1, 1) == 42.0f);

    std::cout << "PASSED" << std::endl;
}

void test_tensor2d_view_out_of_bounds() {
    std::cout << "Running test: Tensor2DView out of bounds... ";
    Tensor2D base(4, 4);
    bool exception_thrown = false;
    try {
        Tensor2DView view(base, 1, 3, 1, 6);
    } catch (const std::out_of_range& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_constructor_and_shape() {
    std::cout << "Running test: Tensor3D constructor and shape... ";
    Tensor3D t3d(2, 3, 4, 7.0f);
    assert(t3d.batch_size() == 2);
    assert(t3d.rows() == 3);
    assert(t3d.cols() == 4);
    for (size_t b = 0; b < t3d.batch_size(); ++b) {
        for (size_t i = 0; i < t3d.rows(); ++i) {
            for (size_t j = 0; j < t3d.cols(); ++j) {
                assert(t3d[b](i, j) == 7.0f);
            }
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_element_access_and_mutation() {
    std::cout << "Running test: Tensor3D element access and mutation... ";
    Tensor3D t3d(2, 2, 2, 0.0f);
    t3d[0](0, 0) = 1.0f;
    t3d[1](1, 1) = 2.0f;
    assert(t3d[0](0, 0) == 1.0f);
    assert(t3d[1](1, 1) == 2.0f);
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_getters() {
    std::cout << "Running test: Tensor3D getters... ";
    Tensor3D t3d(2, 3, 4, 7.0f, Device::CPU);

    auto shape = t3d.shape();
    assert(std::get<0>(shape) == 2);
    assert(std::get<1>(shape) == 3);
    assert(std::get<2>(shape) == 4);

    assert(t3d.get_device() == Device::CPU);
    assert(!t3d.get_id().empty());
    assert(t3d.data() != nullptr);

    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_from_random() {
    std::cout << "Running test: Tensor3D from_random... ";
    Tensor3D t3d = Tensor3D::from_random(2, 2, 2);
    assert(t3d.batch_size() == 2);
    assert(t3d.rows() == 2);
    assert(t3d.cols() == 2);
    for (size_t b = 0; b < t3d.batch_size(); ++b) {
        for (size_t i = 0; i < t3d.rows(); ++i) {
            for (size_t j = 0; j < t3d.cols(); ++j) {
                assert(t3d[b](i, j) >= 0.0f && t3d[b](i, j) <= 1.0f);
            }
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_slice_batch() {
    std::cout << "Running test: Tensor3D slice_batch... ";
    
    Tensor3D t3d = Tensor3D::from_vector(2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    
    Tensor2D slice0 = t3d.slice_batch(0);
    assert(slice0.is_view());
    assert(slice0.rows() == 2);
    assert(slice0.cols() == 2);
    
    assert(slice0(0, 0) == 1.0f);
    assert(slice0(0, 1) == 2.0f);
    assert(slice0(1, 0) == 3.0f);
    assert(slice0(1, 1) == 4.0f);
    
    Tensor2D slice1 = t3d.slice_batch(1);
    assert(slice1.is_view());
    assert(slice1.rows() == 2);
    assert(slice1.cols() == 2);
    
    assert(slice1(0, 0) == 5.0f);
    assert(slice1(0, 1) == 6.0f);
    assert(slice1(1, 0) == 7.0f);
    assert(slice1(1, 1) == 8.0f);
    
    slice0(0, 0) = 42.0f;
    assert(t3d[0](0, 0) == 42.0f);
    assert(slice0(0, 0) == 42.0f);
    
    assert(slice0.data() == t3d.data() + 0 * 2 * 2);
    assert(slice1.data() == t3d.data() + 1 * 2 * 2);
    
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_mat_mul() {
    std::cout << "Running test: Tensor3D mat_mul... ";

    Tensor3D t1 = Tensor3D::from_vector(2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor3D t2 = Tensor3D::from_vector(2, 2, 2, {5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    Tensor3D t3 = t1.mat_mul(t2);

    // Expected t3[0] = t1[0] * t2[0] = [ [19 22], [43 50] ]
    assert(t3[0](0, 0) == 19.0f); // 1*5 + 2*7
    assert(t3[0](0, 1) == 22.0f); // 1*6 + 2*8
    assert(t3[0](1, 0) == 43.0f); // 3*5 + 4*7
    assert(t3[0](1, 1) == 50.0f); // 3*6 + 4*8

    // Expected t3[1] = t1[1] * t2[1] = [ [267 286], [323 346] ]
    assert(t3[1](0, 0) == 267.0f); // 9*13 + 10*15
    assert(t3[1](0, 1) == 286.0f); // 9*14 + 10*16
    assert(t3[1](1, 0) == 323.0f); // 11*13 + 12*15
    assert(t3[1](1, 1) == 346.0f); // 11*14 + 12*16

    std::cout << "PASSED\n";
}

void test_tensor3d_mat_mul_eigen() {
    std::cout << "Running test: Tensor3D mat_mul_eigen... ";

    Tensor3D t1 = Tensor3D::from_vector(2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor3D t2 = Tensor3D::from_vector(2, 2, 2, {5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    Tensor3D t3 = t1.mat_mul_eigen(t2);

    // Expected t3[0] = t1[0] * t2[0] = [ [19 22], [43 50] ]
    assert(t3[0](0, 0) == 19.0f);
    assert(t3[0](0, 1) == 22.0f);
    assert(t3[0](1, 0) == 43.0f);
    assert(t3[0](1, 1) == 50.0f);

    // Expected t3[1] = t1[1] * t2[1] = [ [267 286], [323 346] ]
    assert(t3[1](0, 0) == 267.0f);
    assert(t3[1](0, 1) == 286.0f);
    assert(t3[1](1, 0) == 323.0f);
    assert(t3[1](1, 1) == 346.0f);

    std::cout << "PASSED\n";
}

void test_tensor3d_mat_mul_eigen_parallel() {
    std::cout << "Running test: Tensor3D mat_mul_eigen_parallel... ";

    Tensor3D t1 = Tensor3D::from_vector(2, 2, 2, {1.0f, 2.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f});
    Tensor3D t2 = Tensor3D::from_vector(2, 2, 2, {5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    Tensor3D t3 = t1.mat_mul_eigen_parallel(t2);

    // Expected t3[0] = t1[0] * t2[0] = [ [19 22], [43 50] ]
    assert(t3[0](0, 0) == 19.0f);
    assert(t3[0](0, 1) == 22.0f);
    assert(t3[0](1, 0) == 43.0f);
    assert(t3[0](1, 1) == 50.0f);

    // Expected t3[1] = t1[1] * t2[1] = [ [267 286], [323 346] ]
    assert(t3[1](0, 0) == 267.0f);
    assert(t3[1](0, 1) == 286.0f);
    assert(t3[1](1, 0) == 323.0f);
    assert(t3[1](1, 1) == 346.0f);

    std::cout << "PASSED\n";
}

void test_tensor3d_addition_with_same_shapes() {
    std::cout << "Running test: Tensor3D addition with same shapes... ";
    Tensor3D t1(2, 2, 2, 41.0f);
    Tensor3D t2(2, 2, 2, 1.0f);

    Tensor3D t3 = t1 + t2;

    for (size_t b = 0; b < t3.batch_size(); ++b) {
        for (size_t i = 0; i < t3.rows(); ++i) {
            for (size_t j = 0; j < t3.cols(); ++j) {
                assert(t3[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_tensor3d_addition_in_place_with_same_shapes() {
    std::cout << "Running test: Tensor3D addition in place with same shapes... ";
    Tensor3D t1(2, 2, 2, 41.0f);
    Tensor3D t2(2, 2, 2, 1.0f);

    t1 += t2;

    for (size_t b = 0; b < t1.batch_size(); ++b) {
        for (size_t i = 0; i < t1.rows(); ++i) {
            for (size_t j = 0; j < t1.cols(); ++j) {
                assert(t1[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_tensor3d_subtraction_with_same_shapes() {
    std::cout << "Running test: Tensor3D subtraction with same shapes... ";
    Tensor3D t1(2, 2, 2, 43.0f);
    Tensor3D t2(2, 2, 2, 1.0f);

    Tensor3D t3 = t1 - t2;

    for (size_t b = 0; b < t3.batch_size(); ++b) {
        for (size_t i = 0; i < t3.rows(); ++i) {
            for (size_t j = 0; j < t3.cols(); ++j) {
                assert(t3[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_tensor3d_subtraction_in_place_with_same_shapes() {
    std::cout << "Running test: Tensor3D subtraction in place with same shapes... ";
    Tensor3D t1(2, 2, 2, 43.0f);
    Tensor3D t2(2, 2, 2, 1.0f);

    t1 -= t2;

    for (size_t b = 0; b < t1.batch_size(); ++b) {
        for (size_t i = 0; i < t1.rows(); ++i) {
            for (size_t j = 0; j < t1.cols(); ++j) {
                assert(t1[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_tensor3d_multiplication_with_same_shapes() {
    std::cout << "Running test: Tensor3D multiplication with same shapes... ";
    Tensor3D t1(2, 2, 2, 6.0f);
    Tensor3D t2(2, 2, 2, 7.0f);

    Tensor3D t3 = t1 * t2;

    for (size_t b = 0; b < t3.batch_size(); ++b) {
        for (size_t i = 0; i < t3.rows(); ++i) {
            for (size_t j = 0; j < t3.cols(); ++j) {
                assert(t3[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED" << std::endl;
}   

void test_tensor3d_multiplication_in_place_with_same_shapes() {
    std::cout << "Running test: Tensor3D multiplication in place with same shapes... ";
    Tensor3D t1(2, 2, 2, 6.0f);
    Tensor3D t2(2, 2, 2, 7.0f);

    t1 *= t2;

    for (size_t b = 0; b < t1.batch_size(); ++b) {
        for (size_t i = 0; i < t1.rows(); ++i) {
            for (size_t j = 0; j < t1.cols(); ++j) {
                assert(t1[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_tensor3d_division_with_same_shapes() {
    std::cout << "Running test: Tensor3D division with same shapes... ";
    Tensor3D t1(2, 2, 2, 84.0f);
    Tensor3D t2(2, 2, 2, 2.0f);

    Tensor3D t3 = t1 / t2;

    for (size_t b = 0; b < t3.batch_size(); ++b) {
        for (size_t i = 0; i < t3.rows(); ++i) {
            for (size_t j = 0; j < t3.cols(); ++j) {
                assert(t3[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_tensor3d_division_in_place_with_same_shapes() {
    std::cout << "Running test: Tensor3D division in place with same shapes... ";
    Tensor3D t1(2, 2, 2, 84.0f);
    Tensor3D t2(2, 2, 2, 2.0f);

    t1 /= t2;

    for (size_t b = 0; b < t1.batch_size(); ++b) {
        for (size_t i = 0; i < t1.rows(); ++i) {
            for (size_t j = 0; j < t1.cols(); ++j) {
                assert(t1[b](i, j) == 42.0f);
            }
        }
    }
    std::cout << "PASSED\n";
}

void test_ir_trace_for_arithmetic_operators() {
    std::cout << "Running test: IRTrace for arithmetic operators... ";
    TensorID::reset();
    IRTrace::reset();
    size_t rows = 2, cols = 2;

    Tensor2D a = Tensor2D::from_random(rows, cols);
    Tensor2D b = Tensor2D::from_random(rows, cols);
    Tensor2D c = a + b;
    Tensor2D d = a - b;
    Tensor2D e = a * b;
    Tensor2D f = a / b;

    assert(IRTrace::size() == 4);

    auto ops = IRTrace::get_ops();
    assert(ops[0].op_name == "operator+");
    assert(ops[1].op_name == "operator-");
    assert(ops[2].op_name == "operator*");
    assert(ops[3].op_name == "operator/");
    assert(shapes_equal(ops[3].shape, std::make_pair(rows, cols)));
    assert(ops[3].device == Device::CPU);

    std::cout << "PASSED" << std::endl;
}

void test_ir_trace_for_operators() {
    std::cout << "Running test: IRTrace for operators... ";
    TensorID::reset();
    IRTrace::reset();

    Tensor2D a = Tensor2D::from_random(2, 3);
    Tensor2D b = Tensor2D::from_random(3, 2);
    Tensor2D c = a.mat_mul(b);
    Tensor2D d = c.relu();

    assert(IRTrace::size() == 2);

    auto ops = IRTrace::get_ops();
    assert(ops[0].op_name == "mat_mul");
    assert(ops[0].inputs[0] == a.get_id());
    assert(ops[0].inputs[1] == b.get_id());
    assert(ops[0].output == c.get_id());
    assert(shapes_equal(ops[0].shape, std::make_pair(2UL, 2UL)));
    assert(ops[0].device == Device::CPU);

    assert(ops[1].op_name == "relu");
    assert(ops[1].inputs[0] == c.get_id());
    assert(ops[1].output == d.get_id());
    assert(shapes_equal(ops[1].shape, std::make_pair(2UL, 2UL)));
    assert(ops[1].device == Device::CPU);

    std::cout << "PASSED" << std::endl;
}

void test_ir_trace_for_linear() {
    std::cout << "Running test: IRTrace for Linear... ";
    TensorID::reset();
    IRTrace::reset();

    Linear linear(3, 2);
    linear.set_weights(Tensor2D::from_random(3, 2));
    linear.set_bias(Tensor2D::from_random(1, 2));
    Tensor2D input = Tensor2D::from_random(1, 3);
    Tensor2D output = linear.forward(input);

    auto ops = IRTrace::get_ops();

    assert(IRTrace::size() == 3);
    assert(ops[0].op_name == "mat_mul");
    assert(ops[0].inputs.size() == 2);
    assert(ops[0].output == "tensor_5");
    assert(shapes_equal(ops[0].shape, std::make_pair(1UL, 2UL)));
    assert(ops[0].device == Device::CPU);
    
    assert(ops[1].op_name == "operator+");
    assert(ops[1].inputs.size() == 2);
    assert(ops[1].output == output.get_id());
    assert(shapes_equal(ops[1].shape, std::make_pair(1UL, 2UL)));
    assert(ops[1].device == Device::CPU);
    
    assert(ops[2].op_name == "linear");
    assert(ops[2].inputs.size() == 3);
    assert(ops[2].output == output.get_id());
    assert(shapes_equal(ops[2].shape, std::make_pair(1UL, 2UL)));
    assert(ops[2].device == Device::CPU);
    
    std::cout << "PASSED" << std::endl;
}

void test_ir_trace_for_softmax() {
    std::cout << "Running test: IRTrace for Softmax... ";

    TensorID::reset();
    IRTrace::reset();

    Softmax softmax;
    Tensor2D input = Tensor2D::from_random(2, 3);
    Tensor2D output = softmax.forward(input);

    auto ops = IRTrace::get_ops();

    assert(IRTrace::size() == 1);
    assert(ops[0].op_name == "softmax");
    assert(ops[0].inputs[0] == input.get_id());
    assert(ops[0].inputs.size() == 1);
    assert(ops[0].output == output.get_id());
    assert(shapes_equal(ops[0].shape, output.shape()));
    assert(ops[0].device == output.get_device());

    std::cout << "PASSED" << std::endl;
}

void test_ir_trace_for_sequential() {
    std::cout << "Running test: IRTrace for Sequential... ";

    TensorID::reset();
    IRTrace::reset();

    Sequential model;
    model.add(std::make_unique<Linear>(3, 4));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(4, 2));

    Tensor2D input = Tensor2D::from_random(1, 3);
    Tensor2D output = model.forward(input);

    const auto& ops = IRTrace::get_ops();

    assert(IRTrace::size() == 8);
    assert(ops[2].op_name == "linear");
    assert(ops[3].op_name == "relu");
    assert(ops[6].op_name == "linear");
    assert(ops[7].op_name == "sequential");
    assert(ops[7].output == output.get_id());
    assert(shapes_equal(ops[7].shape, output.shape()));
    assert(ops[7].device == output.get_device());

    std::cout << "PASSED" << std::endl;
}

#ifdef USE_CUDA
void test_to_device_roundtrip() {
    std::cout << "Running test: to(Device) roundtrip transfer... ";
    Tensor2D original = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    Tensor2D gpu_tensor = original.to(Device::GPU);
    assert(gpu_tensor.get_device() == Device::GPU);
    assert(gpu_tensor.shape() == original.shape());
    
    Tensor2D cpu_tensor = gpu_tensor.to(Device::CPU);
    assert(cpu_tensor.get_device() == Device::CPU);
    assert(cpu_tensor.shape() == original.shape());
    
    for (size_t i = 0; i < original.rows(); ++i) {
        for (size_t j = 0; j < original.cols(); ++j) {
            assert(cpu_tensor(i, j) == original(i, j));
        }
    }
    
    std::cout << "PASSED" << std::endl;
}
#endif

void test_copy_from_same_shape() {
    std::cout << "Running test: copy_from() with same shape... ";
    Tensor2D source = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor2D dest = Tensor2D::from_vector(2, 3, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    
    dest.copy_from(source);
    
    for (size_t i = 0; i < source.rows(); ++i) {
        for (size_t j = 0; j < source.cols(); ++j) {
            assert(dest(i, j) == source(i, j));
        }
    }
    
    for (size_t i = 0; i < source.rows(); ++i) {
        for (size_t j = 0; j < source.cols(); ++j) {
            assert(source(i, j) == (i * source.cols() + j + 1.0f));
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_copy_from_different_shape_throws() {
    std::cout << "Running test: copy_from() with different shape throws... ";
    Tensor2D source(2, 3, 1.0f);
    Tensor2D dest(3, 2, 2.0f);
    
    bool exception_thrown = false;
    try {
        dest.copy_from(source);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

#ifdef USE_CUDA
void test_copy_from_different_device_throws() {
    std::cout << "Running test: copy_from() with different device throws... ";
    Tensor2D source(2, 3, 1.0f, Device::CPU);
    Tensor2D dest(2, 3, 2.0f, Device::GPU);
    
    bool exception_thrown = false;
    try {
        dest.copy_from(source);
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}
#endif

void test_copy_from_same_tensor() {
    std::cout << "Running test: copy_from() same tensor... ";
    Tensor2D tensor = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    tensor.copy_from(tensor);
    
    for (size_t i = 0; i < tensor.rows(); ++i) {
        for (size_t j = 0; j < tensor.cols(); ++j) {
            assert(tensor(i, j) == (i * tensor.cols() + j + 1.0f));
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_copy_constructor_deep_copy() {
    std::cout << "Running test: copy constructor deep copy... ";
    Tensor2D original = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::string original_id = original.get_id();
    
    Tensor2D copy(original);
    
    assert(copy.shape() == original.shape());
    assert(copy.get_device() == original.get_device());
    assert(copy.get_id() != original_id);
    
    for (size_t i = 0; i < original.rows(); ++i) {
        for (size_t j = 0; j < original.cols(); ++j) {
            assert(copy(i, j) == original(i, j));
        }
    }
    
    original(0, 0) = 999.0f;
    assert(copy(0, 0) != 999.0f);
    assert(copy(0, 0) == 1.0f);
    std::cout << "PASSED" << std::endl;
}

void test_assignment_operator_deep_copy() {
    std::cout << "Running test: assignment operator deep copy... ";
    Tensor2D original = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::string original_id = original.get_id();
    
    Tensor2D dest = Tensor2D::from_vector(1, 2, {10.0f, 20.0f});
    dest = original;
    
    assert(dest.shape() == original.shape());
    assert(dest.get_device() == original.get_device());
    assert(dest.get_id() != original_id);
    
    for (size_t i = 0; i < original.rows(); ++i) {
        for (size_t j = 0; j < original.cols(); ++j) {
            assert(dest(i, j) == original(i, j));
        }
    }
    
    original(0, 0) = 999.0f;
    assert(dest(0, 0) != 999.0f);
    assert(dest(0, 0) == 1.0f);
    std::cout << "PASSED" << std::endl;
}

void test_self_assignment() {
    std::cout << "Running test: self assignment... ";
    Tensor2D tensor = Tensor2D::from_vector(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::string original_id = tensor.get_id();
    
    tensor = tensor;
    
    assert(tensor.shape() == std::make_pair(2ul, 3ul));
    assert(tensor.get_id() == original_id);
    
    for (size_t i = 0; i < tensor.rows(); ++i) {
        for (size_t j = 0; j < tensor.cols(); ++j) {
            assert(tensor(i, j) == (i * tensor.cols() + j + 1.0f));
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_memory_management_destruction() {
    std::cout << "Running test: memory management on destruction... ";
    Tensor2D cpu_tensor(10, 10, 1.0f, Device::CPU);
    
    #ifdef USE_CUDA
    Tensor2D gpu_tensor(10, 10, 1.0f, Device::GPU);
    #endif
    std::cout << "PASSED" << std::endl;
}

void test_infer_broadcast_shape_3d_with_same_shapes() {
    std::cout << "Running test: infer_broadcast_shape_3d with same shapes... ";
    size_t B = 2, M = 3, N = 4;
    Tensor3D a(B, M, N);
    Tensor3D b(B, M, N);
    auto shape = Tensor3D::infer_broadcast_shape_3d(a.shape(), b.shape());
    assert(shape == std::make_tuple(B, M, N));
    std::cout << "PASSED" << std::endl;
}

void test_infer_broadcast_shape_3d_with_different_shapes() {
    std::cout << "Running test: infer_broadcast_shape_3d with different shapes... ";
    
    // Test case 1: Broadcast batch dimension
    size_t B1 = 5, M1 = 3, N1 = 5;
    size_t B2 = 1, M2 = 3, N2 = 5;
    Tensor3D a1(B1, M1, N1);
    Tensor3D b1(B2, M2, N2);
    auto shape1 = Tensor3D::infer_broadcast_shape_3d(a1.shape(), b1.shape());
    assert(shape1 == std::make_tuple(B1, M1, N1));

    // Test case 2: Broadcast row dimension
    size_t B3 = 5, M3 = 5, N3 = 5;
    size_t B4 = 5, M4 = 1, N4 = 5;
    Tensor3D a2(B3, M3, N3);
    Tensor3D b2(B4, M4, N4);
    auto shape2 = Tensor3D::infer_broadcast_shape_3d(a2.shape(), b2.shape());
    assert(shape2 == std::make_tuple(B3, M3, N3));

    // Test case 3: Broadcast column dimension
    size_t B5 = 5, M5 = 5, N5 = 5;
    size_t B6 = 5, M6 = 5, N6 = 1;
    Tensor3D a3(B5, M5, N5);
    Tensor3D b3(B6, M6, N6);
    auto shape3 = Tensor3D::infer_broadcast_shape_3d(a3.shape(), b3.shape());
    assert(shape3 == std::make_tuple(B5, M5, N5));
    
    std::cout << "PASSED" << std::endl;
}

void test_infer_broadcast_shape_3d_with_incompatible_shapes() {
    std::cout << "Running test: infer_broadcast_shape_3d with incompatible shapes... ";
    size_t B1 = 5, M1 = 3, N1 = 4;
    size_t B2 = 3, M2 = 3, N2 = 4;
    Tensor3D a(B1, M1, N1);
    Tensor3D b(B2, M2, N2);
    bool exception_thrown = false;
    try {
        auto shape = Tensor3D::infer_broadcast_shape_3d(a.shape(), b.shape());
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_for_each_broadcasted_3d_with_same_shapes() {
    std::cout << "Running test: for_each_broadcasted_3d with same shapes... ";
    size_t B = 2, M = 3, N = 4;
    Tensor3D a(B, M, N, 41.0f);
    Tensor3D b(B, M, N, 1.0f);
    Tensor3D c(B, M, N);
    Tensor3D::for_each_broadcasted_3d(a, b, c, [](float a, float b) {
        return a + b;
    });
    assert(c.shape() == std::make_tuple(B, M, N));
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < N; ++k) {
                assert(c(i, j, k) == 42.0f);
            }
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_for_each_broadcasted_3d_with_different_shapes() {
    std::cout << "Running test: for_each_broadcasted_3d with different shapes... ";
        
    // Test case 1: Broadcast batch dimension
    size_t B1 = 5, M1 = 3, N1 = 5;
    size_t B2 = 1, M2 = 3, N2 = 5;
    Tensor3D a1(B1, M1, N1, 41.0f);
    Tensor3D b1(B2, M2, N2, 1.0f);
    Tensor3D c1(B1, M1, N1);
    Tensor3D::for_each_broadcasted_3d(a1, b1, c1, [](float a, float b) {
        return a + b;
    });
    assert(c1.shape() == std::make_tuple(B1, M1, N1));
    for (size_t i = 0; i < B1; ++i) {
        for (size_t j = 0; j < M1; ++j) {
            for (size_t k = 0; k < N1; ++k) {
                assert(c1(i, j, k) == 42.0f);
            }
        }
    }

    // Test case 2: Broadcast row dimension
    size_t B3 = 5, M3 = 5, N3 = 5;
    size_t B4 = 5, M4 = 1, N4 = 5;
    Tensor3D a2(B3, M3, N3, 41.0f);
    Tensor3D b2(B4, M4, N4, 1.0f);
    Tensor3D c2(B3, M3, N3);
    Tensor3D::for_each_broadcasted_3d(a2, b2, c2, [](float a, float b) {
        return a + b;
    });
    assert(c2.shape() == std::make_tuple(B3, M3, N3));
    for (size_t i = 0; i < B3; ++i) {
        for (size_t j = 0; j < M3; ++j) {
            for (size_t k = 0; k < N3; ++k) {
                assert(c2(i, j, k) == 42.0f);
            }
        }
    }

    // Test case 3: Broadcast column dimension
    size_t B5 = 5, M5 = 5, N5 = 5;
    size_t B6 = 5, M6 = 5, N6 = 1;
    Tensor3D a3(B5, M5, N5, 41.0f);
    Tensor3D b3(B6, M6, N6, 1.0f);
    Tensor3D c3(B5, M5, N5);
    Tensor3D::for_each_broadcasted_3d(a3, b3, c3, [](float a, float b) {
        return a + b;
    });
    assert(c3.shape() == std::make_tuple(B5, M5, N5));
    for (size_t i = 0; i < B5; ++i) {
        for (size_t j = 0; j < M5; ++j) {
            for (size_t k = 0; k < N5; ++k) {
                assert(c3(i, j, k) == 42.0f);
            }
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_for_each_broadcasted_3d_with_incompatible_shapes() {
    std::cout << "Running test: for_each_broadcasted_3d with incompatible shapes... ";
    size_t B = 2, M = 3, N = 4;
    Tensor3D a(B, M, N, 41.0f);
    Tensor3D b(B, M, N, 1.0f);
    Tensor3D c(1, M, N);

    bool exception_thrown = false;
    try {
        Tensor3D::for_each_broadcasted_3d(a, b, c, [](float a, float b) {
            return a + b;
        });
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_for_each_broadcasted_3d_2d_with_same_shapes() {
    std::cout << "Running test: for_each_broadcasted_3d_2d with same shapes... ";
    size_t B = 2, M = 3, N = 4;
    Tensor3D a(B, M, N, 41.0f);
    Tensor2D b(M, N, 1.0f);
    Tensor3D c(B, M, N);
    Tensor3D::for_each_broadcasted_3d_2d(a, b, c, [](float a, float b) {
        return a + b;
    });
    assert(c.shape() == std::make_tuple(B, M, N));
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < N; ++k) {
                assert(c(i, j, k) == 42.0f);
            }
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_for_each_broadcasted_3d_2d_with_different_shapes() {
    std::cout << "Running test: for_each_broadcasted_3d_2d with different shapes... ";
    size_t B = 2, M = 3, N = 4;
    Tensor3D a(B, M, N, 41.0f);
    Tensor2D b1(1, N, 1.0f);
    Tensor3D c(B, M, N);
    Tensor3D::for_each_broadcasted_3d_2d(a, b1, c, [](float a, float b) {
        return a + b;
    });
    assert(c.shape() == std::make_tuple(B, M, N));
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < N; ++k) {
                assert(c(i, j, k) == 42.0f);
            }
        }
    }

    Tensor2D b2(M, 1, 1.0f);
    Tensor3D::for_each_broadcasted_3d_2d(a, b2, c, [](float a, float b) {
        return a + b;
    });
    assert(c.shape() == std::make_tuple(B, M, N));
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < M; ++j) {
            for (size_t k = 0; k < N; ++k) {
                assert(c(i, j, k) == 42.0f);
            }
        }
    }

    std::cout << "PASSED" << std::endl;
}

void test_tensor3d_for_each_broadcasted_3d_2d_with_incompatible_shapes() {
    std::cout << "Running test: for_each_broadcasted_3d_2d with incompatible shapes... ";
    size_t B = 2, M = 3, N = 4;
    Tensor3D a(B, M, N, 41.0f);
    Tensor2D b(M, N, 1.0f);
    Tensor3D c(1, M, N);

    bool exception_thrown = false;
    try {
        Tensor3D::for_each_broadcasted_3d_2d(a, b, c, [](float a, float b) {
            return a + b;
        });
    } catch (const std::invalid_argument& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << std::endl;
    std::cout << "--- Running Tensor2D Unit Tests ---" << std::endl;
    std::cout << std::endl;

    test_shape();
    test_reshape();
    test_infer_broadcast_shape();
    std::cout << std::endl;
    
    test_row_wise_expand();
    test_column_wise_expand();
    test_expand_with_incompatible_shapes();
    std::cout << std::endl;

    test_constructor_with_default_value();
    test_fill_and_operator_parentheses();
    test_from_vector();
    test_from_random();
    test_tensor_device();    
    std::cout << std::endl;

    test_tensor_equality();
    test_tensor_inequality();
    std::cout << std::endl;

    test_addition_with_same_shapes();
    test_addition_with_compatible_broadcast_shapes();
    test_addition_with_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_addition_in_place_with_same_shapes();
    test_addition_in_place_compatible_broadcast_shapes();
    test_addition_in_place_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_addition_with_scalar();
    test_addition_with_scalar_in_place();
    std::cout << std::endl;

    test_subtraction_with_same_shapes();
    test_subtraction_with_compatible_broadcast_shapes();
    test_subtraction_with_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_subtraction_in_place_with_same_shapes();
    test_subtraction_in_place_compatible_broadcast_shapes();
    test_subtraction_in_place_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_subtraction_with_scalar();
    test_subtraction_with_scalar_in_place();
    std::cout << std::endl;

    test_multiplication_with_same_shapes();
    test_multiplication_with_compatible_broadcast_shapes();
    test_multiplication_with_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_multiplication_in_place_with_same_shapes();
    test_multiplication_in_place_compatible_broadcast_shapes();
    test_multiplication_in_place_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_multiplication_with_scalar();
    test_multiplication_with_scalar_in_place();
    std::cout << std::endl;

    test_division_with_same_shapes();
    test_division_with_compatible_broadcast_shapes();
    test_division_with_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_division_in_place_with_same_shapes();
    test_division_in_place_compatible_broadcast_shapes();
    test_division_in_place_incompatible_broadcast_shapes();
    std::cout << std::endl;

    test_division_with_scalar();
    test_division_with_scalar_in_place();
    std::cout << std::endl;

    test_relu();
    test_negate();
    test_abs();
    std::cout << std::endl;

    test_relu_in_place();
    test_negate_in_place();
    test_abs_in_place();
    std::cout << std::endl;

    test_sum();
    test_mean();
    test_max();
    test_arg_max();
    std::cout << std::endl;

    test_mat_mul_with_same_shapes();
    test_mat_mul_with_compatible_shapes();
    test_mat_mul_with_identity_matrix();
    test_mat_mul_with_incompatible_shapes();
    std::cout << std::endl;

    test_mat_mul_eigen_with_same_shapes();
    test_mat_mul_eigen_with_compatible_shapes();
    test_mat_mul_eigen_with_identity_matrix();
    test_mat_mul_eigen_with_incompatible_shapes();
    std::cout << std::endl;

    #ifdef USE_CUDA
    test_mat_mul_cuda_with_same_shapes();
    test_mat_mul_cuda_with_compatible_shapes();
    test_mat_mul_cuda_with_identity_matrix();
    test_mat_mul_cuda_with_incompatible_device();
    test_mat_mul_cuda_with_incompatible_shapes();

    test_bmm_cuda_with_same_shapes();
    std::cout << std::endl;
    #endif

    test_linear_forward();
    test_relu_forward();
    test_sequential_forward();
    test_softmax_forward();
    std::cout << std::endl;

    test_tensor2d_view();
    test_tensor2d_view_out_of_bounds();
    std::cout << std::endl;

    test_tensor3d_constructor_and_shape();
    test_tensor3d_element_access_and_mutation();
    test_tensor3d_from_random();
    test_tensor3d_slice_batch();
    test_tensor3d_getters();
    std::cout << std::endl;

    test_copy_from_same_shape();
    test_copy_from_different_shape_throws();
    #ifdef USE_CUDA
    test_to_device_roundtrip();
    test_copy_from_different_device_throws();
    #endif
    std::cout << std::endl;
    
    test_copy_constructor_deep_copy();
    test_assignment_operator_deep_copy();
    test_self_assignment();
    test_copy_from_same_tensor();
    test_memory_management_destruction();
    std::cout << std::endl;

    test_infer_broadcast_shape_3d_with_same_shapes();
    test_infer_broadcast_shape_3d_with_different_shapes();
    test_infer_broadcast_shape_3d_with_incompatible_shapes();
    std::cout << std::endl;

    test_tensor3d_for_each_broadcasted_3d_with_same_shapes();
    test_tensor3d_for_each_broadcasted_3d_with_different_shapes();
    test_tensor3d_for_each_broadcasted_3d_with_incompatible_shapes();
    std::cout << std::endl;

    test_tensor3d_for_each_broadcasted_3d_2d_with_same_shapes();
    test_tensor3d_for_each_broadcasted_3d_2d_with_different_shapes();
    test_tensor3d_for_each_broadcasted_3d_2d_with_incompatible_shapes();
    std::cout << std::endl;

    test_tensor3d_addition_with_same_shapes();
    test_tensor3d_subtraction_with_same_shapes();
    test_tensor3d_multiplication_with_same_shapes();
    test_tensor3d_division_with_same_shapes();
    std::cout << std::endl;

    test_tensor3d_addition_in_place_with_same_shapes();
    test_tensor3d_subtraction_in_place_with_same_shapes();
    test_tensor3d_multiplication_in_place_with_same_shapes();
    test_tensor3d_division_in_place_with_same_shapes();
    std::cout << std::endl;

    test_tensor3d_mat_mul();
    test_tensor3d_mat_mul_eigen();
    test_tensor3d_mat_mul_eigen_parallel();
    std::cout << std::endl;

    test_ir_trace_for_arithmetic_operators();
    test_ir_trace_for_operators();
    test_ir_trace_for_linear();
    test_ir_trace_for_softmax();
    test_ir_trace_for_sequential();
    std::cout << std::endl;

    

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "All tests passed successfully!" << std::endl;

    return 0;
}
