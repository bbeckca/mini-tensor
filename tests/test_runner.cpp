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
#include <numeric>
#include "ir_trace.hpp"

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

    Tensor2D t2(rows, cols, 0.0f, Device::GPU);
    assert(t2.get_device() == Device::GPU);

    Tensor2D t3 = Tensor2D::from_vector(rows, cols, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Device::GPU);
    assert(t3.get_device() == Device::GPU);

    Tensor2D t4 = Tensor2D::from_random(rows, cols, Device::GPU);
    assert(t4.get_device() == Device::GPU);    
    std::cout << "PASSED" << std::endl;
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

void test_tensor3d_mat_mul() {
    std::cout << "Running test: Tensor3D mat_mul... ";

    Tensor3D t1(2, 2, 2);
    Tensor3D t2(2, 2, 2);

    t1[0] = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    t2[0] = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

    t1[1] = Tensor2D::from_vector(2, 2, {9.0f, 10.0f, 11.0f, 12.0f});
    t2[1] = Tensor2D::from_vector(2, 2, {13.0f, 14.0f, 15.0f, 16.0f});

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
    std::cout << "Running test: Tensor3D mat_mul... ";

    Tensor3D t1(2, 2, 2);
    Tensor3D t2(2, 2, 2);

    t1[0] = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    t2[0] = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

    t1[1] = Tensor2D::from_vector(2, 2, {9.0f, 10.0f, 11.0f, 12.0f});
    t2[1] = Tensor2D::from_vector(2, 2, {13.0f, 14.0f, 15.0f, 16.0f});

    Tensor3D t3 = t1.mat_mul_eigen(t2);

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

void test_tensor3d_mat_mul_eigen_parallel() {
    std::cout << "Running test: Tensor3D mat_mul_eigen_parallel... ";

    Tensor3D t1(2, 2, 2);
    Tensor3D t2(2, 2, 2);

    t1[0] = Tensor2D::from_vector(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    t2[0] = Tensor2D::from_vector(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});

    t1[1] = Tensor2D::from_vector(2, 2, {9.0f, 10.0f, 11.0f, 12.0f});
    t2[1] = Tensor2D::from_vector(2, 2, {13.0f, 14.0f, 15.0f, 16.0f});

    Tensor3D t3 = t1.mat_mul_eigen_parallel(t2);

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

void test_ir_trace_for_ops() {
    std::cout << "Running test: IRTrace...\n";
    TensorID::reset();
    IRTrace::reset();

    Tensor2D a = Tensor2D::from_random(2, 3);
    Tensor2D b = Tensor2D::from_random(3, 2);
    Tensor2D c = a.mat_mul(b);
    Tensor2D d = c.relu();

    IRTrace::print();
    assert(IRTrace::size() == 2);

    auto ops = IRTrace::get_ops();
    assert(ops[0].op_name == "mat_mul");
    assert(ops[0].inputs[0] == a.get_id());
    assert(ops[0].inputs[1] == b.get_id());
    assert(ops[0].output == c.get_id());
    assert(ops[0].shape == std::make_pair(2UL, 2UL));
    assert(ops[0].device == Device::CPU);

    assert(ops[1].op_name == "relu");
    assert(ops[1].inputs[0] == c.get_id());
    assert(ops[1].output == d.get_id());
    assert(ops[1].shape == std::make_pair(2UL, 2UL));
    assert(ops[1].device == Device::CPU);

    std::cout << "PASSED" << std::endl;
}

void test_ir_trace_for_linear() {
    std::cout << "\nRunning test: IRTrace for Linear...\n";
    TensorID::reset();
    IRTrace::reset();

    Linear linear(3, 2);
    linear.set_weights(Tensor2D::from_random(3, 2));
    linear.set_bias(Tensor2D::from_random(1, 2));
    Tensor2D input = Tensor2D::from_random(1, 3);
    Tensor2D output = linear.forward(input);

    IRTrace::print();
    std::cout << "IRTrace size: " << IRTrace::size() << std::endl;
    assert(IRTrace::size() == 3);
    
    auto ops = IRTrace::get_ops();
    
    assert(ops[0].op_name == "mat_mul");
    assert(ops[0].inputs.size() == 2);
    assert(ops[0].output == "tensor_5");
    assert(ops[0].shape == std::make_pair(1UL, 2UL));
    assert(ops[0].device == Device::CPU);
    
    assert(ops[1].op_name == "operator+");
    assert(ops[1].inputs.size() == 2);
    assert(ops[1].output == output.get_id());
    assert(ops[1].shape == std::make_pair(1UL, 2UL));
    assert(ops[1].device == Device::CPU);
    
    assert(ops[2].op_name == "linear");
    assert(ops[2].inputs.size() == 3);
    assert(ops[2].output == output.get_id());
    assert(ops[2].shape == std::make_pair(1UL, 2UL));
    assert(ops[2].device == Device::CPU);
    
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
    std::cout << std::endl;

    test_tensor3d_mat_mul();
    test_tensor3d_mat_mul_eigen();
    std::cout << std::endl;

    test_tensor3d_mat_mul_eigen_parallel();
    std::cout << std::endl;

    test_ir_trace_for_ops();
    test_ir_trace_for_linear();
    std::cout << std::endl;

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "All tests passed successfully!" << std::endl;

    return 0;
}
