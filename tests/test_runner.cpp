#include "tensor2d.hpp"
#include <cassert>
#include <iostream>
#include <numeric>

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

    try {
        Tensor2D t3 = t1 + t2;
    } catch (const std::invalid_argument& e) {
        std::cout << "PASSED" << std::endl;
    }
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
    std::cout << std::endl;

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "All tests passed successfully!" << std::endl;

    return 0;
}
