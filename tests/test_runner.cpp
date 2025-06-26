#include "tensor2d.hpp"
#include <cassert>
#include <iostream>
#include <numeric>

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

int main() {
    std::cout << "--- Running Tensor2D Unit Tests ---" << std::endl;
    
    
    test_shape();
    test_reshape();
    test_infer_broadcast_shape();
    test_fill_and_operator_parentheses();
    
    test_addition_with_same_shapes();
    test_addition_with_compatible_broadcast_shapes();
    test_addition_with_incompatible_broadcast_shapes();

    test_addition_in_place_with_same_shapes();
    test_addition_in_place_compatible_broadcast_shapes();
    test_addition_in_place_incompatible_broadcast_shapes();

    test_relu();
    test_negate();
    test_abs();

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "All tests passed successfully!" << std::endl;

    return 0;
}
