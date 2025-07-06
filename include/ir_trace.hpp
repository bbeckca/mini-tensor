#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <variant>
#include <tuple>
#include "device.hpp"

struct Op {
    std::string op_name;
    std::vector<std::string> inputs;
    std::string output;
    std::variant<std::pair<size_t, size_t>, std::tuple<size_t, size_t, size_t>> shape;
    Device device;
};

class IRTrace {
public:
    static void record(const std::string& op_name,
                       const std::vector<std::string>& inputs,
                       const std::string& output,
                       const std::pair<size_t, size_t>& shape,
                       const Device& device) {
        ops_.push_back({op_name, inputs, output, shape, device});
    }

    static void record(const std::string& op_name,
                       const std::vector<std::string>& inputs,
                       const std::string& output,
                       const std::tuple<size_t, size_t, size_t>& shape,
                       const Device& device) {
        ops_.push_back({op_name, inputs, output, shape, device});
    }

    static size_t size() {
        return ops_.size();
    }

    static void print() {
        for (size_t i = 0; i < ops_.size(); ++i) {
            const auto& op = ops_[i];
            std::cout << "[" << i << "] Operation: " << op.op_name << "\n";
            std::cout << "    Inputs : ";
            for (size_t j = 0; j < op.inputs.size(); ++j) {
                std::cout << op.inputs[j];
                if (j + 1 < op.inputs.size()) std::cout << ", ";
            }
            std::cout << "\n";
            std::cout << "    Output : " << op.output << "\n";
            std::cout << "    Shape  : ";
            std::visit([](const auto& shape) {
                if constexpr (std::is_same_v<decltype(shape), const std::pair<size_t, size_t>&>) {
                    std::cout << shape.first << " x " << shape.second;
                } else {
                    std::cout << std::get<0>(shape) << " x " << std::get<1>(shape) << " x " << std::get<2>(shape);
                }
            }, op.shape);
            std::cout << "\n";
            std::cout << "    Device : " << to_string(op.device) << "\n";
        }
    }

    static void reset() {
        ops_.clear();
    }

    static std::vector<Op> get_ops() {
        return ops_;
    }

private:
    static inline std::vector<Op> ops_;
};
