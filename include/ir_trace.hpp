#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "device.hpp"

struct Op {
    std::string op_name;
    std::vector<std::string> inputs;
    std::string output;
    std::pair<size_t, size_t> shape;
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
            std::cout << "    Shape  : " << op.shape.first << " x " << op.shape.second << "\n";
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
