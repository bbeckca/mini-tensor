#pragma once
#include <string>
#include <vector>
#include <iostream>

struct Op {
    std::string op_name;
    std::vector<std::string> inputs;
    std::string output;
};

class IRTrace {
public:
    static void record(const std::string& op_name,
                       const std::vector<std::string>& inputs,
                       const std::string& output) {
        ops_.push_back({op_name, inputs, output});
    }

    static size_t size() {
        return ops_.size();
    }

    static void print() {
        for (size_t i = 0; i < ops_.size(); ++i) {
            const auto& op = ops_[i];
            std::cout << "[" << i << "] " << op.op_name << "(";
            for (size_t j = 0; j < op.inputs.size(); ++j) {
                std::cout << op.inputs[j];
                if (j + 1 < op.inputs.size()) std::cout << ", ";
            }
            std::cout << ") -> " << op.output << "\n";
        }
    }

    static void reset() {
        ops_.clear();
    }

private:
    static inline std::vector<Op> ops_;
};
