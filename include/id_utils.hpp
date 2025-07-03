#pragma once
#include <string>

namespace TensorID {
    inline int next_id = 0;

    inline std::string generate() {
        return "tensor_" + std::to_string(next_id++);
    }

    inline void reset() {
        next_id = 0;
    }
}
