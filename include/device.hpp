#pragma once

enum class Device { CPU, GPU };

inline const char* to_string(Device device) {
    return (device == Device::CPU) ? "CPU" : "GPU";
}
