#pragma once

#include <array>
#include "volrend/common.hpp"

namespace volrend {

struct Camera {
    Camera(int width = 256, int height = 256, float focal = 300.f);
    ~Camera();

    void update();
    void load_cuda();

    // Camera pose as center + 2 axes
    std::array<float, 3> v_right, v_forward, center;

    // Image size
    int width, height;

    // Focal length
    float focal;

    // CUDA memory used in kernel
    struct {
        float* transform = nullptr;
    } device;

   private:
    std::array<float, 12> transform_;
    void free_cuda();
};

}  // namespace volrend
