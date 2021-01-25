#pragma once

#include <memory>
#include "camera.hpp"
#include "n3tree.hpp"

namespace volrend {
struct CUDAVolumeRenderer {
    CUDAVolumeRenderer();
    ~CUDAVolumeRenderer();

    Camera camera;

    // Draw volumetric data
    void render(const N3Tree& tree);

    // Swap buffers
    void swap();

    // Clear the buffer with color
    void clear(float r = 1.f, float g = 1.f, float b = 1.f, float a = 1.f);

    // Resize the buffer
    void resize(int width, int height);

    // Step size
    float step_size = 1.0f / 400.0;
    // If a point has sigma < this amount, considers sigma = 0
    float sigma_thresh = 0.f;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh = 0.01f;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend
