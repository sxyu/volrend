#pragma once

#include <memory>
#include "camera.hpp"
#include "n3tree.hpp"
#include "render_options.hpp"

namespace volrend {
struct CUDAVolumeRenderer {
    CUDAVolumeRenderer();
    ~CUDAVolumeRenderer();

    // Draw volumetric data
    void render(const N3Tree& tree);

    // Swap buffers
    void swap();

    // Clear the buffer with color
    void clear(float r = 1.f, float g = 1.f, float b = 1.f, float a = 1.f);

    // Resize the buffer
    void resize(int width, int height);

    // Camera instance
    Camera camera;

    // Rendering options
    RenderOptions options;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend
