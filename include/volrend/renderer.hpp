#pragma once

#include <memory>
#include "volrend/camera.hpp"
#include "volrend/n3tree.hpp"
#include "volrend/render_options.hpp"

namespace volrend {
// Volume renderer using OpenGL & compute shader
struct VolumeRenderer {
    explicit VolumeRenderer(int device_id = -1);
    ~VolumeRenderer();

    // Render all added trees thru OpennGL
    void render();

    // Set volumetric data to render
    void set(const N3Tree& tree);

    // Clear the volumetric data
    void clear();

    // Resize the buffer
    void resize(int width, int height);

    // Camera instance
    Camera camera;

    // Rendering options
    RenderOptions options;

    // Get name identifying the renderer backend used e.g. CUDA
    const char* get_backend();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend
