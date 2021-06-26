#pragma once

#include <memory>
#include "volrend/camera.hpp"
#include "volrend/n3tree.hpp"
#include "volrend/render_options.hpp"
#include "volrend/mesh.hpp"

namespace volrend {
// Volume renderer using CUDA or compute shader
struct VolumeRenderer {
    explicit VolumeRenderer();
    ~VolumeRenderer();

    // Render the currently set tree
    void render();

    // Set volumetric data to render
    void set(N3Tree& tree);

    // Clear the volumetric data
    void clear();

    // Resize the buffer
    void resize(int width, int height);

    // Get name identifying the renderer backend used e.g. CUDA
    const char* get_backend();

    // Camera instance
    Camera camera;

    // Rendering options
    RenderOptions options;

    // Meshes to draw, currently only supported on CUDA implementation
    std::vector<Mesh> meshes;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend
