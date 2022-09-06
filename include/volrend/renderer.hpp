#pragma once

#include <memory>
#include "volrend/camera.hpp"
#include "volrend/n3tree.hpp"
#include "volrend/mesh.hpp"
#include "volrend/render_options.hpp"

namespace volrend {
// Volume renderer using CUDA or compute shader
struct Renderer {
    explicit Renderer();
    ~Renderer();

    // Render the currently set tree
    void render();

    // Add volumetric data to render
    void add(const N3Tree& tree);
    void add(N3Tree&& tree);

    // Clear the volumetric data
    void clear();

    // Load series of volumes/meshes/lines/points from a npz file
    void open_drawlist(const std::string& path,
            bool default_visible = true);
    void open_drawlist_mem(const char* data, uint64_t size,
            bool default_visible = true);

    // Resize the buffer
    void resize(int width, int height);

    // Get name identifying the renderer backend used e.g. CUDA
    const char* get_backend();

    // Camera instance
    Camera camera;

    // Rendering Options
    RendererOptions options;

    // Meshes to draw
    std::vector<Mesh> meshes;
    // PlenOctrees to draw
    std::vector<N3Tree> trees;

    // Time
    int time = 0;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volrend
