#pragma once

#include "volrend/common.hpp"
#include "volrend/data_format.hpp"
#include "volrend/mesh.hpp"
#include "volrend/camera.hpp"

#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <utility>
#include "cnpy.h"
#include "render_options.hpp"

#include "glm/vec3.hpp"

#ifdef VOLREND_CUDA
#include <cuda_fp16.h>
#else
#include <half.hpp>
#endif

namespace volrend {

// Read-only N3Tree loader & renderer
struct N3Tree {
    N3Tree();

    // Open npz
    void open(const std::string& path);
    // Open memory data stream (for web mostly)
    void open_mem(const char* data, uint64_t size);
    // Load data from npz (destructive since it moves/pads some data)
    void load_npz(cnpy::npz_t& npz);

    // Generate wireframe (returns line vertex positions; 9 * (a-b c-d) ..)
    // assignable to Mesh.vert
    // up to given depth (default none)
    void gen_wireframe(int max_depth = 4) const;

    // Draw the PlenOctree / volume. Returns true if drawn, false if disabled
    // (if false, current we need to deal with the framebuffer in the outer loop)
    bool draw(const Camera& camera, int time = 0) const;

    // Draw the grid structure wireframe visualization
    void draw_wire(const glm::mat4x4& V, glm::mat4x4 K, bool y_up = true,
              int time = 0) const;

    // Spatial branching factor. Only 2 really supported.
    int N = 0;
    // Size of data stored on each leaf
    int data_dim;
    int data_dim_pad;
    // Data format (SH, SG etc)
    DataFormat data_format;
    // Capacity
    int capacity = 0;

    // Scaling for coordinates
    std::array<float, 3> scale;
    // Translation
    std::array<float, 3> offset;

    bool is_data_loaded();

    // Clear the CPU memory.
    void clear_cpu_memory();
    void clear_gpu_memory();

    // Index pack/unpack
    int pack_index(int nd, int i, int j, int k);
    std::tuple<int, int, int, int> unpack_index(int packed);

    // NDC config
    bool use_ndc;
    float ndc_width, ndc_height, ndc_focal;
    glm::vec3 ndc_avg_up, ndc_avg_back, ndc_avg_cen;

    // Main data holder
    cnpy::NpyArray data_;

    // Child link data holder
    cnpy::NpyArray child_;

    // Additional model transform, rotation is axis-angle
    glm::vec3 model_rotation, model_translation;
    float model_scale = 1.f;

    mutable glm::mat4 transform_;

    // Whether to show at all
    bool visible = true;

    // Time stamp to show the volume; if time = -1 then volume is always shown
    int time = -1;

    // A name for the volume
    std::string name = "Volume";

    // Render options
    N3TreeRenderOptions options_;


   private:
    // Paths
    std::string npz_path_, data_path_, poses_bounds_path_;
    bool data_loaded_;

    int N2_, N3_;

    // Wireframe mesh of the octree
    mutable Mesh wire_;
    // The depth level of the octree wireframe; -1 = not yet generated
    mutable int last_wire_depth_ = -1;

    mutable float last_sigma_thresh_;
    int tree_data_stride_;
    int tree_child_stride_;

    unsigned tex_tree_data = -1, tex_tree_child;
};

}  // namespace volrend
