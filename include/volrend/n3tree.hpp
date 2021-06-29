#pragma once

#include "volrend/common.hpp"
#include "volrend/data_format.hpp"

#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <utility>
#include "cnpy.h"

#include "glm/vec3.hpp"

#ifdef VOLREND_CUDA
#include <cuda_fp16.h>
#else
#include <half.hpp>
#endif

namespace volrend {

// Read-only N3Tree loader
struct N3Tree {
    N3Tree();
    explicit N3Tree(const std::string& path);
    ~N3Tree();

    // Open npz
    void open(const std::string& path);
    // Open memory data stream (for web mostly)
    void open_mem(const char* data, uint64_t size);

    // Generate wireframe (returns line vertex positions; 9 * (a-b c-d) ..)
    // assignable to Mesh.vert
    // up to given depth (default none)
    std::vector<float> gen_wireframe(int max_depth = 100000) const;

    // Spatial branching factor. Only 2 really supported.
    int N = 0;
    // Size of data stored on each leaf
    int data_dim;
    // Data format (SH, SG etc)
    DataFormat data_format;
    // Capacity
    int capacity = 0;

    // Scaling for coordinates
    std::array<float, 3> scale;
    // Translation
    std::array<float, 3> offset;

    bool is_data_loaded();
#ifdef VOLREND_CUDA
    bool is_cuda_loaded();
#endif

    // Clear the CPU memory.
    void clear_cpu_memory();

    // Index pack/unpack
    int pack_index(int nd, int i, int j, int k);
    std::tuple<int, int, int, int> unpack_index(int packed);

    // NDC config
    bool use_ndc;
    float ndc_width, ndc_height, ndc_focal;
    glm::vec3 ndc_avg_up, ndc_avg_back, ndc_avg_cen;

#ifdef VOLREND_CUDA
    // CUDA memory
    mutable struct {
        __half* data = nullptr;
        int32_t* child = nullptr;
        float* offset = nullptr;
        float* scale = nullptr;
        float* extra = nullptr;
    } device;
#endif
    // Main data holder
    cnpy::NpyArray data_;

    // Child link data holder
    cnpy::NpyArray child_;

    // Optional extra data, only used for SG/ASG
    cnpy::NpyArray extra_;

   private:
    // Load data from npz (destructive since it moves some data)
    void load_npz(cnpy::npz_t& npz);

    // Paths
    std::string npz_path_, data_path_, poses_bounds_path_;
    bool data_loaded_;

    int N2_, N3_;

    mutable float last_sigma_thresh_;

#ifdef VOLREND_CUDA
    bool cuda_loaded_;
    void load_cuda();
    void free_cuda();
#endif
};

}  // namespace volrend
