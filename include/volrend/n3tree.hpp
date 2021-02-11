#pragma once

#include <string>
#include <vector>
#include <array>
#include <tuple>
#include "cnpy.h"

namespace volrend {

using Rgba = std::array<float, 4>;

// Read-only N3Tree loader
struct N3Tree {
    N3Tree() = delete;
    explicit N3Tree(const std::string& path);
    ~N3Tree();

    void open(const std::string& path);

    // Spatial branching factor
    int N;
    int sh_order;
    // Dimensionality of data on leaf (e.g. 4 for rgba)
    int data_dim;
    // Number of internal nodes
    int n_internal;
    // Capacity
    int capacity;

    // Scaling for coordinates
    float scale;
    // Translation
    std::array<float, 3> offset;

    // Get child of node in given position of subgrid
    int32_t get_child(int nd, int i, int j, int k);

    // Get data at node node in given position of subgrid
    Rgba get_data(int nd, int i, int j, int k);

    // FIXME No longer supported
    // Query. Indices size must be divisible by 3 in order: xyz xyz
    // Returns rgba rgba...
    // std::vector<float> operator[](const std::vector<float>& indices) const;

    bool is_data_loaded();
    bool is_cuda_loaded();

    // Pre-apply operations to leaf (in CUDA memory):
    // Also applies sigmoid & relu
    void precompute_step(float sigma_thresh) const;

    // Index pack/unpack
    int pack_index(int nd, int i, int j, int k);
    std::tuple<int, int, int, int> unpack_index(int packed);

    // NDC config
    bool use_ndc;
    float ndc_width, ndc_height, ndc_focal;

    // CUDA memory
    mutable struct {
        float* data = nullptr;
        int32_t* child = nullptr;
        float* offset = nullptr;
    } device;

   private:
    int N2_, N3_;

    // Main data holder
    std::vector<float> data_;
    cnpy::NpyArray data_cnpy_;

    // Child link data holder
    cnpy::NpyArray child_;

    // Paths
    std::string npz_path_, data_path_, poses_bounds_path_;
    bool data_loaded_, cuda_loaded_;

    mutable float last_sigma_thresh_;

    void load_data();
    void load_cuda();
    void free_cuda();
};

}  // namespace volrend
