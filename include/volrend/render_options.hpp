#pragma once

#include "volrend/common.hpp"

// Max global basis
#define VOLREND_GLOBAL_BASIS_MAX 25

namespace volrend {

// Rendering options
struct RenderOptions {
    // * BASIC RENDERING
    // Epsilon added to steps to prevent hitting current box again
    float step_size = 1e-4f;

    // If a point has sigma < this amount, considers sigma = 0
    float sigma_thresh = 1e-2f;

    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh = 1e-2f;

    // Background brightness
    float background_brightness = 1.f;

    // * VISUALIZATION
    // Rendering bounding box (relative to outer tree bounding box [0, 1])
    // [minx, miny, minz, maxx, maxy, maxz]
    float render_bbox[6] = {0.f, 0.f, 0.f, 1.f, 1.f, 1.f};

    // Range of basis functions to use
    // no effect if RGBA data format
    int basis_minmax[2] = {0, VOLREND_GLOBAL_BASIS_MAX - 1};

    // Rotation applied to viewdirs for all rays
    float rot_dirs[3] = {0.f, 0.f, 0.f};

    // * ADVANCED VISUALIZATION

    // Draw a (rather low-quality) grid to help visualize the octree
    bool show_grid = false;
    // Grid max depth
    int grid_max_depth = 4;

#ifdef VOLREND_CUDA
    // Render depth instead of color, currently CUDA only
    bool render_depth = false;
#endif

    // * Probe for inspecting lumispheres
    bool enable_probe = false;
    float probe[3] = {0.f, 0.f, 1.f};
    int probe_disp_size = 100;
};

}  // namespace volrend
