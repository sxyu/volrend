#pragma once

#include <array>
#include "volrend/common.hpp"

// Max global basis
#define VOLREND_GLOBAL_BASIS_MAX 25

namespace volrend {

struct RenderOptions {
    float step_size = 1e-4f;
    // If a point has sigma < this amount, considers sigma = 0
    float sigma_thresh = 1e-2f;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh = 1e-2f;
    // Background brightness
    float background_brightness = 1.f;

    // Draw a (rather low-quality) grid to help visualize the octree
    bool show_grid = false;

    // Depth
    bool render_depth = false;

    // Rendering bounding box (relative to outer tree bounding box [0, 1])
    // [minx, miny, minz, maxx, maxy, maxz]
    float render_bbox[6] = {0.f, 0.f, 0.f, 1.f, 1.f, 1.f};

    // Whether to show a specific basis function only
    int basis_id = -1;

    // Basis dim
    int _basis_dim;
};

}  // namespace volrend
