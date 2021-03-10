#pragma once

#include "volrend/common.hpp"

namespace volrend {

struct RenderOptions {
    float step_size = 1e-4f;
    // If a point has sigma < this amount, considers sigma = 0
    float sigma_thresh = 1e-2f;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh = 1e-2f;
    // Background brightness
    float background_brightness = 1.f;
    // Color rays which do not hit anything in the bounding box magenta
    bool show_grid = false;
};

}  // namespace volrend
