#pragma once

struct RenderOptions {
    float step_size = 1.0f / 640.0;
    // If a point has sigma < this amount, considers sigma = 0
    float sigma_thresh = 0.f;
    // If remaining light intensity/alpha < this amount stop marching
    float stop_thresh = 0.02f;
    // Background brightness
    float background_brightness = 1.f;
};
