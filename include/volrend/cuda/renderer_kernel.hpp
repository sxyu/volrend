#pragma once

#include <cuda_runtime.h>
#include "volrend/n3tree.hpp"
#include "volrend/camera.hpp"

namespace volrend {
__host__ void launch_renderer(const N3Tree& tree, const Camera& cam,
                              cudaArray_t& arr, float step_size,
                              float sigma_thresh, float stop_thresh,
                              cudaStream_t stream);
}
