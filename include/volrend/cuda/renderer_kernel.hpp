#pragma once

#include <cuda_runtime.h>
#include "volrend/n3tree.hpp"
#include "volrend/camera.hpp"
#include "volrend/render_options.hpp"

namespace volrend {
__host__ void launch_renderer(const N3Tree& tree, const Camera& cam,
                              const RenderOptions& options,
                              cudaArray_t& image_arr, cudaArray_t& depth_arr,
                              cudaStream_t stream, bool offscreen = false);
}  // namespace volrend
