#pragma once
#include "volrend/camera.hpp"
#include "volrend/cuda/common.cuh"

namespace volrend {
namespace device {

__device__ __inline__ static void query_single_from_root(const float* data,
                                      const int32_t* child,
                                      const float* __restrict__ xyz,
                                      float* __restrict__ out,
                                      const int N) {

    float fN = (float) N;
    float q[3] = {xyz[0], xyz[1], xyz[2]};
    if (outside_grid<float>(q)) {
        out[0] = out[1] = out[2] = out[3] = 0.f;
        return;
    }
    while (true) {
        // Find index of query point, in {0, ... N^3}
        int32_t index = 0;
        for (int i = 0; i < 3; ++i) {
            float idx_dimi = floorf(q[i] * fN);
            index = index * N + (int32_t) idx_dimi;
            q[i] = q[i] * fN - idx_dimi;
        }

        // Find child offset
        int32_t skip = child[index];

        // Add to output
        if (skip == 0) {
            const float* val = data + index * 4;
            for (int i = 0; i < 4; ++i) out[i] = val[i];
            break;
        }

        child += skip * N * N * N;
        data += skip * N * N * N * 4;
    }
}

}  // namespace device
}  // namespace volrend
