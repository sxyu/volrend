#pragma once
#include "volrend/camera.hpp"
#include "volrend/cuda/common.cuh"

namespace volrend {
namespace device {

__device__ __inline__ static void query_single_from_root(const float* __restrict__ data,
                                      const int32_t* __restrict__ child,
                                      float* __restrict__ xyz,
                                      float* __restrict__ out,
                                      float* __restrict__ cube_sz,
                                      const int N) {

    float fN = (float) N, inv_fN = 1.f / fN;
    xyz[0] = max(min(xyz[0], 1.f - 1e-6f), 0.f);
    xyz[1] = max(min(xyz[1], 1.f - 1e-6f), 0.f);
    xyz[2] = max(min(xyz[2], 1.f - 1e-6f), 0.f);
    int ptr = 0;
    *cube_sz = inv_fN;
    while (true) {
        // Find index of query point, in {0, ... N^3}
        int32_t index = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            float idx_dimi = floorf(xyz[i] * fN);
            index = index * N + (int32_t) idx_dimi;
            xyz[i] = xyz[i] * fN - idx_dimi;
        }

        // Find child offset
        int32_t skip = child[ptr + index];

        // Add to output
        if (skip == 0) {
            const float* val = data + (ptr + index) * 4;
            out[0] = val[0];
            out[1] = val[1];
            out[2] = val[2];
            out[3] = val[3];
            break;
        }
        *cube_sz *= inv_fN;

        ptr += skip * N * N * N;
    }
}

}  // namespace device
}  // namespace volrend
