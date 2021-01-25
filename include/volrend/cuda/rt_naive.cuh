#pragma once
#include "volrend/cuda/n3tree_query.cuh"

namespace volrend {
namespace device {
namespace {

__device__ __inline__ void _dda_unit(
        const float* __restrict__ cen,
        const float* __restrict__ _invdir,
        float* __restrict__ tmin,
        float* __restrict__ tmax) {
    float t1, t2;
    *tmin = 0.0f;
    *tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = (1.f - cen[i])* _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

__device__ __inline__ void trace_ray_naive(
        const float* __restrict__ tree_data,
        const int32_t* __restrict__ tree_child,
        int tree_N,
        const float* __restrict__ dir,
        const float* __restrict__ cen,
        float step_size,
        float stop_thresh,
        float background_brightness,
        float* __restrict__ out) {

    float tmin, tmax;
    float invdir[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / dir[i];
    }
    _dda_unit(cen, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        out[0] = out[1] = out[2] = background_brightness;
        // out[1] = 0.f;
        return;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
        float pos[3];
        float rgba[4];
        float light_intensity = 1.f;
        // int n_steps = (int) ceilf((tmax - tmin) / step_size);
        // for (int i = 0 ; i < n_steps; ++i) {
        float t = tmin;
        while (t < tmax) {
            // const float t = tmin + i * step_size;
            for (int j = 0; j < 3; ++j) {
                pos[j] = cen[j] + t * dir[j];
            }

            float cube_sz;
            query_single_from_root(tree_data, tree_child,
                    pos, rgba, &cube_sz, tree_N);

            float att;
            float subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

            const float t_subcube = (subcube_tmax - subcube_tmin) * cube_sz;
            const float delta_t = max(t_subcube, step_size);
            att = expf(-delta_t * rgba[3]);
            const float weight = light_intensity * (1.f - att);

            for (int j = 0; j < 3; ++j) {
                out[j] += rgba[j] * weight;
            }
            light_intensity *= att;

            if (light_intensity < stop_thresh) {
                // Almost full opacity, stop
                float scale = 1.f / (1.f - light_intensity);
                out[0] *= scale;
                out[1] *= scale;
                out[2] *= scale;
                return;
            }
            t += delta_t;
        }
        out[0] += light_intensity * background_brightness;
        out[1] += light_intensity * background_brightness;
        out[2] += light_intensity * background_brightness;
    }
}
}  // namespace
}  // namespace device
}  // namespace volrend
