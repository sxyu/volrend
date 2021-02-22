#pragma once
#include "volrend/cuda/n3tree_query.cuh"
#include <cmath>

namespace volrend {
namespace device {
namespace {

__device__ __inline__ void _precalc_sh(
    const int order,
    const float* __restrict__ dir,
    float* __restrict__ out_mult) {

    const float C0 = 0.5 * sqrt(1.0 / M_PI);
    const float C1 = sqrt(3 / (4 * M_PI));
    const float C2[] = {0.5f * sqrtf(15.0f / M_PI),
        -0.5f * sqrtf(15.0f / M_PI),
        0.25f * sqrtf(5.0f / M_PI),
        -0.5f * sqrtf(15.0f / M_PI),
        0.25f * sqrtf(15.0f / M_PI)};

    const float C3[] = {-0.25f * sqrtf(35 / (2 * M_PI)),
        0.5f * sqrtf(105 / M_PI),
        -0.25f * sqrtf(21/(2 * M_PI)),
        0.25f * sqrtf(7 / M_PI),
        -0.25f * sqrtf(21/(2 * M_PI)),
        0.25f * sqrtf(105 / M_PI),
        -0.25f * sqrtf(35/(2 * M_PI))
    };

    out_mult[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    out_mult[1] = - C1 * y;
    out_mult[2] = C1 * z;
    out_mult[3] = -C1 * x;
    if (order > 1) {
        const float xx = x * x, yy = y * y, zz = z * z;
        out_mult[4] = C2[0] * x * y;
        out_mult[5] = C2[1] * y * z;
        out_mult[6] = C2[2] * (2.0 * zz - xx - yy);
        out_mult[7] = C2[3] * x * z;
        out_mult[8] = C2[4] * (xx - yy);
        if (order > 2) {
            const float tmp_zzxxyy = 4 * zz - xx - yy;
            out_mult[9] = C3[0] * y * (3 * xx - yy);
            out_mult[10] = C3[1] * x * y * z;
            out_mult[11] = C3[2] * y * tmp_zzxxyy;
            out_mult[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
            out_mult[13] = C3[4] * x * tmp_zzxxyy;
            out_mult[14] = C3[5] * z * (xx - yy);
            out_mult[15] = C3[6] * x * (xx - 3 * yy);
        }
    }
}

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
        t2 = t1 +  _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

__device__ __inline__ void trace_ray(
        const float* __restrict__ tree_data,
        const int32_t* __restrict__ tree_child,
        int tree_N,
        int data_dim,
        int sh_order,
        const float* __restrict__ dir,
        const float* __restrict__ vdir,
        const float* __restrict__ cen,
        float step_size,
        float stop_thresh,
        float sigma_thresh,
        float background_brightness,
        float delta_scale,
        bool show_miss,
        float* __restrict__ out) {

    float tmin, tmax;
    float invdir[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / (dir[i] + 1e-9);
    }
    _dda_unit(cen, invdir, &tmin, &tmax);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        if (show_miss) {
            out[1] = 0.f;
            out[0] = out[2] = 1.f;
        } else {
            out[0] = out[1] = out[2] = background_brightness;
        }
        return;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
        float pos[3], tmp;
        const float* tree_val;
        float sh_mult[16];
        if (sh_order >= 0) {
            _precalc_sh(sh_order, vdir, sh_mult);
        }

        float light_intensity = 1.f;
        // int n_steps = (int) ceilf((tmax - tmin) / step_size);
        float t = tmin;
        const int n_coe = (sh_order + 1) * (sh_order + 1);
        float cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = cen[j] + t * dir[j];
            }

            query_single_from_root(tree_data, tree_child,
                    pos, &tree_val, &cube_sz, tree_N, data_dim);

            float att;
            float subcube_tmin, subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmin, &subcube_tmax);

            const float t_subcube = (subcube_tmax - subcube_tmin) / cube_sz;
            const float delta_t = t_subcube + step_size;
            if (tree_val[data_dim - 1] > sigma_thresh) {
                att = expf(-delta_t * delta_scale * tree_val[data_dim - 1]);
                const float weight = light_intensity * (1.f - att);

                if (sh_order >= 0) {
#pragma unroll 3
                    for (int t = 0; t < 3; ++ t) {
                        int off = t * n_coe;
                        tmp = sh_mult[0] * tree_val[off] +
                            sh_mult[1] * tree_val[off + 1] +
                            sh_mult[2] * tree_val[off + 2];
#pragma unroll 6
                        for (int i = 3; i < n_coe; ++i) {
                            tmp += sh_mult[i] * tree_val[off + i];
                        }
                        out[t] += weight / (1.f + expf(-tmp));
                    }
                } else {
                    for (int j = 0; j < 3; ++j) {
                        out[j] += tree_val[j] * weight;
                    }
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
