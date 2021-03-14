#pragma once
#include "volrend/cuda/n3tree_query.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include "volrend/common.hpp"
#include "volrend/data_format.hpp"
#include "volrend/render_options.hpp"
#include "volrend/cuda/common.cuh"
#include "volrend/cuda/data_spec.cuh"
#include "volrend/cuda/lumisphere.cuh"

namespace volrend {
namespace device {
namespace {

template<typename scalar_t>
__device__ __inline__ void _dda_world(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir,
        scalar_t* __restrict__ tmin,
        scalar_t* __restrict__ tmax,
        const float* __restrict__ render_bbox) {
    scalar_t t1, t2;
    *tmin = 0.0;
    *tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = (render_bbox[i] - cen[i]) * _invdir[i];
        t2 = (render_bbox[i + 3] - cen[i]) * _invdir[i];
        *tmin = max(*tmin, min(t1, t2));
        *tmax = min(*tmax, max(t1, t2));
    }
}

template<typename scalar_t>
__device__ __inline__ void _dda_unit(
        const scalar_t* __restrict__ cen,
        const scalar_t* __restrict__ _invdir,
        scalar_t* __restrict__ tmax) {
    scalar_t t1, t2;
    *tmax = 1e4;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * _invdir[i];
        t2 = t1 +  _invdir[i];
        *tmax = min(*tmax, max(t1, t2));
    }
}

template <typename scalar_t>
__device__ __inline__ scalar_t _get_delta_scale(
    const scalar_t* __restrict__ scaling,
    scalar_t* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    scalar_t delta_scale = 1.f / _norm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

template<typename scalar_t>
__device__ __inline__ void trace_ray(
        const TreeSpec& tree,
        scalar_t* __restrict__ dir,
        const scalar_t* __restrict__ vdir,
        const scalar_t* __restrict__ cen,
        RenderOptions opt,
        float tmax_bg,
        scalar_t* __restrict__ out) {

    const float delta_scale = _get_delta_scale(
            tree.scale, /*modifies*/ dir);
    tmax_bg /= delta_scale;

    scalar_t tmin, tmax;
    scalar_t invdir[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir[i] = 1.f / (dir[i] + 1e-9);
    }
    _dda_world(cen, invdir, &tmin, &tmax, opt.render_bbox);
    tmax = min(tmax, tmax_bg);

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        if (opt.render_depth) {
            out[0] = out[1] = out[2] = 0.f;
            out[3] = 1.f;
        } else {
            out[3] = 0.f;
        }
        return;
    } else {
        out[0] = out[1] = out[2] = 0.0f;
        scalar_t pos[3], tmp;
        const half* tree_val;
        scalar_t basis_fn[VOLREND_GLOBAL_BASIS_MAX];
        maybe_precalc_basis(tree, vdir, basis_fn);

        scalar_t light_intensity = 1.f;
        scalar_t t = tmin;
        scalar_t cube_sz;
        while (t < tmax) {
            for (int j = 0; j < 3; ++j) {
                pos[j] = cen[j] + t * dir[j];
            }

            query_single_from_root(tree, pos, &tree_val, &cube_sz);

            scalar_t att;
            scalar_t subcube_tmax;
            _dda_unit(pos, invdir, &subcube_tmax);
            if (opt.show_grid) {
                scalar_t max3 = max(max(pos[0], pos[1]), pos[2]);
                scalar_t min3 = min(min(pos[0], pos[1]), pos[2]);
                scalar_t mid3 = pos[0] + pos[1] + pos[2] - min3 - max3;
                const scalar_t edge_draw_thresh = 2e-2f;
                int n_edges = (abs(min3) < edge_draw_thresh) +
                              ((1.f - abs(max3)) < edge_draw_thresh) +
                              (abs(mid3) < edge_draw_thresh ||
                               (1.f - abs(mid3)) < edge_draw_thresh);

                if (n_edges >= 2) {
                    const float remain = light_intensity * .3f;
                    out[0] += remain; out[1] += remain; out[2] += remain;
                    out[3] = 1.f;
                    return;
                }
            }

            const scalar_t t_subcube = subcube_tmax / cube_sz;
            const scalar_t delta_t = t_subcube + opt.step_size;
            if (__half2float(tree_val[tree.data_dim - 1]) > opt.sigma_thresh) {
                att = expf(-delta_t * delta_scale * __half2float(tree_val[tree.data_dim - 1]));
                const scalar_t weight = light_intensity * (1.f - att);

                if (opt.render_depth) {
                    out[0] += weight * t;
                } else {
                    if (tree.data_format.basis_dim >= 0) {
                        if (~opt.basis_id) {
#pragma unroll 3
                            for (int t = 0; t < 3; ++ t) {
                                int off = t * tree.data_format.basis_dim;
                                out[t] += weight / (1.f + expf(-
                                    basis_fn[opt.basis_id] * __half2float(
                                        tree_val[off + opt.basis_id])
                                ));
                            }
                        } else {
#pragma unroll 3
                            for (int t = 0; t < 3; ++ t) {
                                int off = t * tree.data_format.basis_dim;
                                tmp = basis_fn[0] * __half2float(tree_val[off]) +
                                    basis_fn[1] * __half2float(tree_val[off + 1]) +
                                    basis_fn[2] * __half2float(tree_val[off + 2]) +
                                    basis_fn[3] * __half2float(tree_val[off + 3]);
#pragma unroll 6
                                for (int i = 4; i < tree.data_format.basis_dim; ++i) {
                                    tmp += basis_fn[i] * __half2float(tree_val[off + i]);
                                }
                                out[t] += weight / (1.f + expf(-tmp));
                            }
                        }
                    } else {
                        for (int j = 0; j < 3; ++j) {
                            out[j] += __half2float(tree_val[j]) * weight;
                        }
                    }
                }

                light_intensity *= att;

                if (light_intensity < opt.stop_thresh) {
                    // Almost full opacity, stop
                    if (opt.render_depth) {
                        out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
                    }
                    scalar_t scale = 1.f / (1.f - light_intensity);
                    out[0] *= scale; out[1] *= scale; out[2] *= scale;
                    out[3] = 1.f;
                    return;
                }
            }
            t += delta_t;
        }
        if (opt.render_depth) {
            out[0] = out[1] = out[2] = min(out[0] * 0.3f, 1.0f);
            out[3] = 1.f;
        } else {
            out[3] = 1.f - light_intensity;
        }
    }
}

}  // namespace
}  // namespace device
}  // namespace volrend
