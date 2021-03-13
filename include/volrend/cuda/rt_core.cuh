#pragma once
#include "volrend/cuda/n3tree_query.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include "volrend/common.hpp"
#include "volrend/data_format.hpp"
#include "volrend/render_options.hpp"
#include "volrend/cuda/common.cuh"
#include "volrend/cuda/data_spec.cuh"

namespace volrend {
namespace device {
namespace {

// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __constant__ const float C4[] = {
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
};

template<typename scalar_t>
__device__ __inline__ void maybe_precalc_basis(
    const TreeSpec& tree,
    const scalar_t* __restrict__ dir,
    scalar_t* __restrict__ out) {
    const int basis_dim = tree.data_format.basis_dim;
    switch(tree.data_format.format) {
        case DataFormat::ASG:
            {
                // UNTESTED ASG
                const scalar_t* ptr = tree.extra;
                for (int i = 0; i < basis_dim; ++i) {
                    const scalar_t* ptr_mu_x = ptr + 2, * ptr_mu_y = ptr + 5,
                                  * ptr_mu_z = ptr + 8;
                    scalar_t S = _dot3(dir, ptr_mu_z);
                    scalar_t dot_x = _dot3(dir, ptr_mu_x);
                    scalar_t dot_y = _dot3(dir, ptr_mu_y);
                    out[i] = S * expf(-ptr[0] * dot_x * dot_x
                                      -ptr[1] * dot_y * dot_y) / basis_dim;
                    ptr += 11;
                }
            }  // ASG
            break;
        case DataFormat::SG:
            {
                const scalar_t* ptr = tree.extra;
                for (int i = 0; i < basis_dim; ++i) {
                    out[i] = expf(ptr[0] * (_dot3(dir, ptr + 1) - 1.f)) / basis_dim;
                    ptr += 4;
                }
            }  // SG
            break;
        case DataFormat::SH:
            {
                out[0] = C0;
                const scalar_t x = dir[0], y = dir[1], z = dir[2];
                const scalar_t xx = x * x, yy = y * y, zz = z * z;
                const scalar_t xy = x * y, yz = y * z, xz = x * z;
                switch (basis_dim) {
                    case 25:
                        out[16] = C4[0] * xy * (xx - yy);
                        out[17] = C4[1] * yz * (3 * xx - yy);
                        out[18] = C4[2] * xy * (7 * zz - 1.f);
                        out[19] = C4[3] * yz * (7 * zz - 3.f);
                        out[20] = C4[4] * (zz * (35 * zz - 30) + 3);
                        out[21] = C4[5] * xz * (7 * zz - 3);
                        out[22] = C4[6] * (xx - yy) * (7 * zz - 1.f);
                        out[23] = C4[7] * xz * (xx - 3 * yy);
                        out[24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
                        [[fallthrough]];
                    case 16:
                        out[9] = C3[0] * y * (3 * xx - yy);
                        out[10] = C3[1] * xy * z;
                        out[11] = C3[2] * y * (4 * zz - xx - yy);
                        out[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                        out[13] = C3[4] * x * (4 * zz - xx - yy);
                        out[14] = C3[5] * z * (xx - yy);
                        out[15] = C3[6] * x * (xx - 3 * yy);
                        [[fallthrough]];
                    case 9:
                        out[4] = C2[0] * xy;
                        out[5] = C2[1] * yz;
                        out[6] = C2[2] * (2.0 * zz - xx - yy);
                        out[7] = C2[3] * xz;
                        out[8] = C2[4] * (xx - yy);
                        [[fallthrough]];
                    case 4:
                        out[1] = -C1 * y;
                        out[2] = C1 * z;
                        out[3] = -C1 * x;
                }
            }  // SH
            break;

        default:
            // Do nothing
            break;
    }  // switch
}

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
