#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_fp16.h>

#include "volrend/cuda/common.cuh"
#include "volrend/cuda/rt_core.cuh"
#include "volrend/render_options.hpp"
#include "volrend/cuda/data_spec.cuh"

namespace volrend {

#define MAX3(a, b, c) max(max(a, b), c)
#define MIN3(a, b, c) min(min(a, b), c)

namespace {
template<typename scalar_t>
__host__ __device__ __inline__ static void screen2worlddir(
        int ix, int iy,
        const CameraSpec& cam,
        scalar_t* out,
        scalar_t* cen) {
    scalar_t xyz[3] ={ (ix - 0.5f * cam.width) / cam.fx,
                    -(iy - 0.5f * cam.height) / cam.fy, -1.0f};
    _mv3(cam.transform, xyz, out);
    _normalize(out);
    _copy3(cam.transform + 9, cen);
}
template<typename scalar_t>
__host__ __device__ __inline__ void maybe_world2ndc(
        const TreeSpec& tree,
        scalar_t* __restrict__ dir,
        scalar_t* __restrict__ cen) {
    if (tree.ndc_width <= 0)
        return;
    scalar_t t = -(1.f + cen[2]) / dir[2];
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = -((2 * tree.ndc_focal) / tree.ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * tree.ndc_focal) / tree.ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 / cen[2];

    cen[0] = -((2 * tree.ndc_focal) / tree.ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * tree.ndc_focal) / tree.ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 / cen[2];

    _normalize(dir);
}

template<typename scalar_t>
__host__ __device__ __inline__ void rodrigues(
        const scalar_t* __restrict__ aa,
        scalar_t* __restrict__ dir) {
    scalar_t angle = _norm(aa);
    if (angle < 1e-6) return;
    scalar_t k[3];
    for (int i = 0; i < 3; ++i) k[i] = aa[i] / angle;
    scalar_t cos_angle = cos(angle), sin_angle = sin(angle);
    scalar_t cross[3];
    _cross3(k, dir, cross);
    scalar_t dot = _dot3(k, dir);
    for (int i = 0; i < 3; ++i) {
        dir[i] = dir[i] * cos_angle + cross[i] * sin_angle + k[i] * dot * (1.0 - cos_angle);
    }
}

}  // namespace

namespace device {

// Primary rendering kernel
__global__ static void render_kernel(
        cudaSurfaceObject_t surf_obj,
        cudaSurfaceObject_t surf_obj_depth,
        CameraSpec cam,
        TreeSpec tree,
        RenderOptions opt,
        bool offscreen) {
    CUDA_GET_THREAD_ID(idx, cam.width * cam.height);
    const int x = idx % cam.width, y = idx / cam.width;

    uint8_t rgbx_init[4];
    float t_max = 1e9f;
    if (!offscreen) {
        surf2Dread(reinterpret_cast<uint32_t*>(rgbx_init), surf_obj, x * 4,
                    y, cudaBoundaryModeZero);
        surf2Dread(&t_max, surf_obj_depth, x * sizeof(float), y, cudaBoundaryModeZero);
    }

    float dir[3], cen[3], out[4];

    screen2worlddir(x, y, cam, dir, cen);
    float vdir[3] = {dir[0], dir[1], dir[2]};
    maybe_world2ndc(tree, dir, cen);
    for (int i = 0; i < 3; ++i) {
        cen[i] = tree.offset[i] + tree.scale[i] * cen[i];
    }

    rodrigues(opt.rot_dirs, vdir);

    trace_ray(tree, dir, vdir, cen, opt, t_max, out);
    // Compositing with existing color
    const float nalpha = 1.f - out[3];
    if (offscreen) {
        const float remain = opt.background_brightness * nalpha;
        out[0] += remain;
        out[1] += remain;
        out[2] += remain;
    } else {
        out[0] += rgbx_init[0] / 255.f * nalpha;
        out[1] += rgbx_init[1] / 255.f * nalpha;
        out[2] += rgbx_init[2] / 255.f * nalpha;
    }

    // Output pixel color
    uint8_t rgbx[4] = { uint8_t(out[0] * 255), uint8_t(out[1] * 255), uint8_t(out[2] * 255), 255 };
    surf2Dwrite(
            *reinterpret_cast<uint32_t*>(rgbx),
            surf_obj,
            x * 4,
            y,
            cudaBoundaryModeZero); // squelches out-of-bound writes
}

}  // namespace device

__host__ void launch_renderer(const N3Tree& tree,
        const Camera& cam, const RenderOptions& options, cudaArray_t& image_arr,
        cudaArray_t& depth_arr,
        cudaStream_t stream,
        bool offscreen) {
    cudaSurfaceObject_t surf_obj = 0, surf_obj_depth = 0;
    {
        struct cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = image_arr;
        cudaCreateSurfaceObject(&surf_obj, &res_desc);
    }
    if (!offscreen) {
        {
            struct cudaResourceDesc res_desc;
            memset(&res_desc, 0, sizeof(res_desc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = depth_arr;
            cudaCreateSurfaceObject(&surf_obj_depth, &res_desc);
        }
    }

    // 128 is weirdly faster for me than 1024
    // Not sure if this scales to a good GPU
    const int N_CUDA_THREADS = 256;

    const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, N_CUDA_THREADS);
    device::render_kernel<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            surf_obj,
            surf_obj_depth,
            CameraSpec::load(cam),
            TreeSpec::load(tree),
            options,
            offscreen);
}
}  // namespace volrend
