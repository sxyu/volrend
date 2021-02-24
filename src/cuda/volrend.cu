#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>

#include "volrend/n3tree.hpp"
#include "volrend/cuda/rt_core.cuh"
#include "volrend/render_options.hpp"

namespace volrend {

#define MAX3(a, b, c) max(max(a, b), c)
#define MIN3(a, b, c) min(min(a, b), c)

// __host__ __device__ __inline__ static void world2screen(
//         const float* __restrict__ xyz,
//         float focal_x,
//         float focal_y,
//         const float* __restrict__ transform,
//         float* out) {
//     float x, y, z;
//     x = xyz[0] - transform[9];
//     y = xyz[1] - transform[10];
//     z = xyz[2] - transform[11];
//     float zt;
//     out[0] = transform[0] * x + transform[1] * y + transform[2] * z;
//     out[1] = transform[3] * x + transform[4] * y + transform[5] * z;
//     zt     = transform[6] * x + transform[7] * y + transform[8] * z;
//
//     out[0] *= focal_x / zt;
//     out[1] *= focal_y / zt;
// }

template<typename scalar_t>
__host__ __device__ __inline__ static scalar_t _norm(
        scalar_t* dir) {
    return sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
}

template<typename scalar_t>
__host__ __device__ __inline__ static void _normalize(
        scalar_t* dir) {
    scalar_t norm = _norm(dir);
    dir[0] /= norm; dir[1] /= norm; dir[2] /= norm;
}

__host__ __device__ __inline__ static void screen2worlddir(
        int ix, int iy, float focal,
        int width, int height,
        const float* __restrict__ transform,
        float* out,
        float* cen) {
    float x = (ix - 0.5 * width) / focal;
    float y = -(iy - 0.5 * height) / focal;
    float z = sqrtf(x * x + y * y + 1.0);
    x /= z;
    y /= z;
    z = -1.0f / z;

    out[0] = transform[0] * x + transform[3] * y + transform[6] * z;
    out[1] = transform[1] * x + transform[4] * y + transform[7] * z;
    out[2] = transform[2] * x + transform[5] * y + transform[8] * z;
    cen[0] = transform[9]; cen[1] = transform[10]; cen[2] = transform[11];
}
template<typename scalar_t>
__host__ __device__ __inline__ void world2ndc(
        int ndc_width, int ndc_height, scalar_t ndc_focal,
        scalar_t* __restrict__ dir,
        scalar_t* __restrict__ cen, scalar_t near = 1.f) {
    scalar_t t = -(near + cen[2]) / dir[2];
    for (int i = 0; i < 3; ++i) {
        cen[i] = cen[i] + t * dir[i];
    }

    dir[0] = -((2 * ndc_focal) / ndc_width) * (dir[0] / dir[2] - cen[0] / cen[2]);
    dir[1] = -((2 * ndc_focal) / ndc_height) * (dir[1] / dir[2] - cen[1] / cen[2]);
    dir[2] = -2 * near / cen[2];

    cen[0] = -((2 * ndc_focal) / ndc_width) * (cen[0] / cen[2]);
    cen[1] = -((2 * ndc_focal) / ndc_height) * (cen[1] / cen[2]);
    cen[2] = 1 + 2 * near / cen[2];

    _normalize(dir);
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

namespace device {
// Primary rendering kernel
__global__ static void render_kernel(
        cudaSurfaceObject_t surf_obj,
        const int width,
        const int height,
        float focal,
        const float* __restrict__ transform,
        const float* __restrict__ tree_data,
        const int32_t* __restrict__ tree_child,
        const float* __restrict__ tree_offset,
        const float* __restrict__ tree_scale,
        int tree_N,
        int data_dim,
        int sh_order,
        float ndc_width,
        float ndc_height,
        float ndc_focal,
        float step_size,
        float stop_thresh,
        float sigma_thresh,
        float background_brightness,
        bool show_cuda) {
    CUDA_GET_THREAD_ID(idx, width * height);
    const int x   = idx % width;
    const int y   = idx / width;
    // if (x > 0 || y > 0) return;

    float dir[3], cen[3], out[3];
    screen2worlddir(x, y, focal, width, height, transform, dir,
            cen);
    float vdir[3] = {dir[0], dir[1], dir[2]};
    if (ndc_width > 0.f) {
        world2ndc(ndc_width, ndc_height, ndc_focal, dir, cen);
    }
    for (int i = 0; i < 3; ++i) {
        cen[i] = tree_offset[i] + tree_scale[i] * cen[i];
    }

    const float delta_scale = _get_delta_scale(tree_scale, dir);
    trace_ray(tree_data, tree_child, tree_N, data_dim, sh_order,
            dir,
            vdir,
            cen, step_size, stop_thresh,
            sigma_thresh,
            background_brightness,
            delta_scale,
            show_cuda, out);

    // pixel color
    uint8_t rgbx[4];
    rgbx[0]  = uint8_t(out[0] * 255);
    rgbx[1] = uint8_t(out[1] * 255);
    rgbx[2]  = uint8_t(out[2] * 255);
    rgbx[3] = 255;

    surf2Dwrite(
            *reinterpret_cast<uint32_t*>(rgbx),
            surf_obj,
            x * 4,
            y,
            cudaBoundaryModeZero); // squelches out-of-bound writes
}

}  // namespace device

__host__ void launch_renderer(const N3Tree& tree,
        const Camera& cam, const RenderOptions& options, cudaArray_t& arr,
        cudaStream_t stream) {
    struct cudaResourceDesc res_desc;

    // Init surface object
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = arr;

    cudaSurfaceObject_t surf_obj = 0;
    cudaCreateSurfaceObject(&surf_obj, &res_desc);

    // 128 is weirdly faster for me than 1024
    // Not sure if this scales to a good GPU
    const int N_CUDA_THREADS = 128;

    const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height, N_CUDA_THREADS);
    device::render_kernel<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            surf_obj,
            cam.width, cam.height,
            cam.focal, cam.device.transform,
            tree.device.data,
            tree.device.child,
            tree.device.offset,
            tree.device.scale,
            tree.N,
            tree.data_dim,
            tree.sh_order,
            tree.use_ndc ? tree.ndc_width : -1,
            tree.ndc_height,
            tree.ndc_focal,
            options.step_size,
            options.stop_thresh,
            options.sigma_thresh,
            options.background_brightness,
            options.show_grid);
}
}  // namespace volrend
