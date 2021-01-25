#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>

#include "volrend/n3tree.hpp"
#include "volrend/cuda/n3tree_query.cuh"

namespace volrend {
namespace device {

#define MAX3(a, b, c) max(max(a, b), c)
#define MIN3(a, b, c) min(min(a, b), c)

namespace {
__device__ __inline__ void screen2worlddir(
        float x, float y, float focal_norm_x,
        float focal_norm_y,
        const float* __restrict__ transform,
        float* out) {
    x /= focal_norm_x;
    y /= focal_norm_y;
    float z = sqrtf(x * x + y * y + 1.0);
    x /= z;
    y /= z;
    z = 1.0f / z;

    out[0] = transform[0] * x + transform[3] * y + transform[6] * z;
    out[1] = transform[1] * x + transform[4] * y + transform[7] * z;
    out[2] = transform[2] * x + transform[5] * y + transform[8] * z;
}

__device__ void trace_ray_naive(
        const float* __restrict__ tree_data,
        const int32_t* __restrict__ tree_child,
        int tree_N,
        const float* __restrict__ dir,
        const float* __restrict__ cen,
        float step_size,
        float sigma_thresh,
        float stop_thresh,
        float* __restrict__ out) {

    float invdir, t1, t2;
    float tmin = 0.0f, tmax = 1e6f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        invdir = 1.f / dir[i];
        t1 = - cen[i] * invdir;
        t2 = (1.f - cen[i])* invdir;
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }

    if (tmax < 0 || tmin > tmax) {
        // Ray doesn't hit box
        tmin = tmax = 0.f;
    }

    float pos[3];
    float rgba[4];
    out[0] = out[1] = out[2] = 0.0f;
    float light_intensity = 1.f;
    int n_steps = (int) ceilf((tmax - tmin) / step_size);
    for (int i = 0 ; i < n_steps; ++i) {
        const float dist = tmin + i * step_size;
        for (int j = 0; j < 3; ++j) {
            pos[j] = cen[j] + dist * dir[j];
        }

        query_single_from_root(tree_data, tree_child,
                pos, rgba, tree_N);
        if (rgba[3] < sigma_thresh)
            continue;
        for (int j = 0; j < 3; ++j) {
            out[j] += rgba[j] * (light_intensity * rgba[3]);
        }
        light_intensity *= 1.f - rgba[3];

        if (light_intensity < stop_thresh) {
            // Almost full opacity, stop
            float scale = 1.f - light_intensity;
            out[0] /= scale;
            out[1] /= scale;
            out[2] /= scale;
            light_intensity = 0.f;
            break;
        }
    }
    out[0] += light_intensity;
    out[1] += light_intensity;
    out[2] += light_intensity;
}

// Primary rendering kernel
__global__ void render_kernel(
        cudaSurfaceObject_t surf_obj,
        const int width,
        const int height,
        float focal_norm_x,
        float focal_norm_y,
        const float* __restrict__ transform,
        const float* __restrict__ tree_data,
        const int32_t* __restrict__ tree_child,
        int tree_N,
        float step_size,
        float sigma_thresh,
        float stop_thresh) {
    CUDA_GET_THREAD_ID(idx, width * height);
    const int x   = idx % width;
    const int y   = idx / width;

    const float x_norm = x / (0.5f * width) - 1.0f;
    const float y_norm = y / (0.5f * height) - 1.0f;

    float dir[3];
    screen2worlddir(x_norm, y_norm, focal_norm_x, focal_norm_y, transform, dir);

    float out[3];
    trace_ray_naive(tree_data, tree_child, tree_N,
            dir, transform + 9, step_size, sigma_thresh, stop_thresh, out);

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

}  // namespace
}  // namespace device

__host__ void launch_renderer(const N3Tree& tree,
        const Camera& cam, cudaArray_t& arr,
        float step_size, float sigma_thresh,
        float stop_thresh, cudaStream_t stream) {
    struct cudaResourceDesc res_desc;

    // Init surface object
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = arr;

    cudaSurfaceObject_t surf_obj = 0;
    cudaCreateSurfaceObject(&surf_obj, &res_desc);

    const int blocks = N_BLOCKS_NEEDED(cam.width * cam.height);
    float focal_norm_x = cam.focal / (cam.width * 0.5f);
    float focal_norm_y = cam.focal / (cam.height * 0.5f);
    device::render_kernel<<<blocks, N_CUDA_THREADS, 0, stream>>>(
            surf_obj,
            cam.width, cam.height,
            focal_norm_x, focal_norm_y, cam.device.transform,
            tree.device.data,
            tree.device.child,
            tree.N,
            step_size,
            sigma_thresh,
            stop_thresh);
}
}  // namespace volrend
