#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>

#include "volrend/n3tree.hpp"
#include "volrend/cuda/n3tree_query.cuh"

namespace volrend {
namespace device {

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

    out[0] = transform[0] * x + transform[1] * y + transform[2] * z;
    out[1] = transform[4] * x + transform[5] * y + transform[6] * z;
    out[2] = transform[8] * x + transform[9] * y + transform[10] * z;
}

__device__ void trace_ray_naive(
        const float* __restrict__ tree_data,
        const int32_t* __restrict__ tree_child,
        int tree_N,
        const float* __restrict__ dir,
        const float* __restrict__ transform,
        float step_size,
        int max_n_steps,
        float* __restrict__ out) {

    float pos[3];
    float rgba[4];
    out[0] = out[1] = out[2] = 0.0f;
    float light_intensity = 1.f;
    for (int i = 0 ; i < max_n_steps; ++i) {
        const float dist = i * step_size;
        for (int j = 0; j < 3; ++j) {
            pos[j] = transform[3 + 4 * j] + dist * dir[j];
        }

        query_single_from_root(tree_data, tree_child,
                pos, rgba, tree_N);
        for (int j = 0; j < 3; ++j) {
            out[j] += rgba[j] * (light_intensity * rgba[3]);
        }
        light_intensity *= 1.f - rgba[3];
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
        int max_n_steps) {
    CUDA_GET_THREAD_ID(idx, width * height);
    const int x   = idx % width;
    const int y   = idx / width;

    const float x_norm = x / (0.5f * width) - 1.0f;
    const float y_norm = y / (0.5f * height) - 1.0f;

    // const float origin[3] = {transform[3], transform[7], transform[11]};

    float dir[3];
    screen2worlddir(x_norm, y_norm, focal_norm_x, focal_norm_y, transform, dir);

    float out[3];
    trace_ray_naive(tree_data, tree_child, tree_N,
            dir, transform, step_size, max_n_steps, out);

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
        float step_size, int max_n_steps, cudaStream_t stream) {
    struct cudaResourceDesc res_desc;

    // Init surface memory
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
            max_n_steps);
}
}  // namespace volrend
