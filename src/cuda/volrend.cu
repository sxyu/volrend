#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <glm/gtc/type_ptr.hpp>

#include "volrend/n3tree.hpp"
#include "volrend/cuda/rt_naive.cuh"
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

__host__ __device__ __inline__ static void screen2worlddir(
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

namespace device {
// Primary rendering kernel
__global__ static void render_kernel(
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
        float stop_thresh,
        float background_brightness) {
    CUDA_GET_THREAD_ID(idx, width * height);
    const int x   = idx % width;
    const int y   = idx / width;

    const float x_norm = x / (0.5f * width) - 1.0f;
    const float y_norm = y / (0.5f * height) - 1.0f;

    float dir[3], out[3];
    screen2worlddir(x_norm, y_norm, focal_norm_x, focal_norm_y, transform, dir);

    trace_ray_naive(tree_data, tree_child, tree_N,
            dir, transform + 9, step_size, stop_thresh, background_brightness, out);

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
            options.step_size,
            options.stop_thresh,
            options.background_brightness);
}
}  // namespace volrend
