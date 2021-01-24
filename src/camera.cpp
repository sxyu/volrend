#include "volrend/camera.hpp"
#include <iostream>

#include <cuda_runtime.h>
#include "volrend/cuda/common.cuh"

namespace volrend {

namespace {

std::array<float, 3> vec_cross(const std::array<float, 3>& a,
                               const std::array<float, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
}  // namespace

Camera::Camera(int width, int height, float focal)
    : width(width), height(height), focal(focal) {
    center = {0.5f, 0.5f, 1.0f};
    v_forward = {0.0f, 0.0f, -1.0f};
    v_right = {1.0f, 0.0f, 0.0f};
    update();
}

Camera::~Camera() { free_cuda(); }

void Camera::update() {
    std::array<float, 3> v_down = vec_cross(v_forward, v_right);
    for (int i = 0; i < 3; ++i) {
        transform_[4 * i] = v_right[i];
        transform_[4 * i + 1] = v_down[i];
        transform_[4 * i + 2] = v_forward[i];
        transform_[4 * i + 3] = center[i];
    }
}

void Camera::load_cuda() {
    if (device.transform == nullptr) {
        cuda(Malloc((void**)&device.transform, 12 * sizeof(transform_[0])));
    }
    cuda(MemcpyAsync(device.transform, transform_.data(),
                     12 * sizeof(transform_[0]), cudaMemcpyHostToDevice));
}

void Camera::free_cuda() {
    if (device.transform != nullptr) {
        cuda(Free(device.transform));
    }
}

}  // namespace volrend
