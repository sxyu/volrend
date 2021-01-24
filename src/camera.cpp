#include "volrend/camera.hpp"
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "volrend/cuda/common.cuh"

namespace volrend {

Camera::Camera(int width, int height, float focal)
    : width(width), height(height), focal(focal) {
    center = {0.5f, 0.0f, 0.5f};
    v_forward = {0.0f, 1.0f, 0.0f};
    v_world_down = {0.0f, 0.0f, -1.0f};
    _update();
}

Camera::~Camera() { free_cuda(); }

void Camera::_update(bool copy_cuda) {
    v_right = glm::normalize(glm::cross(v_world_down, v_forward));
    v_down = glm::cross(v_forward, v_right);
    transform[0] = v_right;
    transform[1] = v_down;
    transform[2] = v_forward;
    transform[3] = center;

    if (copy_cuda) {
        if (device.transform == nullptr) {
            cuda(Malloc((void**)&device.transform, 12 * sizeof(transform[0])));
        }
        cuda(MemcpyAsync(device.transform, glm::value_ptr(transform),
                         12 * sizeof(transform[0]), cudaMemcpyHostToDevice));
    }
}

void Camera::begin_drag(float x, float y, bool is_pan, bool about_origin) {
    is_dragging_ = true;
    drag_start_ = glm::vec2(x, y);
    drag_start_forward_ = v_forward;
    drag_start_right_ = v_right;
    drag_start_down_ = v_down;
    drag_start_center_ = center;
    is_panning_ = is_pan;
    about_origin_ = about_origin;
}
void Camera::drag_update(float x, float y) {
    if (!is_dragging_) return;
    glm::vec2 drag_curr(x, y);
    glm::vec2 delta = drag_curr - drag_start_;
    delta *= -2.f / std::max(width, height);
    if (is_panning_) {
        center = drag_start_center_ + delta.x * drag_start_right_ +
                 delta.y * drag_start_down_;
    } else {
        if (about_origin_) delta *= -1.f;
        glm::mat4 m(1.0f);
        m = glm::rotate(m, delta.x, v_world_down);
        m = glm::rotate(m, -delta.y, drag_start_right_);

        glm::vec3 v_forward_new = m * glm::vec4(drag_start_forward_, 1.f);

        float dot = glm::dot(glm::cross(v_world_down, v_forward_new),
                             drag_start_right_);
        // Prevent flip over pole
        if (dot < 0.f) return;
        v_forward = v_forward_new;

        if (about_origin_) {
            const glm::vec3 world_cen(0.5f, 0.5f, 0.5f);
            center =
                glm::vec3(m * glm::vec4(drag_start_center_ - world_cen, 1.f)) +
                world_cen;
        }
        _update(false);
    }
}
void Camera::end_drag() { is_dragging_ = false; }

void Camera::free_cuda() {
    if (device.transform != nullptr) {
        cuda(Free(device.transform));
    }
}

}  // namespace volrend
