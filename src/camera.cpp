#include "volrend/camera.hpp"
#include <cmath>

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "volrend/cuda/common.cuh"

namespace volrend {

struct Camera::DragState {
    bool is_dragging = false;
    // Pan instead of rotate
    bool is_panning = false;
    // Rotate about (0.5, 0.5, 0.5)
    bool about_origin = false;
    glm::vec2 drag_start;
    // Save state when drag started
    glm::vec3 drag_start_forward, drag_start_right, drag_start_down;
    glm::vec3 drag_start_center;
};

Camera::Camera(int width, int height, float focal)
    : width(width),
      height(height),
      focal(focal),
      drag_state_(std::make_unique<DragState>()) {
    center = {-0.35f, 0.5, 1.35f};
    v_forward = {0.7071068f, 0.0f, -0.7071068f};
    v_world_down = {0.0f, 0.0f, -1.0f};
    v_origin = {0.5f, 0.5f, 0.5f};
    _update();
}

Camera::~Camera() {
#ifdef VOLREND_CUDA
    if (device.transform != nullptr) {
        cuda(Free(device.transform));
    }
#endif
}

void Camera::_update(bool copy_cuda) {
    v_right = glm::normalize(glm::cross(v_world_down, v_forward));
    v_down = glm::cross(v_forward, v_right);
    transform[0] = v_right;
    transform[1] = v_down;
    transform[2] = v_forward;
    transform[3] = center;

#ifdef VOLREND_CUDA
    if (copy_cuda) {
        if (device.transform == nullptr) {
            cuda(Malloc((void**)&device.transform, 12 * sizeof(transform[0])));
        }
        cuda(MemcpyAsync(device.transform, glm::value_ptr(transform),
                         12 * sizeof(transform[0]), cudaMemcpyHostToDevice));
    }
#endif
}

void Camera::begin_drag(float x, float y, bool is_pan, bool about_origin) {
    drag_state_->is_dragging = true;
    drag_state_->drag_start = glm::vec2(x, y);
    drag_state_->drag_start_forward = v_forward;
    drag_state_->drag_start_right = v_right;
    drag_state_->drag_start_down = v_down;
    drag_state_->drag_start_center = center;
    drag_state_->is_panning = is_pan;
    drag_state_->about_origin = about_origin;
}
void Camera::drag_update(float x, float y) {
    if (!drag_state_->is_dragging) return;
    glm::vec2 drag_curr(x, y);
    glm::vec2 delta = drag_curr - drag_state_->drag_start;
    delta *= -2.f / std::max(width, height);
    if (drag_state_->is_panning) {
        center = drag_state_->drag_start_center +
                 delta.x * drag_state_->drag_start_right +
                 delta.y * drag_state_->drag_start_down;
    } else {
        if (drag_state_->about_origin) delta *= -1.f;
        glm::mat4 m(1.0f);
        m = glm::rotate(m, delta.x, v_world_down);
        m = glm::rotate(m, -delta.y, drag_state_->drag_start_right);

        glm::vec3 v_forward_new =
            m * glm::vec4(drag_state_->drag_start_forward, 1.f);

        float dot = glm::dot(glm::cross(v_world_down, v_forward_new),
                             drag_state_->drag_start_right);
        // Prevent flip over pole
        if (dot < 0.f) return;
        v_forward = glm::normalize(v_forward_new);

        if (drag_state_->about_origin) {
            center = glm::vec3(m * glm::vec4(drag_state_->drag_start_center -
                                                 v_origin,
                                             1.f)) +
                     v_origin;
        }
        _update(false);
    }
}
void Camera::end_drag() { drag_state_->is_dragging = false; }
void Camera::move(const glm::vec3& xyz) {
    center += xyz;
    if (drag_state_->is_dragging) {
        drag_state_->drag_start_center += xyz;
    }
}

void Camera::set_ndc(float ndc_focal, float ndc_width, float ndc_height) {
    focal = 1800.f;
    center = {0.f, 0.f, 0.0f};
    v_forward = {0.0f, 0.0f, -1.0f};
    v_world_down = {0.0f, -1.0f, 0.0f};
    v_origin = {0.0f, 0.0f, -2.0f};
    _update();
}

}  // namespace volrend
