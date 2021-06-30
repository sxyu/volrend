#include "volrend/camera.hpp"
#include <cmath>

#include <glm/gtx/string_cast.hpp>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

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
    glm::vec3 drag_start_back, drag_start_right, drag_start_up;
    glm::vec3 drag_start_center, drag_start_origin;
};

Camera::Camera(int width, int height, float fx, float fy)
    : width(width),
      height(height),
      fx(fx < 0.f ? CAMERA_DEFAULT_FOCAL_LENGTH : fx),
      fy(fy < 0.f ? this->fx : fy),
      drag_state_(std::make_unique<DragState>()) {
    center = {-3.55f, 0.0, 3.55f};
    v_back = {-0.7071068f, 0.0f, 0.7071068f};
    v_world_up = {0.0f, 0.0f, 1.0f};
    origin = {0.0f, 0.0f, 0.0f};
    _update();
}

Camera::~Camera() {
#ifdef VOLREND_CUDA
    if (device.transform != nullptr) {
        cuda(Free(device.transform));
    }
#endif
}

void Camera::_update(bool transform_from_vecs, bool copy_cuda) {
    if (transform_from_vecs) {
        v_back = glm::normalize(v_back);
        v_right = glm::normalize(glm::cross(v_world_up, v_back));
        v_up = glm::cross(v_back, v_right);
        transform[0] = v_right;
        transform[1] = v_up;
        transform[2] = v_back;
        transform[3] = center;
    }

    const float CLIP_NEAR = 1e-3;
    // clang-format off
    K = glm::mat4x4(fx / (0.5f * width), 0, 0, 0,
                  0, -fy / (0.5f * height), 0, 0,
                  0, 0, -1.f, -1,
                  0, 0, -2 * CLIP_NEAR, 0);
    // clang-format on
    w2c = glm::affineInverse(glm::mat4x4(transform));

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
    drag_state_->drag_start_back = v_back;
    drag_state_->drag_start_right = v_right;
    drag_state_->drag_start_up = v_up;
    drag_state_->drag_start_center = center;
    drag_state_->drag_start_origin = origin;
    drag_state_->is_panning = is_pan;
    drag_state_->about_origin = about_origin;
}
void Camera::drag_update(float x, float y) {
    if (!drag_state_->is_dragging) return;
    glm::vec2 drag_curr(x, y);
    glm::vec2 delta = drag_curr - drag_state_->drag_start;
    delta *= -2.f * movement_speed / std::max(width, height);
    if (drag_state_->is_panning) {
        center = drag_state_->drag_start_center +
                 delta.x * drag_state_->drag_start_right -
                 delta.y * drag_state_->drag_start_up;
        if (drag_state_->about_origin) {
            origin = drag_state_->drag_start_origin +
                     delta.x * drag_state_->drag_start_right -
                     delta.y * drag_state_->drag_start_up;
        }
    } else {
        if (drag_state_->about_origin) delta *= -1.f;
        glm::mat4 m(1.0f), m_tmp(1.0f);

        m_tmp = glm::rotate(m_tmp, -delta.y, drag_state_->drag_start_right);
        glm::vec3 v_back_tmp =
            m_tmp * glm::vec4(drag_state_->drag_start_back, 1.f);
        float dot = glm::dot(glm::cross(v_world_up, v_back_tmp),
                             drag_state_->drag_start_right);
        // Prevent flip over pole
        if (dot < 0.f) return;

        m = glm::rotate(m, fmodf(-delta.x, 2.f * M_PI), v_world_up);
        m = glm::rotate(m, -delta.y, drag_state_->drag_start_right);

        glm::vec3 v_back_new = m * glm::vec4(drag_state_->drag_start_back, 1.f);

        v_back = glm::normalize(v_back_new);

        if (drag_state_->about_origin) {
            center =
                glm::vec3(m * glm::vec4(drag_state_->drag_start_center - origin,
                                        1.f)) +
                origin;
        }
        _update(true, false);
    }
}
bool Camera::is_dragging() const { return drag_state_->is_dragging; }
void Camera::end_drag() { drag_state_->is_dragging = false; }
void Camera::move(const glm::vec3& xyz) {
    center += xyz * movement_speed;
    if (drag_state_->is_dragging) {
        drag_state_->drag_start_center += xyz * movement_speed;
    }
}

}  // namespace volrend
