#pragma once

#include <array>
#include "volrend/common.hpp"
#include "glm/mat4x3.hpp"
#include "glm/vec3.hpp"
#include "glm/vec2.hpp"

namespace volrend {
static const float CAMERA_DEFAULT_FOCAL_LENGTH = 600.f;

struct Camera {
    Camera(int width = 256, int height = 256,
           float focal = CAMERA_DEFAULT_FOCAL_LENGTH);
    ~Camera();

    /** Camera motion helpers **/
    // Drag
    void begin_drag(float x, float y, bool is_pan, bool about_origin);
    void drag_update(float x, float y);
    void end_drag();

    /** Camera params **/
    /** Camera pose model */
    glm::vec3 v_forward, v_world_down, center;
    glm::vec3 v_down, v_right;

    // 4x3 affine transform used for actual rendering
    glm::mat4x3 transform;

    // Image size
    int width, height;

    // Focal length
    float focal;

    // CUDA memory used in kernel
    struct {
        float* transform = nullptr;
    } device;

    // Update the transform after modifying v_right/v_forward/center
    // (internal)
    void _update(bool copy_cuda = true);

   private:
    void free_cuda();
    bool is_dragging_ = false;
    // Pan instead of rotate
    bool is_panning_ = false;
    // Rotate about (0.5, 0.5, 0.5)
    bool about_origin_ = false;
    glm::vec2 drag_start_;
    // Save state when drag started
    glm::vec3 drag_start_forward_;
    glm::vec3 drag_start_right_;
    glm::vec3 drag_start_down_;
    glm::vec3 drag_start_center_;
};

}  // namespace volrend
