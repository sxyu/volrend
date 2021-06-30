#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include "volrend/internal/opts.hpp"
#include "volrend/internal/imwrite.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

#include "ImGuizmo.h"

#ifndef __EMSCRIPTEN__
#include "imfilebrowser.h"
#endif

#ifdef VOLREND_CUDA
#include "volrend/cuda/common.cuh"
#endif

namespace volrend {

namespace {

void local_sph(const glm::vec3& vec, const glm::vec3& ax, const glm::vec3& ay,
               const glm::vec3& az, float& u, float& v) {
    float x = glm::dot(vec, ax);
    float y = glm::dot(vec, ay);
    float z = glm::dot(vec, az);

    u = atan2(y, x);
    v = std::asin(z);
}
glm::vec3 local_unsph(float u, float v, const glm::vec3& ax,
                      const glm::vec3& ay, const glm::vec3& az) {
    float x = std::cos(v) * std::cos(u);
    float y = std::cos(v) * std::sin(u);
    float z = std::sin(v);

    return x * ax + y * ay + z * az;
}

template <class scalar_t>
scalar_t lerp(scalar_t a, scalar_t b, float q) {
    return (1 - q) * a + q * b;
}

glm::vec3 sphc_interp(const glm::vec3& vec_start, const glm::vec3& vec_end,
                      float q, const glm::vec3& ax, const glm::vec3& ay,
                      const glm::vec3& az, int loops = 0) {
    float d_start = glm::length(vec_start);
    float d_end = glm::length(vec_end);
    glm::vec3 vec_start_unit = vec_start / d_start;
    glm::vec3 vec_end_unit = vec_end / d_end;

    if (d_start == 0.f && d_end == 0.f) {
        vec_start_unit = vec_end_unit = az;
    } else if (d_start == 0.f && d_end != 0.f) {
        vec_start_unit = vec_end_unit;
    } else if (d_start != 0.f && d_end == 0.f) {
        vec_end_unit = vec_start_unit;
    }

    float u_start, v_start, u_end, v_end;
    local_sph(vec_start_unit, ax, ay, az, u_start, v_start);
    local_sph(vec_end_unit, ax, ay, az, u_end, v_end);

    auto test_v = local_unsph(u_start, v_start, ax, ay, az);
    if (std::abs(u_start - u_end) > M_PI) {
        if (u_end > u_start)
            u_end -= 2 * M_PI;
        else
            u_start -= 2 * M_PI;
    }
    u_end += loops * 2 * M_PI;

    float u_curr = lerp(u_start, u_end, q);
    float v_curr = lerp(v_start, v_end, q);
    float d_curr = lerp(d_start, d_end, q);
    return local_unsph(u_curr, v_curr, ax, ay, az) * d_curr;
}

void save_screenshot(int width, int height, const std::string& path) {
    std::vector<unsigned char> windowPixels(4 * width * height);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                 &windowPixels[0]);

    std::vector<unsigned char> flippedPixels(4 * width * height);
    for (int row = 0; row < height; ++row)
        memcpy(&flippedPixels[row * width * 4],
               &windowPixels[(height - row - 1) * width * 4], 4 * width);

    if (internal::write_png_file(path, flippedPixels.data(), width, height)) {
        printf("Wrote %s", path.c_str());
    } else {
        printf("Failed to save screenshot\n");
    }
}

struct MeshState {
    MeshState() {}
    explicit MeshState(const Mesh& mesh)
        : rotation(mesh.rotation),
          translation(mesh.translation),
          scale(mesh.scale),
          unlit(mesh.unlit) {}

    void set_mesh(Mesh& mesh) const {
        mesh.rotation = rotation;
        mesh.translation = translation;
        mesh.scale = scale;
        mesh.unlit = unlit;
        mesh.visible = true;
    }

    // Model transform, rotatin is axis-angle
    glm::vec3 rotation, translation;
    float scale = 1.f;

    bool unlit = false;
};

// Represents 1 animation keyfrae
struct AnimKF {
    AnimKF() {}
    explicit AnimKF(const VolumeRenderer& rend) { from_renderer(rend); }
    void from_renderer(const VolumeRenderer& rend) {
        center = rend.camera.center;
        origin = rend.camera.origin;
        v_back = glm::normalize(rend.camera.v_back);
        fx = rend.camera.fx;
        fy = rend.camera.fy;
        opt = rend.options;

        mesh_state.clear();
        for (const Mesh& mesh : rend.meshes) {
            if (!mesh.visible) continue;
            MeshState state{mesh};
            mesh_state[mesh.name] = std::move(state);
        }
    }
    void to_renderer(VolumeRenderer& rend) const {
        rend.camera.center = center;
        rend.camera.origin = origin;
        rend.camera.v_back = glm::normalize(v_back);
        rend.camera.fx = fx;
        rend.camera.fy = fy;
        rend.options = opt;
        for (volrend::Mesh& mesh : rend.meshes) {
            if (!mesh_state.count(mesh.name)) {
                mesh.visible = false;
                continue;
            }
            mesh_state.at(mesh.name).set_mesh(mesh);
        }
    }
    // * State
    glm::vec3 center, origin, v_back;
    float fx, fy;
    RenderOptions opt;
    std::map<std::string, MeshState> mesh_state;

    // * Configuration
    // Animation duration
    float t_max = 1.f;
    // Whether to use spherical interpolation
    bool spherical_interp = true;
    // Extra CCW loops about world_up to make, only if spherical_interp
    int loops = 0;
};

std::vector<AnimKF> keyframes;
struct AnimState {
    // General config
    float fps = 30.f;
    std::string output_folder = "ani_out/";

    // * Do not modify these
    // If true, we're in animation mode and camera is on autopilot
    bool animating = false;
    // If true, we are animating in real-time for the user
    // else, we're writing the video images
    bool previewing = true;

    // Key frame ID
    size_t kf_idx = -1;
    // Frame ID
    size_t f_idx = 0;

    void anim_from_start(bool previewing = true) {
        if (keyframes.size() < 2) {
            fprintf(stderr, "WARNING: cannot animate with < 2 keyframes\n");
            return;
        }
        anim_once(keyframes[0], keyframes[1], previewing, -1.f, 0);
        if (!previewing) {
            std::filesystem::create_directories(output_folder);
        }
        f_idx = 0;
    }

    void anim_once(const AnimKF& start, const AnimKF& end,
                   bool previewing = true, float t_max = -1.f,
                   int kf_idx = -1) {
        this->start = start;
        this->end = end;
        curr = start;
        t = 0.f;
        this->t_max = t_max > 0.f ? t_max : end.t_max;
        this->previewing = previewing;
        if (previewing) {
            _last_tp = std::chrono::high_resolution_clock::now();
        }
        this->kf_idx = kf_idx;
        animating = true;
    }

    void update(VolumeRenderer& rend) {
        if (previewing) {
            std::chrono::high_resolution_clock::time_point now_tp =
                std::chrono::high_resolution_clock::now();
            double ms =
                std::chrono::duration<double, std::milli>(now_tp - _last_tp)
                    .count();
            t += ms / 1000.f;
            _last_tp = now_tp;
        } else {
            ++f_idx;
            t += 1.f / fps;
        }
        float q = std::min(t / t_max, 1.f);
        curr.origin = lerp(start.origin, end.origin, q);

        glm::vec3 az = rend.camera.v_world_up;
        glm::vec3 ax, ay;
        if (end.spherical_interp || end.mesh_state.size()) {
            ax = glm::normalize(rend.camera.v_back -
                                glm::dot(rend.camera.v_back, az) * az);
            ay = glm::normalize(glm::cross(az, ax));
        }
        if (end.spherical_interp) {
            int loops = ~kf_idx ? end.loops : 0;

            glm::vec3 start_vec = start.center - start.origin;
            glm::vec3 end_vec = end.center - end.origin;
            glm::vec3 curr_vec =
                sphc_interp(start_vec, end_vec, q, ax, ay, az, loops);
            curr.center = curr.origin + curr_vec;

            curr.v_back =
                sphc_interp(start.v_back, end.v_back, q, ax, ay, az, loops);
        } else {
            curr.center = lerp(start.center, end.center, q);
            curr.v_back = lerp(start.v_back, end.v_back, q);
        }
        curr.fx = lerp(start.fx, end.fx, q);
        curr.fy = lerp(start.fy, end.fy, q);
        curr.opt = end.opt;
        curr.opt.background_brightness = lerp(start.opt.background_brightness,
                                              end.opt.background_brightness, q);
        curr.opt.step_size = lerp(start.opt.step_size, end.opt.step_size, q);
        curr.opt.stop_thresh =
            lerp(start.opt.stop_thresh, end.opt.stop_thresh, q);
        curr.opt.sigma_thresh =
            lerp(start.opt.sigma_thresh, end.opt.sigma_thresh, q);
        if (start.opt.enable_probe) {
            for (int i = 0; i < 3; ++i) {
                curr.opt.probe[i] =
                    lerp(start.opt.probe[i], end.opt.probe[i], q);
            }
        }
        if (end.opt.show_grid) {
            int start_depth =
                start.opt.show_grid ? start.opt.grid_max_depth : 0;
            if (start_depth != end.opt.grid_max_depth) {
                curr.opt.grid_max_depth =
                    lerp(start_depth, end.opt.grid_max_depth, q);
            }
        }
        for (int i = 0; i < 6; ++i) {
            curr.opt.render_bbox[i] =
                lerp(start.opt.render_bbox[i], end.opt.render_bbox[i], q);
        }

        glm::vec3 start_rot_dirs(start.opt.rot_dirs[0], start.opt.rot_dirs[1],
                                 start.opt.rot_dirs[2]);
        glm::vec3 end_rot_dirs(end.opt.rot_dirs[0], end.opt.rot_dirs[1],
                               end.opt.rot_dirs[2]);
        if (start_rot_dirs != end_rot_dirs) {
            glm::vec3 curr_rot_dirs =
                sphc_interp(start_rot_dirs, end_rot_dirs, q, ax, ay, az);
            for (int i = 0; i < 3; ++i) {
                curr.opt.rot_dirs[i] = curr_rot_dirs[i];
            }
        }

        curr.mesh_state = end.mesh_state;
        for (const std::pair<std::string, MeshState>& meshp : end.mesh_state) {
            std::string name = meshp.first;
            if (start.mesh_state.count(name)) {
                const MeshState& start_state = start.mesh_state[name];
                const MeshState& end_state = meshp.second;
                MeshState& state = curr.mesh_state[name];
                state.rotation = sphc_interp(start_state.rotation,
                                             end_state.rotation, q, ax, ay, az);
                state.translation =
                    lerp(start_state.translation, end_state.translation, q);
                state.scale = lerp(start_state.scale, end_state.scale, q);
            }
        }

        curr.to_renderer(rend);

        if (t >= t_max) {
            if (kf_idx >= keyframes.size() - 2) {
                animating = false;
            } else {
                ++kf_idx;
                anim_once(keyframes[kf_idx], keyframes[kf_idx + 1], previewing,
                          -1.f, kf_idx);
            }
        }
    }

   private:
    AnimKF start, end, curr;
    float t_max = 1.f;
    float t = 0.0f;
    std::chrono::high_resolution_clock::time_point _last_tp;
} anim;

#define GET_RENDERER(window) \
    (*((VolumeRenderer*)glfwGetWindowUserPointer(window)))

int gizmo_mesh_op = ImGuizmo::TRANSLATE;
int gizmo_mesh_space = ImGuizmo::LOCAL;

void draw_imgui(VolumeRenderer& rend, N3Tree& tree) {
    auto& cam = rend.camera;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // BEGIN gizmo handling
    // clang-format off
    static glm::mat4 camera_persp_prj(1.f, 0.f, 0.f, 0.f,
                                         0.f, 1.f, 0.f, 0.f,
                                         0.f, 0.f, -1.f, -1.f,
                                         0.f, 0.f, -0.001f, 0.f);
    // clang-format on
    ImGuiIO& io = ImGui::GetIO();

    camera_persp_prj[0][0] = cam.fx / cam.width * 2.0;
    camera_persp_prj[1][1] = cam.fy / cam.height * 2.0;
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetGizmoSizeClipSpace(0.05f);

    ImGuizmo::BeginFrame();

    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    glm::mat4 w2c = glm::affineInverse(glm::mat4(cam.transform));
    // END gizmo handling

    ImGui::SetNextWindowSize(ImVec2(400.f, 480.f), ImGuiCond_Once);

    // Begin animator window
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::Begin("animator");

    static ImGui::FileBrowser select_output_folder_dialog(
        ImGuiFileBrowserFlags_SelectDirectory |
        ImGuiFileBrowserFlags_CreateNewDir);
    if (select_output_folder_dialog.GetTitle().empty()) {
        select_output_folder_dialog.SetTitle("Set animation output folder");
    }

    if (ImGui::Button("Preview")) {
        anim.anim_from_start(/* previewing */ true);
        if (keyframes.size()) {
            keyframes[0].to_renderer(rend);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Render")) {
        anim.anim_from_start(/* previewing */ false);
        if (keyframes.size()) {
            keyframes[0].to_renderer(rend);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        anim.animating = false;
    }
    ImGui::Text("Output dir: %s", anim.output_folder.c_str());
    if (ImGui::Button("Change output dir")) {
        select_output_folder_dialog.Open();
    }

    ImGui::SameLine();
    ImGui::PushItemWidth(60.f);
    ImGui::InputFloat("anim fps", &anim.fps);
    ImGui::PopItemWidth();

    select_output_folder_dialog.Display();
    if (select_output_folder_dialog.HasSelected()) {
        std::string path = select_output_folder_dialog.GetSelected().string();
        if (!path.empty() && path.back() != '/') path.push_back('/');
        printf("Animation output folder set to %s\n", path.c_str());
        anim.output_folder = path;
        select_output_folder_dialog.ClearSelected();
    }

    static bool lock_fx_fy = true;
    if (!anim.animating || anim.previewing) {
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Keyframes")) {
            std::vector<int> kf_to_del;
            for (size_t i = 0; i < keyframes.size(); ++i) {
                std::string si = std::to_string(i);
                AnimKF& kf = keyframes[i];
                ImGui::Text("%s", si.c_str());
                ImGui::SameLine();
                if (ImGui::Button(("goto##kf" + si).c_str())) {
                    anim.anim_once(AnimKF{rend}, kf, true, 0.3f);
                }
                ImGui::SameLine();
                if (ImGui::Button(("set##kf" + si).c_str())) {
                    kf.from_renderer(rend);
                }
                ImGui::SameLine();
                if (i > 0) {
                    ImGui::TextUnformatted("dur=");
                    ImGui::SameLine();
                    ImGui::PushItemWidth(40.f);
                    ImGui::InputFloat(("s##kf" + si).c_str(), &kf.t_max);
                    ImGui::PopItemWidth();
                    ImGui::SameLine();
                    ImGui::Checkbox(("sph##kf" + si).c_str(),
                                    &kf.spherical_interp);
                    if (kf.spherical_interp) {
                        ImGui::SameLine();
                        ImGui::PushItemWidth(30.f);
                        ImGui::InputInt(("loop##kf" + si).c_str(), &kf.loops,
                                        0);
                        ImGui::PopItemWidth();
                    }
                    ImGui::SameLine();
                }
                if (ImGui::Button(("x##kf" + si).c_str())) {
                    kf_to_del.push_back(i);
                }
                ImGui::SameLine();
                if (ImGui::TreeNode(("data##kf" + si).c_str())) {
                    if (ImGui::TreeNode("Camera")) {
                        ImGui::InputFloat3("center", glm::value_ptr(kf.center));
                        ImGui::InputFloat3("origin", glm::value_ptr(kf.origin));
                        ImGui::Spacing();
                        ImGui::InputFloat3("v_back_",
                                           glm::value_ptr(kf.v_back));
                        if (ImGui::Button("normalize v_back")) {
                            kf.v_back = glm::normalize(kf.v_back);
                        }
                        ImGui::Checkbox("fx=fy", &lock_fx_fy);
                        if (lock_fx_fy) {
                            if (ImGui::InputFloat("focal", &kf.fx)) {
                                kf.fy = kf.fx;
                            }
                        } else {
                            ImGui::InputFloat("fx", &kf.fx);
                            ImGui::InputFloat("fy", &kf.fy);
                        }
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNode("Render")) {
                        static float inv_step_size = 1.0f / kf.opt.step_size;
                        if (ImGui::SliderFloat("1/eps", &inv_step_size, 128.f,
                                               20000.f)) {
                            kf.opt.step_size = 1.f / inv_step_size;
                        }
                        ImGui::SliderFloat("sigma_thresh", &kf.opt.sigma_thresh,
                                           0.f, 100.0f);
                        ImGui::SliderFloat("stop_thresh", &kf.opt.stop_thresh,
                                           0.001f, 0.4f);
                        ImGui::SliderFloat("bg_brightness",
                                           &kf.opt.background_brightness, 0.f,
                                           1.0f);
                        ImGui::PushItemWidth(230);
                        ImGui::SliderFloat3("bb_min", kf.opt.render_bbox, 0.0,
                                            1.0);
                        ImGui::SliderFloat3("bb_max", kf.opt.render_bbox + 3,
                                            0.0, 1.0);
                        ImGui::SliderInt2(
                            "decomp", kf.opt.basis_minmax, 0,
                            std::max(tree.data_format.basis_dim - 1, 0));
                        ImGui::SliderFloat3("viewdir shift", kf.opt.rot_dirs,
                                            -M_PI / 4, M_PI / 4);
                        ImGui::PopItemWidth();
                        if (ImGui::Button("Reset Viewdir Shift")) {
                            for (int i = 0; i < 3; ++i)
                                kf.opt.rot_dirs[i] = 0.f;
                        }

                        ImGui::Checkbox("Show Grid", &kf.opt.show_grid);
#ifdef VOLREND_CUDA
                        ImGui::SameLine();
                        ImGui::Checkbox("Render Depth", &kf.opt.render_depth);
#endif
                        if (kf.opt.show_grid) {
                            ImGui::SliderInt("grid max depth",
                                             &kf.opt.grid_max_depth, 0, 7);
                        }

                        ImGui::TreePop();
                    }
                    if (kf.mesh_state.size()) {
                        if (ImGui::TreeNode("Meshes")) {
                            if (ImGui::Button("Rotate all")) {
                                for (auto& p : kf.mesh_state) {
                                    p.second.rotation[2] += 2 * M_PI;
                                }
                            }
                            for (auto& p : kf.mesh_state) {
                                auto& mesh = p.second;
                                if (ImGui::TreeNode(p.first.c_str())) {
                                    ImGui::PushItemWidth(230);
                                    ImGui::InputFloat3(
                                        "trans",
                                        glm::value_ptr(mesh.translation));
                                    ImGui::InputFloat3(
                                        "rot", glm::value_ptr(mesh.rotation));
                                    ImGui::InputFloat("scale", &mesh.scale);
                                    ImGui::PopItemWidth();

                                    ImGui::Checkbox("unlit", &mesh.unlit);

                                    ImGui::TreePop();
                                }
                            }
                            ImGui::TreePop();
                        }
                    }
                    ImGui::TreePop();
                }
            }
            if (kf_to_del.size()) {
                size_t j = 0, ii = 0;
                for (size_t i = 0; i < keyframes.size(); ++i) {
                    if (kf_to_del[j] == i) {
                        ++j;
                        continue;
                    }
                    keyframes[ii] = keyframes[i];
                    ++ii;
                }
                keyframes.resize(keyframes.size() - kf_to_del.size());
            }
            if (ImGui::Button("add KF at curr")) {
                keyframes.emplace_back(rend);
            }
        }
    }
    ImGui::End();

    if (!anim.animating || anim.previewing) {
        static char title[128] = {0};
        if (title[0] == 0) {
            sprintf(title, "volrend backend: %s", rend.get_backend());
        }

        ImGui::SetNextWindowPos(ImVec2(10, 500), ImGuiCond_Once);
        // Begin standard control window
        ImGui::Begin(title);
#ifndef __EMSCRIPTEN__
        static ImGui::FileBrowser open_obj_mesh_dialog(
            ImGuiFileBrowserFlags_MultipleSelection);
        if (open_obj_mesh_dialog.GetTitle().empty()) {
            open_obj_mesh_dialog.SetTypeFilters({".obj"});
            open_obj_mesh_dialog.SetTitle("Load basic triangle OBJ");
        }
        static ImGui::FileBrowser open_tree_dialog,
            save_screenshot_dialog(ImGuiFileBrowserFlags_EnterNewFilename);
        if (open_tree_dialog.GetTitle().empty()) {
            open_tree_dialog.SetTypeFilters({".npz"});
            open_tree_dialog.SetTitle("Load N3Tree npz from svox");
        }
        if (save_screenshot_dialog.GetTitle().empty()) {
            save_screenshot_dialog.SetTypeFilters({".png"});
            save_screenshot_dialog.SetTitle("Save screenshot (png)");
        }

        if (ImGui::Button("Open Tree")) {
            open_tree_dialog.Open();
        }
        ImGui::SameLine();
        if (ImGui::Button("Save Screenshot")) {
            save_screenshot_dialog.Open();
        }

        open_tree_dialog.Display();
        if (open_tree_dialog.HasSelected()) {
            // Load octree
            std::string path = open_tree_dialog.GetSelected().string();
            printf("Load N3Tree npz: %s\n", path.c_str());
            tree.open(path);
            rend.set(tree);
            open_tree_dialog.ClearSelected();
        }

        save_screenshot_dialog.Display();
        if (save_screenshot_dialog.HasSelected()) {
            // Save screenshot
            std::string path = save_screenshot_dialog.GetSelected().string();
            if (path.size() < 4 ||
                path.compare(path.size() - 4, 4, ".png", 0, 4) != 0) {
                path.append(".png");
            }
            save_screenshot_dialog.ClearSelected();
            save_screenshot(cam.width, cam.height, path);
        }
#endif

        ImGui::SetNextTreeNodeOpen(false, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Camera")) {
            // Update vectors indirectly since we need to normalize on
            // change (press update button) and it would be too confusing to
            // keep normalizing
            static glm::vec3 world_up_tmp = rend.camera.v_world_up;
            static glm::vec3 world_down_prev = rend.camera.v_world_up;
            static glm::vec3 back_tmp = rend.camera.v_back;
            static glm::vec3 forward_prev = rend.camera.v_back;
            if (cam.v_world_up != world_down_prev)
                world_up_tmp = world_down_prev = cam.v_world_up;
            if (cam.v_back != forward_prev)
                back_tmp = forward_prev = cam.v_back;

            ImGui::InputFloat3("center", glm::value_ptr(cam.center));
            ImGui::InputFloat3("origin", glm::value_ptr(cam.origin));
            ImGui::Checkbox("fx=fy", &lock_fx_fy);
            if (lock_fx_fy) {
                if (ImGui::InputFloat("focal", &cam.fx)) {
                    cam.fy = cam.fx;
                }
            } else {
                ImGui::InputFloat("fx", &cam.fx);
                ImGui::InputFloat("fy", &cam.fy);
            }
            if (ImGui::TreeNode("Directions")) {
                ImGui::InputFloat3("world_up", glm::value_ptr(world_up_tmp));
                ImGui::InputFloat3("back", glm::value_ptr(back_tmp));
                if (ImGui::Button("normalize & update dirs")) {
                    cam.v_world_up = glm::normalize(world_up_tmp);
                    cam.v_back = glm::normalize(back_tmp);
                }
                ImGui::TreePop();
            }
        }  // End camera node

        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Render")) {
            static float inv_step_size = 1.0f / rend.options.step_size;
            if (ImGui::SliderFloat("1/eps", &inv_step_size, 128.f, 20000.f)) {
                rend.options.step_size = 1.f / inv_step_size;
            }
            ImGui::SliderFloat("sigma_thresh", &rend.options.sigma_thresh, 0.f,
                               100.0f);
            ImGui::SliderFloat("stop_thresh", &rend.options.stop_thresh, 0.001f,
                               0.4f);
            ImGui::SliderFloat("bg_brightness",
                               &rend.options.background_brightness, 0.f, 1.0f);

        }  // End render node
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Visualization")) {
            ImGui::PushItemWidth(230);
            ImGui::SliderFloat3("bb_min", rend.options.render_bbox, 0.0, 1.0);
            ImGui::SliderFloat3("bb_max", rend.options.render_bbox + 3, 0.0,
                                1.0);
            ImGui::SliderInt2("decomp", rend.options.basis_minmax, 0,
                              std::max(tree.data_format.basis_dim - 1, 0));
            ImGui::SliderFloat3("viewdir shift", rend.options.rot_dirs,
                                -M_PI / 4, M_PI / 4);
            ImGui::PopItemWidth();
            if (ImGui::Button("Reset Viewdir Shift")) {
                for (int i = 0; i < 3; ++i) rend.options.rot_dirs[i] = 0.f;
            }

            ImGui::Checkbox("Show Grid", &rend.options.show_grid);
#ifdef VOLREND_CUDA
            ImGui::SameLine();
            ImGui::Checkbox("Render Depth", &rend.options.render_depth);
#endif
            if (rend.options.show_grid) {
                ImGui::SliderInt("grid max depth", &rend.options.grid_max_depth,
                                 0, 7);
            }
        }

        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Manipulation")) {
            static std::vector<glm::mat4> gizmo_mesh_trans;
            gizmo_mesh_trans.resize(rend.meshes.size());

            ImGui::TextUnformatted("gizmo op");
            ImGui::SameLine();
            ImGui::RadioButton("trans##giztrans", &gizmo_mesh_op,
                               ImGuizmo::TRANSLATE);
            ImGui::SameLine();
            ImGui::RadioButton("rot##gizrot", &gizmo_mesh_op, ImGuizmo::ROTATE);
            ImGui::SameLine();
            ImGui::RadioButton("scale##gizscale", &gizmo_mesh_op,
                               ImGuizmo::SCALE_Z);

            ImGui::TextUnformatted("gizmo space");
            ImGui::SameLine();
            ImGui::RadioButton("local##gizlocal", &gizmo_mesh_space,
                               ImGuizmo::LOCAL);
            ImGui::SameLine();
            ImGui::RadioButton("world##gizworld", &gizmo_mesh_space,
                               ImGuizmo::WORLD);

            ImGui::BeginGroup();
            std::vector<int> meshes_to_del;
            for (int i = 0; i < (int)rend.meshes.size(); ++i) {
                auto& mesh = rend.meshes[i];
                if (ImGui::TreeNode(mesh.name.c_str())) {
                    if (mesh.visible) {
                        glm::mat4& gizmo_trans = gizmo_mesh_trans[i];
                        gizmo_trans = mesh.transform_;
                        if (gizmo_mesh_op == ImGuizmo::SCALE_Z) {
                            glm::mat4 tmp(1);
                            tmp[3] = gizmo_trans[3];
                            gizmo_trans = tmp;
                        }
                        ImGuizmo::SetID(i + 1);
                        if (ImGuizmo::Manipulate(
                                glm::value_ptr(w2c),
                                glm::value_ptr(camera_persp_prj),
                                (ImGuizmo::OPERATION)gizmo_mesh_op,
                                (ImGuizmo::MODE)gizmo_mesh_space,
                                glm::value_ptr(gizmo_trans), NULL, NULL, NULL,
                                NULL)) {
                            if (gizmo_mesh_op == ImGuizmo::ROTATE) {
                                glm::quat rot_q = glm::quat_cast(
                                    glm::mat3(gizmo_trans) / mesh.scale);
                                mesh.rotation =
                                    glm::axis(rot_q) * glm::angle(rot_q);
                            } else if (gizmo_mesh_op == ImGuizmo::SCALE_Z) {
                                mesh.scale *=
                                    gizmo_trans[2][2] /
                                    mesh.transform_[2][2];  // max_scale;
                            }
                            mesh.translation = gizmo_trans[3];
                        }
                    }
                    ImGui::PushItemWidth(230);
                    ImGui::InputFloat3("trans",
                                       glm::value_ptr(mesh.translation));
                    ImGui::InputFloat3("rot", glm::value_ptr(mesh.rotation));
                    ImGui::InputFloat("scale", &mesh.scale);
                    ImGui::PopItemWidth();
                    ImGui::Checkbox("visible", &mesh.visible);
                    ImGui::SameLine();
                    ImGui::Checkbox("unlit", &mesh.unlit);
                    ImGui::SameLine();
                    if (ImGui::Button("delete")) meshes_to_del.push_back(i);

                    ImGui::TreePop();
                }
            }

            if (meshes_to_del.size()) {
                int j = 0;
                std::vector<Mesh> tmp;
                tmp.reserve(rend.meshes.size() - meshes_to_del.size());
                for (int i = 0; i < rend.meshes.size(); ++i) {
                    if (i == meshes_to_del[j]) {
                        ++j;
                        continue;
                    }
                    tmp.push_back(std::move(rend.meshes[i]));
                }
                rend.meshes.swap(tmp);
            }
            ImGui::EndGroup();
            if (ImGui::Button("Sphere##addsphere")) {
                static int sphereid = 0;
                {
                    Mesh sph = Mesh::Sphere();
                    sph.scale = 0.1f;
                    sph.translation[2] = 1.0f;
                    sph.update();
                    if (sphereid)
                        sph.name = sph.name + std::to_string(sphereid);
                    ++sphereid;
                    rend.meshes.push_back(std::move(sph));
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cube##addcube")) {
                static int cubeid = 0;
                {
                    Mesh cube = Mesh::Cube();
                    cube.scale = 0.2f;
                    cube.translation[2] = 1.0f;
                    cube.update();
                    if (cubeid) cube.name = cube.name + std::to_string(cubeid);
                    ++cubeid;
                    rend.meshes.push_back(std::move(cube));
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Latti##addlattice")) {
                static int lattid = 0;
                {
                    Mesh latt = Mesh::Lattice();
                    if (tree.N > 0) {
                        latt.scale = 1.f / std::min(std::min(tree.scale[0],
                                                             tree.scale[1]),
                                                    tree.scale[2]);
                        for (int i = 0; i < 3; ++i) {
                            latt.translation[i] =
                                -1.f / tree.scale[0] * tree.offset[0];
                        }
                    }
                    latt.update();
                    if (lattid) latt.name = latt.name + std::to_string(lattid);
                    ++lattid;
                    rend.meshes.push_back(std::move(latt));
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Load OBJ")) {
                open_obj_mesh_dialog.Open();
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear All")) {
                rend.meshes.clear();
            }

#ifdef VOLREND_CUDA
            if (tree.capacity) {
                ImGui::BeginGroup();
                ImGui::Checkbox("Enable Lumisphere Probe",
                                &rend.options.enable_probe);
                if (rend.options.enable_probe) {
                    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
                    if (ImGui::TreeNode("Probe")) {
                        static glm::mat4 probe_trans;
                        static bool show_probe_gizmo = true;
                        float* probe = rend.options.probe;
                        probe_trans = glm::translate(
                            glm::mat4(1.f),
                            glm::vec3(probe[0], probe[1], probe[2]));

                        ImGui::Checkbox("Show gizmo", &show_probe_gizmo);
                        if (show_probe_gizmo) {
                            ImGuizmo::SetID(0);
                            if (ImGuizmo::Manipulate(
                                    glm::value_ptr(w2c),
                                    glm::value_ptr(camera_persp_prj),
                                    ImGuizmo::TRANSLATE, ImGuizmo::LOCAL,
                                    glm::value_ptr(probe_trans), NULL, NULL,
                                    NULL, NULL)) {
                                for (int i = 0; i < 3; ++i)
                                    probe[i] = probe_trans[3][i];
                            }
                        }
                        ImGui::InputFloat3("probe", probe);
                        ImGui::SliderInt("probe_win_sz",
                                         &rend.options.probe_disp_size, 50,
                                         800);
                        ImGui::TreePop();
                    }
                }
                ImGui::EndGroup();
            }
#endif
        }
        open_obj_mesh_dialog.Display();
        if (open_obj_mesh_dialog.HasSelected()) {
            // Load mesh
            auto sels = open_obj_mesh_dialog.GetMultiSelected();
            for (auto& fpath : sels) {
                const std::string path = fpath.string();
                printf("Load OBJ: %s\n", path.c_str());
                Mesh tmp = Mesh::load_basic_obj(path);
                if (tmp.vert.size()) {
                    // Auto offset
                    std::ifstream ifs(path + ".offs");
                    if (ifs) {
                        ifs >> tmp.translation.x >> tmp.translation.y >>
                            tmp.translation.z;
                        if (ifs) {
                            ifs >> tmp.scale;
                        }
                    }
                    tmp.update();
                    rend.meshes.push_back(std::move(tmp));
                    puts("Load success\n");
                } else {
                    puts("Load failed\n");
                }
            }
            open_obj_mesh_dialog.ClearSelected();
        }

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void glfw_error_callback(int error, const char* description) {
    fputs(description, stderr);
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action,
                       int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        auto& rend = GET_RENDERER(window);
        auto& cam = rend.camera;
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_W:
            case GLFW_KEY_S:
            case GLFW_KEY_A:
            case GLFW_KEY_D:
            case GLFW_KEY_E:
            case GLFW_KEY_Q:
                if (!anim.animating) {
                    // Camera movement
                    float speed = 0.002f;
                    if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                    if (key == GLFW_KEY_S || key == GLFW_KEY_A ||
                        key == GLFW_KEY_E)
                        speed = -speed;
                    const auto& vec =
                        (key == GLFW_KEY_A || key == GLFW_KEY_D)   ? cam.v_right
                        : (key == GLFW_KEY_W || key == GLFW_KEY_S) ? -cam.v_back
                                                                   : -cam.v_up;
                    cam.move(vec * speed);
                }
                break;

            case GLFW_KEY_Z: {
                // Cycle gizmo op
                if (gizmo_mesh_op == ImGuizmo::TRANSLATE)
                    gizmo_mesh_op = ImGuizmo::ROTATE;
                else if (gizmo_mesh_op == ImGuizmo::ROTATE)
                    gizmo_mesh_op = ImGuizmo::SCALE_Z;
                else
                    gizmo_mesh_op = ImGuizmo::TRANSLATE;
            } break;

            case GLFW_KEY_X: {
                // Cycle gizmo space
                if (gizmo_mesh_space == ImGuizmo::LOCAL)
                    gizmo_mesh_space = ImGuizmo::WORLD;
                else
                    gizmo_mesh_space = ImGuizmo::LOCAL;
            } break;

#ifdef VOLREND_CUDA
            case GLFW_KEY_I:
            case GLFW_KEY_J:
            case GLFW_KEY_K:
            case GLFW_KEY_L:
            case GLFW_KEY_U:
            case GLFW_KEY_O:
                if (rend.options.enable_probe) {
                    // Probe movement
                    float speed = 0.002f;
                    if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                    if (key == GLFW_KEY_J || key == GLFW_KEY_K ||
                        key == GLFW_KEY_U)
                        speed = -speed;
                    int dim = (key == GLFW_KEY_J || key == GLFW_KEY_L)   ? 0
                              : (key == GLFW_KEY_I || key == GLFW_KEY_K) ? 1
                                                                         : 2;
                    rend.options.probe[dim] += speed;
                }
                break;
#endif

            case GLFW_KEY_MINUS:
                cam.fx *= 0.99f;
                cam.fy *= 0.99f;
                break;

            case GLFW_KEY_EQUAL:
                cam.fx *= 1.01f;
                cam.fy *= 1.01f;
                break;

            case GLFW_KEY_0:
                cam.fx = CAMERA_DEFAULT_FOCAL_LENGTH;
                cam.fy = CAMERA_DEFAULT_FOCAL_LENGTH;
                break;

            case GLFW_KEY_1:
                cam.v_world_up = glm::vec3(0.f, 0.f, 1.f);
                break;

            case GLFW_KEY_2:
                cam.v_world_up = glm::vec3(0.f, 0.f, -1.f);
                break;

            case GLFW_KEY_3:
                cam.v_world_up = glm::vec3(0.f, 1.f, 0.f);
                break;

            case GLFW_KEY_4:
                cam.v_world_up = glm::vec3(0.f, -1.f, 0.f);
                break;

            case GLFW_KEY_5:
                cam.v_world_up = glm::vec3(1.f, 0.f, 0.f);
                break;

            case GLFW_KEY_6:
                cam.v_world_up = glm::vec3(-1.f, 0.f, 0.f);
                break;
        }
    }
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action,
                                int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    auto& rend = GET_RENDERER(window);
    auto& cam = rend.camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (action == GLFW_PRESS && !anim.animating) {
        cam.begin_drag(
            x, y, (mods & GLFW_MOD_SHIFT) || button == GLFW_MOUSE_BUTTON_MIDDLE,
            button == GLFW_MOUSE_BUTTON_RIGHT ||
                button == GLFW_MOUSE_BUTTON_MIDDLE);
    } else if (action == GLFW_RELEASE) {
        cam.end_drag();
    }
}

void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_RENDERER(window).camera.drag_update(x, y);
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse || anim.animating) return;
    auto& cam = GET_RENDERER(window).camera;
    // Focal length adjusting was very annoying so changed it to movement in
    // z cam.focal *= (yoffset > 0.f) ? 1.01f : 0.99f;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

GLFWwindow* glfw_init(const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) std::exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_DEPTH_BITS, GL_TRUE);
#ifdef VOLREND_CUDA
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window =
        glfwCreateWindow(width, height, "volrend viewer", NULL, NULL);

    glClearDepth(1.0);
    glDepthFunc(GL_LESS);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (window == nullptr) {
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fputs("GLEW init failed\n", stderr);
        getchar();
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, false);
    char* glsl_version = NULL;
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename = nullptr;
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    GET_RENDERER(window).resize(width, height);
}

}  // namespace
}  // namespace volrend

int main(int argc, char* argv[]) {
    using namespace volrend;

    cxxopts::Options cxxoptions(
        "volrend_anim",
        "PlenOctree animation engine (c) PlenOctree authors 2021");

    internal::add_common_opts(cxxoptions);
    // clang-format off
    cxxoptions.add_options()
        ("center", "camera center position (world); ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value(
                                                        "-3.5,0,3.5"))
        ("back", "camera's back direction unit vector (world) for orientation; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("-0.7071068,0,0.7071068"))
        ("origin", "origin for right click rotation controls; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,0"))
        ("world_up", "world up direction for rotating controls e.g. "
                     "0,0,1=blender; ignored for NDC",
                cxxopts::value<std::vector<float>>()->default_value("0,0,1"))
        ("grid", "show grid with given max resolution (4 is reasonable)", cxxopts::value<int>())
        ("probe", "enable lumisphere_probe and place it at given x,y,z",
                   cxxopts::value<std::vector<float>>())
        ;
    // clang-format on

    cxxoptions.positional_help("npz_file");

    cxxopts::ParseResult args = internal::parse_options(cxxoptions, argc, argv);

#ifdef VOLREND_CUDA
    const int device_id = args["gpu"].as<int>();
    if (~device_id) {
        cuda(SetDevice(device_id));
    }
#endif

    N3Tree tree;
    bool init_loaded = false;
    if (args.count("file")) {
        init_loaded = true;
        tree.open(args["file"].as<std::string>());
    }
    int width = args["width"].as<int>(), height = args["height"].as<int>();
    float fx = args["fx"].as<float>();
    float fy = args["fy"].as<float>();

    GLFWwindow* window = glfw_init(width, height);
    glfwSetWindowTitle(window, "PlenOctree animator");

    {
        VolumeRenderer rend;
        if (fx > 0.f) {
            rend.camera.fx = fx;
        }

        rend.options = internal::render_options_from_args(args);
        if (init_loaded && tree.use_ndc) {
            // Special inital coordinates for NDC
            // (pick average camera)
            rend.camera.center = glm::vec3(0);
            rend.camera.origin = glm::vec3(0, 0, -3);
            rend.camera.v_back = glm::vec3(0, 0, 1);
            rend.camera.v_world_up = glm::vec3(0, 1, 0);
            if (fx <= 0) {
                rend.camera.fx = rend.camera.fy = tree.ndc_focal * 0.25f;
            }
            rend.camera.movement_speed = 0.1f;
        } else {
            auto cen = args["center"].as<std::vector<float>>();
            rend.camera.center = glm::vec3(cen[0], cen[1], cen[2]);
            auto origin = args["origin"].as<std::vector<float>>();
            rend.camera.origin = glm::vec3(origin[0], origin[1], origin[2]);
            auto world_up = args["world_up"].as<std::vector<float>>();
            rend.camera.v_world_up =
                glm::vec3(world_up[0], world_up[1], world_up[2]);
            auto back = args["back"].as<std::vector<float>>();
            rend.camera.v_back = glm::vec3(back[0], back[1], back[2]);
        }
        if (fy <= 0.f) {
            rend.camera.fy = rend.camera.fx;
        }

        {
            std::string drawlist_load_path = args["draw"].as<std::string>();
            if (drawlist_load_path.size()) {
                rend.meshes = Mesh::open_drawlist(drawlist_load_path);
            }
        }

        glfwGetFramebufferSize(window, &width, &height);
        rend.set(tree);
        rend.resize(width, height);

        // Set user pointer and callbacks
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetScrollCallback(window, glfw_scroll_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        while (!glfwWindowShouldClose(window)) {
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_PROGRAM_POINT_SIZE);
            glPointSize(4.f);

            rend.render();
            if (anim.animating) {
                if (!anim.previewing) {
                    std::stringstream sst;
                    sst << anim.output_folder << std::setfill('0')
                        << std::setw(6) << anim.f_idx << ".png";
                    save_screenshot(rend.camera.width, rend.camera.height,
                                    sst.str());
                }
                anim.update(rend);
            }

            draw_imgui(rend, tree);

            glfwSwapBuffers(window);
            glFinish();
            glfwPollEvents();
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
