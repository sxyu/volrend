#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>

#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include "volrend/internal/opts.hpp"
#include "volrend/internal/imwrite.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

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

    float u_curr = (1.f - q) * u_start + q * u_end;
    float v_curr = (1.f - q) * v_start + q * v_end;
    float d_curr = (1.f - q) * d_start + q * d_end;
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
        std::cout << "Wrote " << path << "\n";
    } else {
        std::cout << "Failed to save screenshot\n";
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
            std::cerr << "WARNING: cannot animate with < 2 keyframes\n";
            return;
        }
        anim_once(keyframes[0], keyframes[1], previewing, -1.f, 0);
        if (!previewing) {
            if (!std::filesystem::exists(output_folder)) {
                std::filesystem::create_directory(output_folder);
            }
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
        curr.origin = (1.f - q) * start.origin + q * end.origin;

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
            curr.center = (1.f - q) * start.center + q * end.center;
            curr.v_back = (1.f - q) * start.v_back + q * end.v_back;
        }
        curr.fx = (1.f - q) * start.fx + q * end.fx;
        curr.fy = (1.f - q) * start.fy + q * end.fy;
        curr.opt = end.opt;
        curr.opt.background_brightness =
            (1.f - q) * start.opt.background_brightness +
            q * end.opt.background_brightness;
        curr.opt.step_size =
            (1.f - q) * start.opt.step_size + q * end.opt.step_size;
        curr.opt.stop_thresh =
            (1.f - q) * start.opt.stop_thresh + q * end.opt.stop_thresh;
        curr.opt.sigma_thresh =
            (1.f - q) * start.opt.sigma_thresh + q * end.opt.sigma_thresh;
        if (start.opt.enable_probe) {
            for (int i = 0; i < 3; ++i) {
                curr.opt.probe[i] =
                    (1.f - q) * start.opt.probe[i] + q * end.opt.probe[i];
            }
        }
        for (int i = 0; i < 6; ++i) {
            curr.opt.render_bbox[i] = (1.f - q) * start.opt.render_bbox[i] +
                                      q * end.opt.render_bbox[i];
        }

        // FIXME interp rot dirs (axis-angle) properly
        for (int i = 0; i < 3; ++i) {
            curr.opt.rot_dirs[i] =
                (1.f - q) * start.opt.rot_dirs[i] + q * end.opt.rot_dirs[i];
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
                state.translation = (1.f - q) * start_state.translation +
                                    q * end_state.translation;
                state.scale =
                    (1.f - q) * start_state.scale + q * end_state.scale;
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

void draw_imgui(VolumeRenderer& rend, N3Tree& tree) {
    auto& cam = rend.camera;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

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
        std::cout << "Animation output folder set to " << path << "\n";
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

#ifdef VOLREND_CUDA
                        ImGui::Checkbox("Show Grid", &kf.opt.show_grid);
                        ImGui::SameLine();
                        ImGui::Checkbox("Render Depth", &kf.opt.render_depth);
                        if (kf.opt.show_grid) {
                            ImGui::SliderInt("grid max depth",
                                             &kf.opt.grid_max_depth, 0, 7);
                        }
#endif

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
#ifdef VOLREND_CUDA
        static ImGui::FileBrowser open_obj_mesh_dialog(
            ImGuiFileBrowserFlags_MultipleSelection);
        if (open_obj_mesh_dialog.GetTitle().empty()) {
            open_obj_mesh_dialog.SetTypeFilters({".obj"});
            open_obj_mesh_dialog.SetTitle("Load basic triangle OBJ");
        }
#endif
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
            std::cout << "Load N3Tree npz: " << path << "\n";
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

#ifdef VOLREND_CUDA
            ImGui::Checkbox("Show Grid", &rend.options.show_grid);
            ImGui::SameLine();
            ImGui::Checkbox("Render Depth", &rend.options.render_depth);
            if (rend.options.show_grid) {
                ImGui::SliderInt("grid max depth", &rend.options.grid_max_depth,
                                 0, 7);
            }
#endif
        }

#ifdef VOLREND_CUDA
        ImGui::SetNextTreeNodeOpen(true, ImGuiCond_Once);
        if (ImGui::CollapsingHeader("Manipulation")) {
            ImGui::BeginGroup();
            for (int i = 0; i < (int)rend.meshes.size(); ++i) {
                auto& mesh = rend.meshes[i];
                if (ImGui::TreeNode(mesh.name.c_str())) {
                    ImGui::PushItemWidth(230);
                    ImGui::SliderFloat3(
                        "trans", glm::value_ptr(mesh.translation), -2.0f, 2.0f);
                    ImGui::SliderFloat3("rot", glm::value_ptr(mesh.rotation),
                                        -M_PI, M_PI);
                    ImGui::SliderFloat("scale", &mesh.scale, 0.01f, 10.0f);
                    ImGui::PopItemWidth();
                    ImGui::Checkbox("visible", &mesh.visible);
                    ImGui::SameLine();
                    ImGui::Checkbox("unlit", &mesh.unlit);

                    ImGui::TreePop();
                }
            }
            ImGui::EndGroup();
            if (ImGui::Button("Add Sphere")) {
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
            if (ImGui::Button("Cube")) {
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
            if (ImGui::Button("Load Tri OBJ")) {
                open_obj_mesh_dialog.Open();
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear All")) {
                rend.meshes.clear();
            }

            ImGui::BeginGroup();
            ImGui::Checkbox("Enable Lumisphere Probe",
                            &rend.options.enable_probe);
            if (rend.options.enable_probe) {
                ImGui::SliderFloat3("probe", rend.options.probe, -2.f, 2.f);
                ImGui::SliderInt("probe_win_sz", &rend.options.probe_disp_size,
                                 50, 800);
            }
            ImGui::EndGroup();
        }
        open_obj_mesh_dialog.Display();
        if (open_obj_mesh_dialog.HasSelected()) {
            // Load mesh
            auto sels = open_obj_mesh_dialog.GetMultiSelected();
            for (auto& fpath : sels) {
                const std::string path = fpath.string();
                Mesh tmp;
                std::cout << "Load OBJ: " << path << "\n";
                tmp.load_basic_obj(path);
                if (tmp.vert.size()) {
                    // Auto offset
                    std::ifstream ifs(path + ".offs");
                    if (ifs) {
                        ifs >> tmp.translation.x >> tmp.translation.y >>
                            tmp.translation.z;
                    }
                    tmp.update();
                    rend.meshes.push_back(std::move(tmp));
                    std::cout << "Load success\n";
                } else {
                    std::cout << "Load failed\n";
                }
            }
            open_obj_mesh_dialog.ClearSelected();
        }

#endif
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
                        (key == GLFW_KEY_A || key == GLFW_KEY_D)
                            ? cam.v_right
                            : (key == GLFW_KEY_W || key == GLFW_KEY_S)
                                  ? -cam.v_back
                                  : -cam.v_up;
                    cam.move(vec * speed);
                }
                break;

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
                    int dim =
                        (key == GLFW_KEY_J || key == GLFW_KEY_L)
                            ? 0
                            : (key == GLFW_KEY_I || key == GLFW_KEY_K) ? 1 : 2;
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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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

    cxxopts::Options cxxoptions("volrend_anim",
                                "OpenGL octree volume rendering animation "
                                "engine (c) VOLREND contributors 2021");

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
    glfwSetWindowTitle(window, "volrend animator");

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
#ifdef VOLREND_CUDA
            glEnable(GL_DEPTH_TEST);
#endif

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
