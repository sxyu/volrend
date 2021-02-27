#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <cstdlib>
#include <cstdio>

#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include "imgui_impl_opengl3.h"
#include "imgui_impl_glfw.h"

namespace volrend {

// Starting CUDA/OpenGL interop code from
// https://gist.github.com/allanmac/4ff11985c3562830989f

namespace {

#define GET_RENDERER(window) \
    (*((VolumeRenderer*)glfwGetWindowUserPointer(window)))

void glfw_update_title(GLFWwindow* window) {
    // static fps counters
    // Source: http://antongerdelan.net/opengl/glcontext2.html
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        char tmp[128];
        sprintf(tmp, "volrend viewer - FPS: %.2f", fps);
        glfwSetWindowTitle(window, tmp);
        // glfwSetWindowTitle(window, "volrend viewer");
        frame_count = 0;
    }

    frame_count++;
}

void draw_imgui(VolumeRenderer& rend) {
    auto& cam = rend.camera;
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20.f, 20.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(250.f, 170.f), ImGuiCond_Once);
    ImGui::Begin("Camera");

    // Update vectors indirectly since we need to normalize on change
    // (press update button) and it would be too confusing to keep normalizing
    static glm::vec3 world_up_tmp = rend.camera.v_world_up;
    static glm::vec3 world_down_prev = rend.camera.v_world_up;
    static glm::vec3 back_tmp = rend.camera.v_back;
    static glm::vec3 forward_prev = rend.camera.v_back;
    if (cam.v_world_up != world_down_prev)
        world_up_tmp = world_down_prev = cam.v_world_up;
    if (cam.v_back != forward_prev) back_tmp = forward_prev = cam.v_back;

    ImGui::InputFloat3("center", glm::value_ptr(cam.center));
    ImGui::InputFloat3("origin", glm::value_ptr(cam.origin));
    ImGui::SliderFloat("focal", &cam.focal, 300.f, 7000.f);
    ImGui::Spacing();
    ImGui::InputFloat3("world_up", glm::value_ptr(world_up_tmp));
    ImGui::InputFloat3("back", glm::value_ptr(back_tmp));
    if (ImGui::Button("update dirs")) {
        cam.v_world_up = glm::normalize(world_up_tmp);
        cam.v_back = glm::normalize(back_tmp);
    }
    ImGui::End();
    // End camera window

    // Render window
    ImGui::SetNextWindowPos(ImVec2(20.f, 195.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(250.f, 145.f), ImGuiCond_Once);
    ImGui::Begin("Rendering");

    static float inv_step_size = 1.0f / rend.options.step_size;
    if (ImGui::SliderFloat("1/step_size", &inv_step_size, 128.f, 10000.f)) {
        rend.options.step_size = 1.f / inv_step_size;
    }
    ImGui::SliderFloat("sigma_thresh", &rend.options.sigma_thresh, 0.f, 100.0f);
    ImGui::SliderFloat("stop_thresh", &rend.options.stop_thresh, 0.001f, 0.4f);
    ImGui::SliderFloat("bg_brightness", &rend.options.background_brightness,
                       0.f, 1.0f);
#ifdef VOLREND_CUDA
    ImGui::Checkbox("show_grid", &rend.options.show_grid);
    ImGui::SameLine();
#endif
    ImGui::Text("Backend: %s", rend.get_backend());

    ImGui::End();
    // End render window

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
            case GLFW_KEY_Q: {
                float speed = 0.002f;
                if (mods & GLFW_MOD_SHIFT) speed *= 5.f;
                if (key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_E)
                    speed = -speed;
                const auto& vec = (key == GLFW_KEY_A || key == GLFW_KEY_D)
                                      ? cam.v_right
                                      : (key == GLFW_KEY_W || key == GLFW_KEY_S)
                                            ? -cam.v_back
                                            : -cam.v_up;
                cam.move(vec * speed);
            } break;

            case GLFW_KEY_MINUS:
                cam.focal *= 0.99f;
                break;

            case GLFW_KEY_EQUAL:
                cam.focal *= 1.01f;
                break;

            case GLFW_KEY_0:
                cam.focal = CAMERA_DEFAULT_FOCAL_LENGTH;
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
    if (action == GLFW_PRESS) {
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
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto& cam = GET_RENDERER(window).camera;
    // Focal length adjusting was very annoying so changed it to movement in z
    // cam.focal *= (yoffset > 0.f) ? 1.01f : 0.99f;
    const float speed_fact = 1e-1f;
    cam.move(cam.v_back * ((yoffset < 0.f) ? speed_fact : -speed_fact));
}

GLFWwindow* glfw_init(const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) std::exit(EXIT_FAILURE);

    // glfwWindowHint(GLFW_DEPTH_BITS, GL_TRUE);
    glfwWindowHint(GLFW_DEPTH_BITS, GL_FALSE);
    glfwWindowHint(GLFW_STENCIL_BITS, GL_FALSE);

    // glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);
    // glEnable(GL_FRAMEBUFFER_SRGB);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glEnable(GL_DEPTH_TEST);
    // glEnable(GL_CULL_FACE);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    GLFWwindow* window =
        glfwCreateWindow(width, height, "volrend viewer", NULL, NULL);

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
    glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

    return window;
}

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    GET_RENDERER(window).resize(width, height);
}

}  // namespace
}  // namespace volrend

int main(int argc, char* argv[]) {
    using namespace volrend;
    if (argc <= 1) {
        fprintf(stderr, "Expect argument: npz file\n");
        return 1;
    }
    const int device_id = (argc > 2) ? atoi(argv[2]) : -1;

    N3Tree tree(argv[1]);
    int width = 800, height = 800;
    if (tree.use_ndc) {
        width = 1008;
        height = 756;
    }
    GLFWwindow* window = glfw_init(width, height);
    {
        VolumeRenderer rend(device_id);
        if (tree.use_ndc) {
            // Special inital coordinates for NDC
            // (pick average camera)
            rend.camera.center = glm::vec3(0);
            rend.camera.origin = glm::vec3(0, 0, -3);
            rend.camera.v_back = glm::vec3(0, 0, 1);
            rend.camera.v_world_up = glm::vec3(0, 1, 0);
            rend.camera.focal = tree.ndc_focal * 0.25f;
            rend.camera.movement_speed = 0.1f;
            rend.options.step_size = 1.f / 4000.f;
        }
        rend.set(tree);

        // get initial width/height
        {
            glfwGetFramebufferSize(window, &width, &height);
            rend.resize(width, height);
        }

        // SET USER POINTER AND CALLBACKS
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetScrollCallback(window, glfw_scroll_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        // LOOP UNTIL DONE
        while (!glfwWindowShouldClose(window)) {
            // MONITOR FPS
            glfw_update_title(window);

            rend.render();

            draw_imgui(rend);

            glfwSwapBuffers(window);
            glFinish();
            glfwPollEvents();
            // glfwWaitEvents();
            // break;
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
