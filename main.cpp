#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <cuda_gl_interop.h>

#include "volrend/cuda/common.cuh"
#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

namespace volrend {

// ORIGINAL CODE FROM
// https://gist.github.com/allanmac/4ff11985c3562830989f

// FPS COUNTER FROM HERE:
// http://antongerdelan.net/opengl/glcontext2.html

namespace {

#define GET_RENDERER(window) \
    (*((CUDAVolumeRenderer*)glfwGetWindowUserPointer(window)))

void glfw_fps(GLFWwindow* window) {
    // static fps counters
    static double stamp_prev = 0.0;
    static int frame_count = 0;

    // locals
    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5) {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        int width, height;
        char tmp[128];

        glfwGetFramebufferSize(window, &width, &height);

        sprintf(tmp, "(%u x %u) - FPS: %.2f", width, height, fps);

        glfwSetWindowTitle(window, tmp);

        frame_count = 0;
    }

    frame_count++;
}

void glfw_error_callback(int error, const char* description) {
    fputs(description, stderr);
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action,
                       int mods) {
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
                float speed = 0.005f;
                if (mods & GLFW_MOD_SHIFT) speed *= 2.f;
                if (key == GLFW_KEY_S || key == GLFW_KEY_A || key == GLFW_KEY_E)
                    speed = -speed;
                const auto& vec = (key == GLFW_KEY_A || key == GLFW_KEY_D)
                                      ? cam.v_right
                                      : (key == GLFW_KEY_W || key == GLFW_KEY_S)
                                            ? cam.v_forward
                                            : cam.v_down;
                cam.center += vec * speed;
            } break;

            case GLFW_KEY_1:
                cam.center = {0.5f, 0.0f, 0.5f};
                cam.v_forward = {0.0f, 1.0f, 0.0f};
                cam.v_world_down = {0.0f, 0.0f, -1.0f};
                break;

            case GLFW_KEY_2:
                cam.center = {0.5f, 1.0f, 0.5f};
                cam.v_forward = {0.0f, -1.0f, 0.0f};
                cam.v_world_down = {0.0f, 0.0f, 1.0f};
                break;

            case GLFW_KEY_3:
                cam.center = {0.0f, 0.5f, 0.5f};
                cam.v_forward = {1.0f, 0.0f, 0.0f};
                cam.v_world_down = {0.0f, 1.0f, 0.0f};
                break;

            case GLFW_KEY_4:
                cam.center = {1.0f, 0.5f, 0.5f};
                cam.v_forward = {-1.0f, 0.0f, 0.0f};
                cam.v_world_down = {0.0f, -1.0f, 0.0f};
                break;

            case GLFW_KEY_5:
                cam.center = {0.5f, 0.5f, 1.0f};
                cam.v_forward = {0.0f, 0.0f, -1.0f};
                cam.v_world_down = {1.0f, 0.0f, 0.0f};
                break;

            case GLFW_KEY_6:
                cam.center = {0.5f, 0.5f, 0.0f};
                cam.v_forward = {0.0f, 0.0f, 1.0f};
                cam.v_world_down = {-1.0f, 0.0f, 0.0f};
                break;
        }
    }
}

void glfw_mouse_button_callback(GLFWwindow* window, int button, int action,
                                int mods) {
    auto& rend = GET_RENDERER(window);
    auto& cam = rend.camera;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            cam.begin_drag(x, y, mods & GLFW_MOD_SHIFT,
                           button == GLFW_MOUSE_BUTTON_RIGHT);
        } else if (action == GLFW_RELEASE) {
            cam.end_drag();
        }
    }
}

void glfw_cursor_pos_callback(GLFWwindow* window, double x, double y) {
    GET_RENDERER(window).camera.drag_update(x, y);
}

void glfw_init(GLFWwindow** window, const int width, const int height) {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    *window =
        glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);

    if (*window == NULL) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(*window);

    // set up GLEQ
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fputs("GLEW init failed\n", stderr);
        getchar();
        glfwTerminate();
        return;
    }

    // ignore vsync for now
    glfwSwapInterval(0);

    // only copy r/g/b
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
}

void glfw_window_size_callback(GLFWwindow* window, int width, int height) {
    GET_RENDERER(window).resize(width, height);
}

}  // namespace
}  // namespace volrend

int main(int argc, char* argv[]) {
    using namespace volrend;
    GLFWwindow* window;

    glfw_init(&window, 960, 1039);
    cudaError_t cuda_err;

    GLint gl_device_id;
    GLuint gl_device_count;
    cuda_err = cuda(
        GLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));

    int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
    cuda_err = cuda(SetDevice(cuda_device_id));

    // MULTI-GPU?
    const bool multi_gpu = gl_device_id != cuda_device_id;

    // INFO
    struct cudaDeviceProp props;

    cuda_err = cuda(GetDeviceProperties(&props, gl_device_id));
    printf("OpenGL : %-24s (%d)\n", props.name, props.multiProcessorCount);

    cuda_err = cuda(GetDeviceProperties(&props, cuda_device_id));
    printf("CUDA   : %-24s (%d)\n", props.name, props.multiProcessorCount);

    {
        N3Tree tree("lego.npz");
        CUDAVolumeRenderer rend;

        // get initial width/height
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            rend.resize(width, height);
        }

        // SET USER POINTER AND CALLBACKS
        glfwSetWindowUserPointer(window, &rend);
        glfwSetKeyCallback(window, glfw_key_callback);
        glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
        glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);
        glfwSetFramebufferSizeCallback(window, glfw_window_size_callback);

        // LOOP UNTIL DONE
        while (!glfwWindowShouldClose(window)) {
            // MONITOR FPS
            glfw_fps(window);

            rend.render(tree);
            rend.swap();

            glfwSwapBuffers(window);
            glfwPollEvents();
            // glfwWaitEvents();
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    cuda(DeviceReset());
}
