#include <cstdlib>
#include <iostream>
#include <chrono>

#include <GLES3/gl3.h>
#include <GLFW/glfw3.h>

#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/html5.h>
#include <emscripten/fetch.h>

#define BEGIN_PROFILE auto start = std::chrono::high_resolution_clock::now()
#define PROFILE(x)                                                    \
    do {                                                              \
        printf("%s: %f ns\n", #x,                                     \
               std::chrono::duration<double, std::nano>(              \
                   std::chrono::high_resolution_clock::now() - start) \
                   .count());                                         \
        start = std::chrono::high_resolution_clock::now();            \
    } while (false)

namespace {
// cppReportProgress is a JS function (emModule.js) which will be called if you
// call report_progress in C++
EM_JS(void, report_progress, (double x), { cppReportProgress(x); });

GLFWwindow* window;
volrend::N3Tree tree;
volrend::VolumeRenderer renderer;
const int FPS_AVERAGE_FRAMES = 100;
struct {
    bool measure_fps = false;
    int curr_fps_frame = -1;
    std::chrono::high_resolution_clock::time_point tstart;
} gui;

bool init_gl() {
    /* Initialize GLFW */
    if (!glfwInit()) return false;

    int width = 800, height = 800;
    window = glfwCreateWindow(width, height, "volrend viewer", NULL, NULL);

    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glClearColor(1., 1., 1., 1.);  // Clear white
    glClear(GL_COLOR_BUFFER_BIT);

    glfwGetFramebufferSize(window, &width, &height);
    renderer.options.step_size = 1e-3f;
    renderer.options.stop_thresh = 2e-2f;
    renderer.camera.movement_speed = 2.0f;

    renderer.set(tree);
    renderer.resize(width, height);
    return true;
}

// ---------
// Events
void redraw() {
    glClear(GL_COLOR_BUFFER_BIT);
    if (gui.measure_fps) {
        if (gui.curr_fps_frame == FPS_AVERAGE_FRAMES) gui.curr_fps_frame = 0;
        if (gui.curr_fps_frame == -1) {
            gui.tstart = std::chrono::high_resolution_clock::now();
            gui.curr_fps_frame = 0;
        } else if (gui.curr_fps_frame == 0) {
            auto tend = std::chrono::high_resolution_clock::now();
            printf("FPS: %f\n", FPS_AVERAGE_FRAMES * 1e3 /
                                    std::chrono::duration<double, std::milli>(
                                        tend - gui.tstart)
                                        .count());
            gui.tstart = std::chrono::high_resolution_clock::now();
        }
        renderer.render();
        ++gui.curr_fps_frame;
    } else {
        renderer.render();
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void on_key(int key, bool ctrl, bool shift, bool alt) {
    if (key == GLFW_KEY_T) {
        std::cout << "FPS counting started\n";
        gui.measure_fps ^= 1;
        gui.curr_fps_frame = -1;
    }
}
void on_mousedown(int x, int y, bool middle) {
    renderer.camera.begin_drag(x, y, middle, !middle);
}
void on_mousemove(int x, int y) {
    if (renderer.camera.is_dragging()) {
        renderer.camera.drag_update(x, y);
    }
}
void on_mouseup() { renderer.camera.end_drag(); }
bool is_camera_moving() { return renderer.camera.is_dragging(); }
void on_mousewheel(bool upwards, int distance, int x, int y) {
    const float speed_fact = 1e-1f;
    renderer.camera.move(renderer.camera.v_back *
                         (upwards ? -speed_fact : speed_fact));
}

void on_resize(int width, int height) { renderer.resize(width, height); }

// Remote octree file loading
void load_remote(const std::string& url) {
    auto _load_remote_download_success = [](emscripten_fetch_t* fetch) {
        // Decompress the tree in memory
        tree.open_mem(fetch->data, fetch->numBytes);
        emscripten_fetch_close(fetch);  // Free data associated with the fetch.
        renderer.set(tree);
        tree.clear_cpu_memory();
        // redraw();
        report_progress(101.0f);  // Report finished loading
    };

    auto _load_remote_download_failed = [](emscripten_fetch_t* fetch) {
        printf("Downloading %s failed, HTTP failure status code: %d.\n",
               fetch->url, fetch->status);
        emscripten_fetch_close(fetch);  // Also free data on failure.
    };

    auto _load_remote_download_progress = [](emscripten_fetch_t* fetch) {
        report_progress(fetch->dataOffset * 100.0 / fetch->totalBytes);
    };

    emscripten_fetch_attr_t attr;

    // Download with emscripten fetch API
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = _load_remote_download_success;
    attr.onerror = _load_remote_download_failed;
    attr.onprogress = _load_remote_download_progress;
    emscripten_fetch(&attr, url.c_str());
}

// JS function bindings
EMSCRIPTEN_BINDINGS(Volrend) {
    using namespace emscripten;
    function("on_key", &on_key);
    function("is_camera_moving", &is_camera_moving);
    function("on_mousedown", &on_mousedown);
    function("on_mousemove", &on_mousemove);
    function("on_mouseup", &on_mouseup);
    function("on_mousewheel", &on_mousewheel);
    function("on_resize", &on_resize);
    function("redraw", &redraw);
    function("load_remote", &load_remote);
}
}  // namespace

int main(int argc, char** argv) {
    if (!init_gl()) return EXIT_FAILURE;
    // Make camera move a bit faster (feels slow for some reason)

    emscripten_set_main_loop(redraw, 0, 1);
}
