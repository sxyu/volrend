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
EM_JS(void, update_fps, (double x), { cppUpdateFPS(x); });

GLFWwindow* window;
volrend::N3Tree tree;
volrend::VolumeRenderer renderer;
const int FPS_AVERAGE_FRAMES = 20;
struct {
    bool measure_fps = true;
    int curr_fps_frame = -1;
    double curr_fps = 0.0;
    std::chrono::high_resolution_clock::time_point tstart;
} gui;

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
            gui.curr_fps =
                FPS_AVERAGE_FRAMES * 1e3 /
                std::chrono::duration<double, std::milli>(tend - gui.tstart)
                    .count();
            update_fps(gui.curr_fps);
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

float get_fps() { return gui.curr_fps; }
void toggle_fps_counter() {
    gui.measure_fps ^= 1;
    gui.curr_fps_frame = -1;
}

std::string get_basis_format() { return tree.data_format.to_string(); }
int get_basis_dim() { return tree.data_format.basis_dim; }

void on_key(int key, bool ctrl, bool shift, bool alt) {
    if (key == GLFW_KEY_T) {
        toggle_fps_counter();
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

volrend::RenderOptions get_options() { return renderer.options; }
void set_options(const volrend::RenderOptions& opt) { renderer.options = opt; }
float get_focal() { return renderer.camera.fx; }
void set_focal(float fx) { renderer.camera.fy = renderer.camera.fx = fx; }

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
    value_object<volrend::RenderOptions>("RenderOptions")
        .field("step_size", &volrend::RenderOptions::step_size)
        .field("sigma_thresh", &volrend::RenderOptions::sigma_thresh)
        .field("stop_thresh", &volrend::RenderOptions::stop_thresh)
        .field("background_brightness",
               &volrend::RenderOptions::background_brightness)
        .field("render_bbox", &volrend::RenderOptions::render_bbox)
        .field("basis_minmax", &volrend::RenderOptions::basis_minmax)
        .field("rot_dirs", &volrend::RenderOptions::rot_dirs);

    value_array<std::array<int, 2>>("array_int_2")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>());

    value_array<std::array<float, 3>>("array_float_3")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>());

    value_array<std::array<float, 6>>("array_float_6")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>())
        .element(emscripten::index<3>())
        .element(emscripten::index<4>())
        .element(emscripten::index<5>());
    function("get_options", &get_options);
    function("set_options", &set_options);
    function("get_focal", &get_focal);
    function("set_focal", &set_focal);
    function("toggle_fps_counter", &toggle_fps_counter);
    function("get_fps", &get_fps);
    function("get_basis_dim", &get_basis_dim);
    function("get_basis_format", &get_basis_format);
}

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

    glClearColor(1., 1., 1., 1.);  // Clear white
    glClear(GL_COLOR_BUFFER_BIT);

    glfwGetFramebufferSize(window, &width, &height);
    renderer.options.step_size = 1e-4f;
    renderer.options.stop_thresh = 1e-2f;
    renderer.camera.movement_speed = 2.0f;

    renderer.set(tree);
    renderer.resize(width, height);
    emscripten_set_main_loop(redraw, 0, 1);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (!init_gl()) return EXIT_FAILURE;
}
