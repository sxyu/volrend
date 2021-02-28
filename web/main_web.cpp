#include <cstdlib>
#include <iostream>

#include <GLES3/gl3.h>
#include <GLFW/glfw3.h>

#include "volrend/renderer.hpp"
#include "volrend/n3tree.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/html5.h>
#include <emscripten/fetch.h>

namespace {
// cppReportProgress is a JS function (emModule.js) which will be called if you
// call report_progress in C++
EM_JS(void, report_progress, (double x), { cppReportProgress(x); });

GLFWwindow* window;
volrend::N3Tree tree;
volrend::VolumeRenderer renderer;

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
    // renderer.options.step_size = 1.0f / 4000.0;

    renderer.set(tree);
    renderer.resize(width, height);
    return true;
}

// ---------
// Events
void redraw() {
    glClear(GL_COLOR_BUFFER_BIT);

    renderer.render();

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void on_key(int key, bool ctrl, bool shift, bool alt) {}
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
        redraw();
        report_progress(101.0f);
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
    renderer.camera.movement_speed = 2.0f;
}
