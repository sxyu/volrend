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
GLFWwindow* window;  // GLFW window
volrend::N3Tree tree;
// volrend::VolumeRenderer renderer;

bool init_gl() {
    /* Initialize the library */
    if (!glfwInit()) return false;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(800, 800, "volrend viewer", NULL, NULL);
    // Init glfw
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glClearColor(1., 1., 1., 1.);  // Clear white
    return true;
}

// ---------

void redraw() {
    glClear(GL_COLOR_BUFFER_BIT);
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    std::cout << "redraw"
              << "\n";

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void on_key(int key, bool ctrl, bool shift, bool alt) {}
void on_mousedown(int x, int y) {}
void on_mousemove(int x, int y) {}
void on_mouseup(int x, int y) {}
void on_mousewheel(bool upwards, int distance, int x, int y) {}

// Remote file load handling
void load_remote(const std::string& url) {
    auto _load_remote_download_success = [](emscripten_fetch_t* fetch) {
        // Decompress the tree in memory
        tree.open_mem(fetch->data, fetch->numBytes);
        emscripten_fetch_close(fetch);  // Free data associated with the fetch.
        std::cout << tree.capacity << "cap\n";
        redraw();
    };

    auto _load_remote_download_failed = [](emscripten_fetch_t* fetch) {
        printf("Downloading %s failed, HTTP failure status code: %d.\n",
               fetch->url, fetch->status);
        emscripten_fetch_close(fetch);  // Also free data on failure.
    };

    auto _load_remote_download_progress = [](emscripten_fetch_t* fetch) {
        if (fetch->totalBytes) {
            printf("Downloading %s.. %.2f%% complete.\n", fetch->url,
                   fetch->dataOffset * 100.0 / fetch->totalBytes);
        } else {
            printf("Downloading %s.. %lld bytes complete.\n", fetch->url,
                   fetch->dataOffset + fetch->numBytes);
        }
    };

    emscripten_fetch_attr_t attr;

    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = _load_remote_download_success;
    attr.onerror = _load_remote_download_failed;
    attr.onprogress = _load_remote_download_progress;
    emscripten_fetch(&attr, url.c_str());
}

EMSCRIPTEN_BINDINGS(Volrend) {
    using namespace emscripten;
    function("on_key", &on_key);
    function("on_mousedown", &on_mousedown);
    function("on_mousemove", &on_mousemove);
    function("on_mouseup", &on_mouseup);
    function("on_mousewheel", &on_mousewheel);
    function("redraw", &redraw);
    function("load_remote", &load_remote);
}
}  // namespace

int main(int argc, char** argv) {
    if (!init_gl()) return EXIT_FAILURE;  // GL initialization failed
    redraw();
    int ubo_num, ubo_sz;
    glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, &ubo_num);
    glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &ubo_sz);
    std::cout << "Your HW buffer texture texel count limit is "
              << ubo_num * ubo_sz
              << " items.\n"
                 "We will create up to "
              << 8 << " textures to fit the volume data\n";
}
