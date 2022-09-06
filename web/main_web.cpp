#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <memory>

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
EM_JS(void, populate_layers, (), { populateLayers(); });
EM_JS(void, show_loading_screen, (), { showLoadingScreen(); });
EM_JS(void, update_fps, (double x), { cppUpdateFPS(x); });

GLFWwindow* window;
std::unique_ptr<volrend::Renderer> renderer;
const int FPS_AVERAGE_FRAMES = 20;
struct {
    bool mesh_default_visible = true;

    bool measure_fps = true;
    int curr_fps_frame = -1;
    double curr_fps = 0.0;
    std::chrono::high_resolution_clock::time_point tstart;

    int update_frames = FPS_AVERAGE_FRAMES * 2;
    void require_update() {
        update_frames = FPS_AVERAGE_FRAMES * 2;
    }
} gui;

// ---------
// Events
void redraw() {
    if (gui.update_frames <= 0) return;
    --gui.update_frames;
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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
        renderer->render();
        ++gui.curr_fps_frame;
    } else {
        renderer->render();
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
}

float get_fps() { return gui.curr_fps; }
void toggle_fps_counter() {
    gui.measure_fps ^= 1;
    gui.curr_fps_frame = -1;
}

// std::string get_basis_format() { return tree.data_format.to_string(); }
// int get_basis_dim() { return tree.data_format.basis_dim; }

void set_time(int time) {
    renderer->time = time;
    gui.require_update();
}
int get_time() { return renderer->time; }
int mesh_max_time() {
    int time = 0;
    for (auto& mesh : renderer->meshes) {
        time = std::max<int>(mesh.time, time);
    }
    return time;
}

void on_key(int key, bool ctrl, bool shift, bool alt) {
    auto& cam = renderer->camera;
    switch (key) {
        case GLFW_KEY_T:
            toggle_fps_counter();
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
void on_mousedown(int x, int y, bool middle) {
    renderer->camera.begin_drag(x, y, middle, !middle);
}
void on_mousemove(int x, int y) {
    if (renderer->camera.is_dragging()) {
        renderer->camera.drag_update(x, y);
        gui.require_update();
    }
}
void on_mouseup() { renderer->camera.end_drag(); }
bool is_camera_moving() { return renderer->camera.is_dragging(); }
void on_mousewheel(bool upwards, int distance, int x, int y) {
    float speed_fact = 1e-1f;
    float radius = fmaxf(glm::length(renderer->camera.center), 1.f);
    speed_fact *= radius;
    renderer->camera.move(renderer->camera.v_back *
                          (upwards ? -speed_fact : speed_fact));
    gui.require_update();
}

void on_resize(int width, int height) {
    renderer->resize(width, height);
    gui.require_update();
}

// ** Data Loading
// Remote octree file loading from a url
void load_tree_remote(const std::string& url) {
    show_loading_screen();
    auto _load_remote_download_success = [](emscripten_fetch_t* fetch) {
        // Decompress the tree in memory
        volrend::N3Tree tree;
        tree.open_mem(fetch->data, fetch->numBytes);
        emscripten_fetch_close(fetch);  // Free data associated with the fetch.
        renderer->add(std::move(tree));
        tree.clear_cpu_memory();
        report_progress(101.0f);  // Report finished loading
        gui.require_update();
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

// Load from emscripten MEMFS (retrieved from file input in JS)
void load_tree_local(const std::string& path) {
    report_progress(50.0f);  // Fake progress (loaded to MEMFS at this point)
    volrend::N3Tree tree;
    tree.open(path);
    renderer->set(std::move(tree));
    tree.clear_cpu_memory();
    report_progress(101.0f);  // Report finished loading
    gui.require_update();
}

// Load OBJ mesh from URL (weirdly slow on firefox after refreshing..)
void load_obj_remote(const std::string& url) {
    auto _load_remote_download_success = [](emscripten_fetch_t* fetch) {
        const char* data = fetch->data;
        std::string str(data, data + fetch->numBytes);
        printf("OBJ download: Received %llu bytes\n", fetch->numBytes);
        emscripten_fetch_close(fetch);  // Free data associated with the fetch.
        volrend::Mesh tmp = volrend::Mesh::load_mem_basic_obj(str);
        if (tmp.vert.size()) {
            tmp.update();
            renderer->meshes.push_back(std::move(tmp));
            printf("Load OBJ success\n");
            populate_layers();
        } else {
            printf("Load OBJ failed\n");
        }
        gui.require_update();
    };

    auto _load_remote_download_failed = [](emscripten_fetch_t* fetch) {
        printf("Downloading OBJ %s failed, HTTP failure status code: %d.\n",
               fetch->url, fetch->status);
        emscripten_fetch_close(fetch);
    };

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = _load_remote_download_success;
    attr.onerror = _load_remote_download_failed;

    emscripten_fetch(&attr, url.c_str());
}

// Load OBJ mesh from MEMFS
void load_obj_local(const std::string& path) {
    volrend::Mesh tmp = volrend::Mesh::load_basic_obj(path);
    if (tmp.vert.size()) {
        tmp.update();
        renderer->meshes.push_back(std::move(tmp));
        printf("Load OBJ success\n");
    } else {
        printf("Load OBJ failed\n");
    }
    report_progress(101.0f);  // Report finished loading
    gui.require_update();
}

void _append_meshes(std::vector<volrend::Mesh>& mesh_list,
                    std::vector<volrend::Mesh>&& to_add) {
    size_t n_meshes = mesh_list.size();
    mesh_list.resize(n_meshes + to_add.size());
    std::move(to_add.begin(), to_add.end(), mesh_list.begin() + n_meshes);
}

// Load custom drawlist format from url
void load_drawlist_remote(const std::string& url) {
    auto _load_remote_download_success = [](emscripten_fetch_t* fetch) {
        printf("Drawlist download: Received %llu bytes\n", fetch->numBytes);
        _append_meshes(renderer->meshes, volrend::Mesh::open_drawlist_mem(
                                             fetch->data, fetch->numBytes,
                                             gui.mesh_default_visible));
        emscripten_fetch_close(fetch);  // Free data associated with the fetch.
        populate_layers();
        gui.require_update();
    };

    auto _load_remote_download_failed = [](emscripten_fetch_t* fetch) {
        printf("Downloading meshes %s failed, HTTP failure status code: %d.\n",
               fetch->url, fetch->status);
        emscripten_fetch_close(fetch);
    };

    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY;
    attr.onsuccess = _load_remote_download_success;
    attr.onerror = _load_remote_download_failed;

    emscripten_fetch(&attr, url.c_str());
}

// Load custom drawlist format from MEMFS
void load_drawlist_local(const std::string& path) {
    _append_meshes(renderer->meshes, volrend::Mesh::open_drawlist(path));
    report_progress(101.0f);  // Report finished loading
    gui.require_update();
}

// Minor util to check if string ends with another
bool _ends_with(const std::string& str, const std::string& end) {
    if (str.length() >= end.length())
        return !str.compare(str.length() - end.length(), end.length(), end);
    return false;
}

// Automatically choose what to load according to extension
void load_remote(const std::string& url) {
    if (_ends_with(url, ".obj")) {
        load_obj_remote(url);
    } else if (_ends_with(url, ".draw.npz")) {
        load_drawlist_remote(url);
    } else {
        load_tree_remote(url);
    }
}

void load_local(const std::string& path) {
    if (_ends_with(path, ".obj")) {
        load_obj_local(path);
    } else if (_ends_with(path, ".draw.npz")) {
        load_drawlist_local(path);
    } else {
        load_tree_local(path);
    }
}

// ** Options
volrend::RenderOptions get_options() { return renderer->options; }
void set_options(const volrend::RenderOptions& opt) { renderer->options = opt; }

// ** Camera utilities
glm::vec3 arr2vec3(const std::array<float, 3>& arr) {
    return glm::vec3(arr[0], arr[1], arr[2]);
}
std::array<float, 3> vec32arr(const glm::vec3& v) {
    return std::array<float, 3>{v[0], v[1], v[2]};
}
//
float get_focal() { return renderer->camera.fx; }
void set_focal(float fx) {
    renderer->camera.fy = renderer->camera.fx = fx;
    gui.require_update();
}
std::array<float, 3> get_world_up() {
    return vec32arr(renderer->camera.v_world_up);
}
void set_world_up(std::array<float, 3> xyz) {
    renderer->camera.v_world_up = glm::normalize(arr2vec3(xyz));
    gui.require_update();
}
std::array<float, 3> get_cam_origin() {
    return vec32arr(renderer->camera.origin);
}
void set_cam_origin(std::array<float, 3> xyz) {
    renderer->camera.origin = arr2vec3(xyz);
    gui.require_update();
}
std::array<float, 3> get_cam_back() {
    return vec32arr(renderer->camera.v_back);
}
void set_cam_back(std::array<float, 3> xyz) {
    renderer->camera.v_back = glm::normalize(arr2vec3(xyz));
    gui.require_update();
}
std::array<float, 3> get_cam_center() {
    return vec32arr(renderer->camera.center);
}
void set_cam_center(std::array<float, 3> xyz) {
    renderer->camera.center = arr2vec3(xyz);
    gui.require_update();
}

// ** Mesh utilities
void mesh_add_cube(std::array<float, 3> xyz, float scale,
                   std::array<float, 3> color) {
    static int cubeid = 0;
    {
        volrend::Mesh cube = volrend::Mesh::Cube(arr2vec3(color));
        cube.model_scale = scale;
        cube.model_translation = arr2vec3(xyz);
        cube.update();
        if (cubeid) cube.name = cube.name + std::to_string(cubeid);
        ++cubeid;
        renderer->meshes.push_back(std::move(cube));
        gui.require_update();
    }
}

void mesh_add_sphere(std::array<float, 3> xyz, float scale,
                     std::array<float, 3> color) {
    static int sphereid = 0;
    {
        volrend::Mesh sph = volrend::Mesh::Sphere(30, 30, arr2vec3(color));
        sph.model_scale = scale;
        sph.model_translation = arr2vec3(xyz);
        sph.update();
        if (sphereid) sph.name = sph.name + std::to_string(sphereid);
        ++sphereid;
        renderer->meshes.push_back(std::move(sph));
        gui.require_update();
    }
}

bool _check_mesh_id(int mesh_id) {
    return (mesh_id < 0 || mesh_id >= (int)renderer->meshes.size());
}

std::string mesh_get_name(int mesh_id) {
    if (_check_mesh_id(mesh_id)) return "";
    return renderer->meshes[mesh_id].name;
}
// Color of first vertex
std::array<float, 3> mesh_get_color(int mesh_id) {
    if (_check_mesh_id(mesh_id)) return std::array<float, 3>{0.f, 0.f, 0.f};
    const float* ptr = &renderer->meshes[mesh_id].vert[3];
    return std::array<float, 3>{ptr[0], ptr[1], ptr[2]};
}
bool mesh_get_visible(int mesh_id) {
    if (_check_mesh_id(mesh_id)) return false;
    return renderer->meshes[mesh_id].visible;
}
void mesh_set_translation(int mesh_id, std::array<float, 3> xyz) {
    if (_check_mesh_id(mesh_id)) return;
    renderer->meshes[mesh_id].model_translation = arr2vec3(xyz);
    gui.require_update();
}
void mesh_set_rotation(int mesh_id, std::array<float, 3> aa) {
    if (_check_mesh_id(mesh_id)) return;
    renderer->meshes[mesh_id].model_rotation = arr2vec3(aa);
    gui.require_update();
}
void mesh_set_scale(int mesh_id, float scale) {
    if (_check_mesh_id(mesh_id)) return;
    renderer->meshes[mesh_id].model_scale = scale;
    gui.require_update();
}
void mesh_set_visible(int mesh_id, bool visible) {
    if (_check_mesh_id(mesh_id)) return;
    renderer->meshes[mesh_id].visible = visible;
    gui.require_update();
}
// Default visibility of meshes loaded thru remote drawlists
void mesh_set_default_visible(bool visible) {
    gui.mesh_default_visible = visible;
    gui.require_update();
}
void mesh_set_unlit(int mesh_id, bool unlit) {
    if (_check_mesh_id(mesh_id)) return;
    renderer->meshes[mesh_id].unlit = unlit;
    gui.require_update();
}
void mesh_delete(int mesh_id) {
    if (_check_mesh_id(mesh_id)) return;
    renderer->meshes.erase(renderer->meshes.begin() + mesh_id);
    gui.require_update();
}
int mesh_count() { return (int)renderer->meshes.size(); }
void mesh_clear_all() {
    renderer->meshes.clear();
    gui.require_update();
}

// JS function bindings
EMSCRIPTEN_BINDINGS(Volrend) {
    using namespace emscripten;
    // Core
    function("redraw", &redraw);

    // Events
    function("on_key", &on_key);
    function("on_mousedown", &on_mousedown);
    function("on_mousemove", &on_mousemove);
    function("on_mouseup", &on_mouseup);
    function("on_mousewheel", &on_mousewheel);
    function("on_resize", &on_resize);

    // Data loading
    function("load_tree_remote", &load_tree_remote);
    function("load_tree_local", &load_tree_local);
    function("load_obj_remote", &load_obj_remote);
    function("load_obj_local", &load_obj_local);
    function("load_drawlist_remote", &load_drawlist_remote);
    function("load_drawlist_local", &load_drawlist_local);
    function("load_remote", &load_remote);
    function("load_local", &load_local);

    // Meshes
    function("mesh_add_cube", &mesh_add_cube);
    function("mesh_add_sphere", &mesh_add_sphere);
    function("mesh_get_name", &mesh_get_name);
    function("mesh_get_color", &mesh_get_color);
    function("mesh_get_visible", &mesh_get_visible);
    function("mesh_set_translation", &mesh_set_translation);
    function("mesh_set_rotation", &mesh_set_rotation);
    function("mesh_set_scale", &mesh_set_scale);
    function("mesh_set_visible", &mesh_set_visible);
    function("mesh_set_default_visible", &mesh_set_default_visible);
    function("mesh_set_unlit", &mesh_set_unlit);
    function("mesh_delete", &mesh_delete);
    function("mesh_count", &mesh_count);
    function("mesh_clear_all", &mesh_clear_all);

    // Misc
    function("is_camera_moving", &is_camera_moving);

    value_object<volrend::RenderOptions>("RenderOptions")
        .field("step_size", &volrend::RenderOptions::step_size)
        .field("sigma_thresh", &volrend::RenderOptions::sigma_thresh)
        .field("stop_thresh", &volrend::RenderOptions::stop_thresh)
        .field("background_brightness",
               &volrend::RenderOptions::background_brightness)
        .field("render_bbox", &volrend::RenderOptions::render_bbox)
        .field("basis_minmax", &volrend::RenderOptions::basis_minmax)
        .field("rot_dirs", &volrend::RenderOptions::rot_dirs)
        .field("show_grid", &volrend::RenderOptions::show_grid)
        .field("grid_max_depth", &volrend::RenderOptions::grid_max_depth);

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
    function("get_world_up", &get_world_up);
    function("set_world_up", &set_world_up);
    function("get_cam_origin", &get_cam_origin);
    function("set_cam_origin", &set_cam_origin);
    function("get_cam_back", &get_cam_back);
    function("set_cam_back", &set_cam_back);
    function("get_cam_center", &get_cam_center);
    function("set_cam_center", &set_cam_center);
    function("toggle_fps_counter", &toggle_fps_counter);
    function("get_fps", &get_fps);
    // function("get_basis_dim", &get_basis_dim);
    // function("get_basis_format", &get_basis_format);

    function("get_time", &get_time);
    function("mesh_max_time", &mesh_max_time);
    function("set_time", &set_time);
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

    glClearDepthf(1.0);
    glDepthFunc(GL_LESS);

    glfwGetFramebufferSize(window, &width, &height);
    renderer = std::make_unique<volrend::Renderer>();
    renderer->camera.movement_speed = 2.0f;
    renderer->resize(width, height);
    emscripten_set_main_loop(redraw, 0, 1);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (!init_gl()) return EXIT_FAILURE;
}
