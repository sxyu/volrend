#include "volrend/common.hpp"

#include "volrend/n3tree.hpp"
#include "volrend/renderer.hpp"
#include "volrend/mesh.hpp"
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <cstdio>
#include <cstdint>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "volrend/internal/glutil.hpp"
#include "volrend/internal/fxaa.shader"

namespace volrend {

namespace {
// DRAWLIST READ UTILS
// Split a string by '__'
std::vector<std::string> split_by_2underscore(const std::string& s) {
    std::vector<std::string> r;
    size_t j = 0;
    for (size_t i = 1; i < s.size(); ++i) {
        if (s[i] == '_' && s[i - 1] == '_') {
            if (i - 1 - j > 0) {
                r.push_back(s.substr(j, i - 1 - j));
            }
            j = i + 1;
        }
    }
    if (j < s.size()) {
        r.push_back(s.substr(j));
    }
    return r;
}

// Get int with default val from a NpyArray map
int map_get_int(const std::map<std::string, cnpy::NpyArray>& m,
                const std::string& key, const int defval, std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return defval;
    } else {
        if (it->second.word_size == 1) {
            return *it->second.data<int8_t>();
        } else if (it->second.word_size == 2) {
            return *it->second.data<int16_t>();
        } else if (it->second.word_size == 4) {
            return *it->second.data<int32_t>();
        } else if (it->second.word_size == 8) {
            return (int)*it->second.data<int64_t>();
        }
        errs << "Invalid word size for int " << it->second.word_size << "\n";
        return 0;
    }
}

float map_get_float(const std::map<std::string, cnpy::NpyArray>& m,
                    const std::string& key, const float defval,
                    std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return defval;
    } else {
        if (it->second.word_size == 2) {
            return *it->second.data<half>();
        } else if (it->second.word_size == 4) {
            return *it->second.data<float>();
        } else if (it->second.word_size == 8) {
            return (float)*it->second.data<double>();
        }
        errs << "Invalid word size for float " << it->second.word_size << "\n";
        return 0;
    }
}

glm::vec3 map_get_vec3(const std::map<std::string, cnpy::NpyArray>& m,
                       const std::string& key, const glm::vec3& defval,
                       std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return defval;
    } else {
        glm::vec3 r;
        auto assn_ptr = [&](auto* ptr) {};
#define _ASSN_PTR_V3(dtype)                          \
    do {                                             \
        const dtype* ptr = it->second.data<dtype>(); \
        r[0] = (float)ptr[0];                        \
        r[1] = (float)ptr[1];                        \
        r[2] = (float)ptr[2];                        \
    } while (0)

        if (it->second.shape.size() != 1 || it->second.shape[0] != 3) {
            errs << "Invalid shape for float3, must be (3,)";
        }

        if (it->second.word_size == 2) {
            _ASSN_PTR_V3(half);
        } else if (it->second.word_size == 4) {
            _ASSN_PTR_V3(float);
        } else if (it->second.word_size == 8) {
            _ASSN_PTR_V3(double);
        } else {
            errs << "Invalid word size for float " << it->second.word_size
                 << "\n";
        }
#undef _ASSN_PTR_V3
        return r;
    }
}

std::vector<float> map_get_floatarr(
    const std::map<std::string, cnpy::NpyArray>& m, const std::string& key,
    std::ostream& errs) {
    const auto it = m.find(key);
    std::vector<float> result;
    if (it == m.end()) {
        return result;
    }

#define _ASSN_PTR_ARR(dtype)                                       \
    do {                                                           \
        const dtype* ptr = it->second.data<dtype>();               \
        std::copy(ptr, ptr + it->second.num_vals, result.begin()); \
    } while (0)

    result.resize(it->second.num_vals);
    if (it->second.word_size == 1) {
        _ASSN_PTR_ARR(uint8_t);
        for (float& v : result) v /= 255.0f;
    } else if (it->second.word_size == 2) {
        _ASSN_PTR_ARR(half);
    } else if (it->second.word_size == 4) {
        _ASSN_PTR_ARR(float);
    } else if (it->second.word_size == 8) {
        _ASSN_PTR_ARR(double);
    } else {
        errs << "Invalid word size for float " << it->second.word_size << "\n";
    }
#undef _ASSN_PTR_ARR
    return result;
}

std::vector<int> map_get_intarr(const std::map<std::string, cnpy::NpyArray>& m,
                                const std::string& key, std::ostream& errs) {
    const auto it = m.find(key);
    std::vector<int> result;
    if (it == m.end()) {
        return result;
    }

#define _ASSN_PTR_ARR(dtype)                                       \
    do {                                                           \
        const dtype* ptr = it->second.data<dtype>();               \
        std::copy(ptr, ptr + it->second.num_vals, result.begin()); \
    } while (0)

    result.resize(it->second.num_vals);
    if (it->second.word_size == 1) {
        _ASSN_PTR_ARR(int8_t);
    } else if (it->second.word_size == 2) {
        _ASSN_PTR_ARR(int16_t);
    } else if (it->second.word_size == 4) {
        _ASSN_PTR_ARR(int32_t);
    } else if (it->second.word_size == 8) {
        _ASSN_PTR_ARR(int64_t);
    } else {
        errs << "Invalid word size for int " << it->second.word_size << "\n";
    }
#undef _ASSN_PTR_ARR
    return result;
}
const uint8_t* map_get_raw(const std::map<std::string, cnpy::NpyArray>& m,
                                const std::string& key, std::ostream& errs) {
    const auto it = m.find(key);
    if (it == m.end()) {
        errs << "Failed to get raw data in npz for key: " << key << "\n";
        return nullptr;
    }
    return it->second.data<uint8_t>();
}

// END DRAWLIST READ UTILS
}  // namespace

struct Renderer::Impl {
    Impl(Camera& camera,
            RendererOptions& options,
            std::vector<Mesh>& meshes,
            std::vector<N3Tree>& trees,
         int& time, int max_tries = 4)
        : camera(camera), options(options), time(time), meshes(meshes), trees(trees) {
        trees.reserve(5);
    }

    ~Impl() { }

    void render() {
        if (!fxaa_program) {
            // Init
            fbo_screen = GLFramebuffer::Screen(camera.width, camera.height);
            fbo_mesh = GLFramebuffer(
                    camera.width, camera.height,
                    {
                        { GL_COLOR_ATTACHMENT0,
                            { GLImage2D::TEXTURE, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT} },
                        { GL_COLOR_ATTACHMENT1,
                            { GLImage2D::TEXTURE, GL_R16F, GL_RED, GL_HALF_FLOAT} },
                        { GL_DEPTH_ATTACHMENT,
                            { GLImage2D::RENDER_BUFFER, GL_DEPTH_COMPONENT24 } }
                    }
                    );
            fbo_tree = GLFramebuffer(
                    camera.width, camera.height,
                    {
                        { GL_COLOR_ATTACHMENT0,
                            { GLImage2D::TEXTURE, GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT} },
                        { GL_COLOR_ATTACHMENT1,
                            { GLImage2D::TEXTURE, GL_R16F, GL_RED, GL_HALF_FLOAT} },
                    }
                    );

            resize(camera.width, camera.height);
            fbo_screen.bind();

            fxaa_program = GLShader(FXAA_SHADER_SRC, "FXAA");
            fxaa_program.set_texture_uniforms({"tex_input"});
            glUseProgram(0);
        }

        camera._update();
        GLfloat depth_inf[] = {1e9, 1e9, 1e9, 1e9};

        fbo_mesh.bind();
        // glEnable(GL_DEPTH_TEST);
        // glDepthFunc(GL_LESS);

#ifdef __EMSCRIPTEN__
        // GLES 3
        glClearDepthf(1e9f);
#else
        glClearDepth(1e9f);
#endif
        GLfloat clear_color[] = {options.background_brightness,
                                 options.background_brightness,
                                 options.background_brightness, 1.f};

        glClearBufferfv(GL_COLOR, 0, clear_color);
        glClearBufferfv(GL_COLOR, 1, depth_inf);
        glClearBufferfv(GL_DEPTH, 0, depth_inf);
        glClear(GL_DEPTH_BUFFER_BIT);

        // Draw meshes/lines/point clouds
        for (const Mesh& mesh : meshes) {
            mesh.draw(camera.w2c, camera.K, false, time);
        }

        // Draw trees in ivnese order (to improve quality of the pseudo-compositing)
        std::vector<std::pair<float, size_t> > trees_sort(trees.size());
        for (size_t i = 0; i < trees.size(); ++i) {
            auto& tree = trees[i];
            glm::vec3 center = tree.model_translation;
            for (int j = 0; j < 3; ++j) {
                center[j] += (1 - tree.offset[j] *  2) * 0.5 / tree.scale[j];
            }
            float dist = glm::length(center - camera.center);
            trees_sort[i].first = -dist;
            trees_sort[i].second = i;
        }
        std::sort(trees_sort.begin(), trees_sort.end());

        // Draw octree structure visualizations (if requested)
        for (size_t i = 0; i < trees.size(); ++i) {
            auto& tree = trees[trees_sort[i].second];
            tree.draw_wire(camera.w2c, camera.K, false, time);
        }

        // Disable depth buffer now
        glDisable(GL_DEPTH_TEST);

        fbo_screen.bind();
        int fbo_id = 0;
        GLFramebuffer* tree_fbos[2] = { &fbo_mesh, &fbo_tree };

        // Draw PlenOctrees
        for (size_t i = 0; i < trees.size(); ++i) {
            auto& tree = trees[trees_sort[i].second];
            fbo_id ^= 1;
            (*tree_fbos[fbo_id ^ 1])[GL_COLOR_ATTACHMENT0].bind_unit(0);
            (*tree_fbos[fbo_id ^ 1])[GL_COLOR_ATTACHMENT1].bind_unit(1);
            tree_fbos[fbo_id]->bind();
            glClearBufferfv(GL_COLOR, 0, clear_color);
            glClearBufferfv(GL_COLOR, 1, depth_inf);
            // glClear(GL_COLOR_BUFFER_BIT);
            if (!tree.draw(camera, time)) {
                // Not drawn, reuse fbo
                fbo_id ^= 1;
            }
        }
        fbo_screen.bind();
        fxaa_program.use();

        if (options.use_fxaa) {
            glUniform2f(fxaa_program["resolution"],
                    static_cast<float>(camera.width),
                    static_cast<float>(camera.height));
            (*tree_fbos[fbo_id])[GL_COLOR_ATTACHMENT0].bind_unit(0);
            util::draw_fs_quad();
        } else {
            tree_fbos[fbo_id]->blit_to(fbo_screen, GL_COLOR_ATTACHMENT0);
        }
        glUseProgram(0);
    }

    void add(const N3Tree& tree) {
        trees.push_back(tree);
    }
    void add(N3Tree&& tree) {
        trees.push_back(tree);
    }

    void clear() { trees.clear(); }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width <= 0 || height <= 0) return;
        camera.width = width;
        camera.height = height;

        fbo_screen.resize(width, height);
        fbo_mesh.resize(width, height);
        fbo_tree.resize(width, height);
    }

    void _load_npz(cnpy::npz_t& npz, bool default_visible) {
        puts("INFO: Loading drawlist npz");
        std::map<std::string,
                 std::pair<std::string /*type*/,
                           cnpy::npz_t /*fields*/>>
            obj_parse_map;

        if (npz.count("data") && npz.count("child") && npz.count("offset") &&
            npz.count("data_dim") && npz.count("data_format") &&
            (npz.count("invradius3") || npz.count("invradius"))) {
            puts("INFO: Loading volume at root of drawlist npz");
            N3Tree tree;
            tree.load_npz(npz);
            trees.push_back(std::move(tree));
            tree.clear_cpu_memory();
        }

        for (const std::pair<std::string, cnpy::NpyArray>& kv : npz) {
            const std::string& fullname = kv.first;
            std::vector<std::string> spl = split_by_2underscore(fullname);
            if (spl.size() == 1) {
                // Skip if not string (for which entire array is
                // considered 1 element in numpy)
                if (kv.second.word_size != kv.second.num_bytes()) continue;
                // Object type
                std::string objtype(kv.second.data_holder.begin(),
                                     kv.second.data_holder.end());
                for (size_t i = 4; i < objtype.size(); i += 4)
                    objtype[i / 4] = std::tolower(objtype[i]);
                objtype.resize(objtype.size() / 4);
                obj_parse_map[spl[0]].first = objtype;
            } else if (spl.size() == 2) {
                // Field
                obj_parse_map[spl[0]].second[spl[1]] = kv.second;
            } else {
                printf(
                    "Mesh load_npz warning: invalid field '%s"
                    "', must be of the form <name>=mesh_type or "
                    "<name>__<field>=val\n",
                    fullname.c_str());
            }
        }

        std::stringstream errs;
        const glm::vec3 DEFAULT_COLOR{1.f, 0.5f, 0.2f};
        for (auto& kv : obj_parse_map) {
            const std::string& obj_name = kv.first;
            const std::string& obj_type = kv.second.first;
            cnpy::npz_t& fields = kv.second.second;

            if (obj_type == "volume") {
                // Volume data
                N3Tree tree;
                tree.load_npz(fields);
                tree.name = obj_name;
                tree.time = map_get_int(fields, "time", -1, errs);
                tree.model_scale = map_get_float(fields, "scale", 1.0f, errs);
                tree.model_translation =
                    map_get_vec3(fields, "translation", glm::vec3{0.f, 0.f, 0.f}, errs);
                tree.model_rotation =
                    map_get_vec3(fields, "rotation", glm::vec3{0.f, 0.f, 0.f}, errs);
                tree.visible = map_get_int(fields, "visible",
                        default_visible, errs) != 0;
                trees.push_back(std::move(tree));
                tree.clear_cpu_memory();
            } else {
                // Mesh/lines/point cloud data
                Mesh me;
                glm::vec3 color = map_get_vec3(fields, "color", DEFAULT_COLOR, errs);
                if (obj_type == "cube") {
                    me = Mesh::Cube(color);
                } else if (obj_type == "sphere") {
                    auto rings = map_get_int(fields, "rings", 15, errs);
                    auto sectors = map_get_int(fields, "sectors", 30, errs);
                    me = Mesh::Sphere(rings, sectors, color);
                } else if (obj_type == "line") {
                    auto a = map_get_vec3(fields, "a", glm::vec3(0.f, 0.f, 0.f), errs);
                    auto b = map_get_vec3(fields, "b", glm::vec3(0.f, 0.f, 1.f), errs);
                    me = Mesh::Line(a, b, color);
                } else if (obj_type == "camerafrustum") {
                    auto focal_length =
                        map_get_float(fields, "focal_length", 1111.0f, errs);
                    auto image_width =
                        map_get_float(fields, "image_width", 800.0f, errs);
                    auto image_height =
                        map_get_float(fields, "image_height", 800.0f, errs);
                    auto z = map_get_float(fields, "z", -0.3f, errs);
                    me = Mesh::CameraFrustum(focal_length, image_width,
                            image_height, z, color);
                    if (fields.count("t")) {
                        auto t = map_get_floatarr(fields, "t", errs);
                        auto r = map_get_floatarr(fields, "r", errs);
                        if (r.size() != t.size() || r.size() % 3) {
                            errs << "camerafrustums r, t have different sizes or "
                                "not "
                                "multiple of 3\n";
                        }
                        const size_t n_verts = me.vert.size() / me.vert_size;
                        const size_t n_reps = t.size() / 3;
                        me.repeat(n_reps);
                        for (int i = 0; i < n_reps; ++i) {
                            const int j = i * 3;
                            glm::vec3 ri{r[j], r[j + 1], r[j + 2]};
                            glm::vec3 ti{t[j], t[j + 1], t[j + 2]};
                            me.apply_transform(ri, ti, n_verts * i, n_verts * (i + 1));
                        }
                        bool connect = map_get_int(fields, "connect", 0, errs) != 0;
                        if (connect) {
                            // Connect camera centers in a trajectory
                            const size_t start_idx = me.faces.size();
                            me.faces.resize(start_idx + (n_reps - 1) * 2);
                            for (int i = 0; i < n_reps - 1; ++i) {
                                me.faces[start_idx + i * 2] = n_verts * i;
                                me.faces[start_idx + i * 2 + 1] = n_verts * (i + 1);
                            }
                        }
                    }
                } else if (obj_type == "image") {
                    auto focal_length =
                        map_get_float(fields, "focal_length", 1111.0f, errs);
                    auto z = map_get_float(fields, "z", -0.3f, errs);
                    auto t = map_get_floatarr(fields, "t", errs);
                    auto r = map_get_floatarr(fields, "r", errs);
                    size_t data_size = fields["data"].num_bytes();
                    const uint8_t* data = map_get_raw(fields, "data", errs);

                    int image_width, image_height, channels;

                    uint8_t* img_dec = stbi_load_from_memory(data, data_size, &image_width, &image_height, &channels, 3);
                    me = Mesh::Image(focal_length, image_width, image_height, z, r, t, img_dec);
                    stbi_image_free(img_dec);
                    if (fields.count("t")) {
                        if (r.size() != t.size() || r.size() % 3) {
                            errs << "camerafrustums r, t have different sizes or "
                                "not "
                                "multiple of 3\n";
                        }
                        const size_t n_verts = me.vert.size() / me.vert_size;
                        const size_t n_reps = t.size() / 3;
                        for (int i = 0; i < n_reps; ++i) {
                            const int j = i * 3;
                            glm::vec3 ri{r[j], r[j + 1], r[j + 2]};
                            glm::vec3 ti{t[j], t[j + 1], t[j + 2]};
                            me.apply_transform(ri, ti, n_verts * i, n_verts * (i + 1));
                        }
                        bool connect = map_get_int(fields, "connect", 0, errs) != 0;
                        if (connect) {
                            // Connect camera centers in a trajectory
                            const size_t start_idx = me.faces.size();
                            me.faces.resize(start_idx + (n_reps - 1) * 2);
                            for (int i = 0; i < n_reps - 1; ++i) {
                                me.faces[start_idx + i * 2] = n_verts * i;
                                me.faces[start_idx + i * 2 + 1] = n_verts * (i + 1);
                            }
                        }
                    }
                } else if (obj_type == "lines") {
                    // Lines
                    auto data = map_get_floatarr(fields, "points", errs);
                    me = Mesh::Lines(data, color);
                    if (fields.count("segs")) {
                        // By default, the points are connected in a single line
                        // i -> i+1 etc
                        // specify this to connect every consecutive pair of indices
                        // 0a 0b 1a 1b 2a 2b ...
                        auto lines = map_get_intarr(fields, "segs", errs);
                        me.faces.resize(lines.size());
                        std::copy(lines.begin(), lines.end(), me.faces.begin());
                    }
                } else if (obj_type == "points") {
                    // Point cloud
                    auto data = map_get_floatarr(fields, "points", errs);
                    me = Mesh::Points(data, color);
                    me.point_size = map_get_float(fields, "point_size", 1.f, errs);
                } else if (obj_type == "mesh") {
                    // Most generic mesh
                    auto data = map_get_floatarr(fields, "points", errs);
                    me = Mesh::Points(data, color);
                    // Face_size = 1: points  2: lines  3: triangles
                    me.face_size = map_get_int(fields, "face_size", 3, errs);
                    if (me.face_size <= 0 || me.face_size > 3) {
                        me.face_size = 3;
                        errs << "Mesh face size must be one of 1,2,3\n";
                    }
                    if (fields.count("faces")) {
                        auto faces = map_get_intarr(fields, "faces", errs);
                        if (faces.size() % me.face_size) {
                            errs << "Faces must have face_size=" << me.face_size
                                << " elements\n";
                        }
                        me.faces.resize(faces.size());
                        std::copy(faces.begin(), faces.end(), me.faces.begin());
                    }
                    if (me.face_size == 3) {
                        me.estimate_normals();
                    }
                } else {
                    errs << "Mesh '" << obj_name << "' has unsupported type '"
                        << obj_type << "'\n";
                    continue;
                }
                if (fields.count("vert_color")) {
                    // Support manual vertex colors
                    auto vert_color = map_get_floatarr(fields, "vert_color", errs);
                    if (vert_color.size() * me.vert_size != me.vert.size() * 3) {
                        errs << "Mesh " << obj_name
                            << " vert_color has invalid size\n";
                        continue;
                    }
                    const float* in_ptr = vert_color.data();
                    float* out_ptr = me.vert.data() + 3;
                    for (int i = 0; i < vert_color.size(); i += 3) {
                        for (int j = 0; j < 3; ++j) {
                            out_ptr[j] = in_ptr[j];
                        }
                        in_ptr += 3;
                        out_ptr += me.vert_size;
                    }
                }
                me.name = obj_name;
                me.time = map_get_int(fields, "time", -1, errs);
                me.model_scale = map_get_float(fields, "scale", 1.0f, errs);
                me.model_translation =
                    map_get_vec3(fields, "translation", glm::vec3{0.f, 0.f, 0.f}, errs);
                me.model_rotation =
                    map_get_vec3(fields, "rotation", glm::vec3{0.f, 0.f, 0.f}, errs);
                me.visible = map_get_int(fields, "visible", default_visible, errs) != 0;
                me.unlit = map_get_int(fields, "unlit", int(me.face_size != 3), errs) != 0;
                me.update();
                meshes.push_back(std::move(me));
            }
        }

        std::string errstr = errs.str();
        if (errstr.size()) {
            printf("Mesh load_npz encountered errors while parsing:\n%s",
                   errstr.c_str());
        }
    }

    void open_drawlist(const std::string& path,
            bool default_visible) {
        auto npz = cnpy::npz_load(path);
        return _load_npz(npz, default_visible);
    }

    void open_drawlist_mem(
            const char* data, uint64_t size,
            bool default_visible) {
        auto npz = cnpy::npz_load_mem(data, size);
        return _load_npz(npz, default_visible);
    }

   private:

    Camera& camera;
    RendererOptions& options;

    int& time;

    GLShader fxaa_program;
    GLFramebuffer fbo_screen, fbo_mesh, fbo_tree;

    std::vector<Mesh>& meshes;
    std::vector<N3Tree>& trees;
};

Renderer::Renderer()
    : impl_(std::make_unique<Impl>(camera, options, meshes, trees, time)) {}

Renderer::~Renderer() {}

void Renderer::render() { impl_->render(); }

void Renderer::add(const N3Tree& tree) { impl_->add(tree); }
void Renderer::add(N3Tree&& tree) { impl_->add(tree); }
void Renderer::clear() { impl_->clear(); }
void Renderer::open_drawlist(const std::string& path, bool default_visible) { impl_->open_drawlist(path, default_visible); }
void Renderer::open_drawlist_mem(const char* data, uint64_t size, bool default_visible) {
    impl_->open_drawlist_mem(data, size, default_visible); }

void Renderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* Renderer::get_backend() { return "Shader"; }

}  // namespace volrend
