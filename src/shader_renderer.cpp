#include "volrend/common.hpp"

// Shader backend only enabled when build with VOLREND_USE_CUDA=OFF
#ifndef VOLREND_CUDA
#include "volrend/renderer.hpp"
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <cstdint>
#include <string>

#ifdef __EMSCRIPTEN__
// WebGL
#include <GLES3/gl3.h>
#else
#include <GL/glew.h>
#endif

#include <GLFW/glfw3.h>

#include "volrend/internal/rt_frag.inl"

namespace volrend {

namespace {

const char* PASSTHRU_VERT_SHADER_SRC =
    R"glsl(#version 300 es
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
)glsl";

const float quad_verts[] = {
    -1.f, -1.f, 0.f, 1.f, -1.f, 0.f, -1.f, 1.f, 0.f, 1.f, 1.f, 0.f,
};

void check_compile_errors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout
                << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
                << infoLog
                << "\n -- ---------------------------------------------------  "
                << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout
                << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
                << infoLog
                << "\n -- --------------------------------------------------- "
                   "-- "
                << std::endl;
        }
    }
}

struct _RenderUniforms {
    GLint cam_transform, cam_focal, cam_reso;
    GLint opt_step_size, opt_backgrond_brightness, opt_stop_thresh,
        opt_sigma_thresh;
    GLint tree_data_tex, tree_child_tex, tree_extra_tex;
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, int max_tries = 4)
        : camera(camera), options(options) {}

    ~Impl() {
        glDeleteProgram(program);
        glDeleteTextures(1, &tex_tree_data);
        glDeleteTextures(1, &tex_tree_child);
        glDeleteTextures(1, &tex_tree_extra);
    }

    void start() {
        if (started_) return;
        resize(0, 0);

        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &tex_max_size);
        int tex_3d_max_size;
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &tex_3d_max_size);
        std::cout << " texture dim limit: " << tex_max_size << "\n";
        std::cout << " texture 3D dim limit: " << tex_3d_max_size << "\n";

        glGenTextures(1, &tex_tree_data);
        glGenTextures(1, &tex_tree_child);
        glGenTextures(1, &tex_tree_extra);

        quad_init();
        shader_init();
        started_ = true;
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        if (tree == nullptr || !started_) return;

        camera._update();
        // FIXME reduce uniform transfers?
        glUniformMatrix4x3fv(u.cam_transform, 1, GL_FALSE,
                             glm::value_ptr(camera.transform));
        glUniform1f(u.cam_focal, camera.focal);
        glUniform2f(u.cam_reso, (float)camera.width, (float)camera.height);
        glUniform1f(u.opt_step_size, options.step_size);
        glUniform1f(u.opt_backgrond_brightness, options.background_brightness);
        glUniform1f(u.opt_stop_thresh, options.stop_thresh);
        glUniform1f(u.opt_sigma_thresh, options.sigma_thresh);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_tree_child);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_tree_data);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, tex_tree_extra);

        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        glBindVertexArray(0);
    }

    void set(N3Tree& tree) {
        start();
        if (tree.capacity > 0) {
            this->tree = &tree;
            upload_data();
            upload_child_links();
            upload_tree_spec();
        }
    }

    void clear() { this->tree = nullptr; }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width > 0) {
            camera.width = width;
            camera.height = height;
        }
        glViewport(0, 0, width, height);
    }

   private:
    void auto_size_2d(size_t size, size_t& width, size_t& height) {
        if (size == 0) {
            width = height = 0;
            return;
        }
        height = std::sqrt(size);
        width = (size - 1) / height + 1;
        if (height > tex_max_size || width > tex_max_size) {
            throw std::runtime_error(
                "Octree data exceeds hardward 2D texture limit\n");
        }
    }

    void upload_data() {
        const GLint data_size =
            tree->capacity * tree->N * tree->N * tree->N * tree->data_dim;
        size_t width, height;
        auto_size_2d(data_size, width, height);
        // FIXME can we remove the copy to float here?
        // Can't seem to get half glTexImage2D to work
        const size_t pad = width * height - data_size;
        tree->data_.data_holder.resize((data_size + pad) * sizeof(float));
        auto* data_ptr_half = tree->data_.data<half>();
        auto* data_ptr = tree->data_.data<float>();
        std::copy_backward(data_ptr_half, data_ptr_half + data_size,
                           data_ptr + data_size);

        glUniform1i(glGetUniformLocation(program, "tree_data_dim"), width);

        glBindTexture(GL_TEXTURE_2D, tex_tree_data);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED,
                     GL_FLOAT, tree->data_.data<half>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Maybe upload extra data
        const size_t extra_sz = tree->extra_.data_holder.size() / sizeof(float);
        if (extra_sz) {
            glBindTexture(GL_TEXTURE_2D, tex_tree_extra);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                         extra_sz / tree->data_format.basis_dim,
                         tree->data_format.basis_dim, 0, GL_RED, GL_FLOAT,
                         tree->extra_.data<float>());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_child_links() {
        const size_t child_size =
            size_t(tree->capacity) * tree->N * tree->N * tree->N;
        size_t width, height;
        auto_size_2d(child_size, width, height);

        const size_t pad = width * height - child_size;
        tree->child_.data_holder.resize((child_size + pad) * sizeof(int32_t));
        glUniform1i(glGetUniformLocation(program, "tree_child_dim"), width);

        glBindTexture(GL_TEXTURE_2D, tex_tree_child);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0,
                     GL_RED_INTEGER, GL_INT, tree->child_.data<int32_t>());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void upload_tree_spec() {
        glUniform1i(glGetUniformLocation(program, "tree.N"), tree->N);
        glUniform1i(glGetUniformLocation(program, "tree.data_dim"),
                    tree->data_dim);
        glUniform1i(glGetUniformLocation(program, "tree.format"),
                    (int)tree->data_format.format);
        glUniform1i(glGetUniformLocation(program, "tree.basis_dim"),
                    tree->data_format.basis_dim);
        glUniform3f(glGetUniformLocation(program, "tree.center"),
                    tree->offset[0], tree->offset[1], tree->offset[2]);
        glUniform3f(glGetUniformLocation(program, "tree.scale"), tree->scale[0],
                    tree->scale[1], tree->scale[2]);
        if (tree->use_ndc) {
            glUniform1f(glGetUniformLocation(program, "tree.ndc_width"),
                        tree->ndc_width);
            glUniform1f(glGetUniformLocation(program, "tree.ndc_height"),
                        tree->ndc_height);
            glUniform1f(glGetUniformLocation(program, "tree.ndc_focal"),
                        tree->ndc_focal);
        } else {
            glUniform1f(glGetUniformLocation(program, "tree.ndc_width"), -1.f);
        }
    }

    void shader_init() {
        // Dummy vertex shader
        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert_shader, 1, &PASSTHRU_VERT_SHADER_SRC, NULL);
        glCompileShader(vert_shader);
        check_compile_errors(vert_shader, "VERTEX");

        // Fragment shader
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

        glShaderSource(frag_shader, 1, &RT_FRAG_SRC, NULL);
        glCompileShader(frag_shader);
        check_compile_errors(frag_shader, "FRAGMENT");

        program = glCreateProgram();
        glAttachShader(program, vert_shader);
        glAttachShader(program, frag_shader);
        glLinkProgram(program);
        check_compile_errors(program, "PROGRAM");

        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        glUseProgram(program);

        u.cam_transform = glGetUniformLocation(program, "cam.transform");
        u.cam_focal = glGetUniformLocation(program, "cam.focal");
        u.cam_reso = glGetUniformLocation(program, "cam.reso");
        u.opt_step_size = glGetUniformLocation(program, "opt.step_size");
        u.opt_backgrond_brightness =
            glGetUniformLocation(program, "opt.background_brightness");
        u.opt_stop_thresh = glGetUniformLocation(program, "opt.stop_thresh");
        u.opt_sigma_thresh = glGetUniformLocation(program, "opt.sigma_thresh");
        u.tree_data_tex = glGetUniformLocation(program, "tree_data_tex");
        u.tree_child_tex = glGetUniformLocation(program, "tree_child_tex");
        u.tree_extra_tex = glGetUniformLocation(program, "tree_extra_tex");
        glUniform1i(u.tree_child_tex, 0);
        glUniform1i(u.tree_data_tex, 1);
        glUniform1i(u.tree_extra_tex, 2);
    }

    void quad_init() {
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glGenVertexArrays(1, &vao_quad);
        glBindVertexArray(vao_quad);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad_verts, (GLvoid*)quad_verts,
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              (GLvoid*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
    }

    Camera& camera;
    RenderOptions& options;

    N3Tree* tree;

    GLuint program = -1;
    GLuint tex_tree_data = -1, tex_tree_child, tex_tree_extra;
    GLuint vao_quad;
    GLint tex_max_size;

    std::string shader_fname = "shaders/rt.frag";

    _RenderUniforms u;
    bool started_ = false;
};

VolumeRenderer::VolumeRenderer(int device_id)
    : impl_(std::make_unique<Impl>(camera, options)) {}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend

#endif
