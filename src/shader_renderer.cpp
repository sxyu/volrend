#include "volrend/common.hpp"

// Shader backend only enabled when build with VOLREND_USE_CUDA=OFF
#ifndef VOLREND_CUDA
#include "volrend/renderer.hpp"
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <cstdint>
#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace volrend {

namespace {

const int MAX_BUFFER_TEXTURE_BLOCKS = 8;

const char* passthru_vert_shader_src = R"glsl(
#version 330
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
    GLint tree_data_tex, tree_child_tex;
};

}  // namespace

struct VolumeRenderer::Impl {
    Impl(Camera& camera, RenderOptions& options, int max_tries = 4)
        : camera(camera), options(options) {
        int i;
        for (i = 0; i < max_tries; ++i) {
            if (std::ifstream(shader_fname)) break;
            shader_fname = "../" + shader_fname;
        }
        if (i == max_tries) {
            throw std::runtime_error(
                "Could not find the compute shader! "
                "Please launch pogram in project directory or a subdirectory.");
        }
        glGenBuffers(MAX_BUFFER_TEXTURE_BLOCKS, tbo_tree_data);
        glGenBuffers(1, &tbo_tree_child);
        glGenTextures(MAX_BUFFER_TEXTURE_BLOCKS, tbo_tex_tree_data);
        glGenTextures(1, &tbo_tex_tree_child);
        resize(0, 0);

        glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &tbo_size_limit);
        std::cout << "Your HW buffer texture texel count limit is "
                  << tbo_size_limit
                  << " items.\n"
                     "We will create up to "
                  << MAX_BUFFER_TEXTURE_BLOCKS
                  << " textures to fit the volume data\n";

        shader_init();
        quad_init();
    }

    ~Impl() {
        glDeleteProgram(program);
        glDeleteTextures(MAX_BUFFER_TEXTURE_BLOCKS, tbo_tex_tree_data);
        glDeleteTextures(1, &tbo_tex_tree_child);
        glDeleteBuffers(4, tbo_tree_data);
        glDeleteBuffers(1, &tbo_tree_child);
    }

    void render() {
        glClear(GL_COLOR_BUFFER_BIT);

        camera._update();
        if (tree == nullptr) return;
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
        glBindTexture(GL_TEXTURE_BUFFER, tbo_tex_tree_child);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, tbo_tree_child);
        glUniform1i(u.tree_child_tex, 0);

        {
            std::vector<int> tmp(MAX_BUFFER_TEXTURE_BLOCKS);
            for (int i = 0; i < tbo_blocks_needed; ++i) {
                glActiveTexture(GL_TEXTURE1 + i);
                glBindTexture(GL_TEXTURE_BUFFER, tbo_tex_tree_data[i]);
                glTexBuffer(GL_TEXTURE_BUFFER, GL_R16F, tbo_tree_data[i]);
                tmp[i] = i + 1;
            }
            glUniform1iv(u.tree_data_tex, MAX_BUFFER_TEXTURE_BLOCKS,
                         tmp.data());
        }

        glBindVertexArray(vao_quad);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)4);
        glBindVertexArray(0);
    }

    void set(const N3Tree& tree) {
        tbo_blocks_needed =
            (tree.capacity * tree.N * tree.N * tree.N * tree.data_dim - 1) /
                tbo_size_limit +
            1;
        if (tbo_blocks_needed > MAX_BUFFER_TEXTURE_BLOCKS) {
            std::cerr << "Renderer set tree FAILED: required memory exceeds "
                         "implementation limit "
                      << size_t(MAX_BUFFER_TEXTURE_BLOCKS) * tbo_size_limit *
                             sizeof(half)
                      << " bytes, please use CUDA or SSBO-based renderer\n";
            return;
        }
        this->tree = &tree;
        upload_data();
        upload_child_links();
        upload_tree_spec();
    }

    void clear() {
        for (int i = 0; i < tbo_blocks_needed; ++i) {
            glBindBuffer(GL_TEXTURE_BUFFER, tbo_tree_data[i]);
            glBufferData(GL_TEXTURE_BUFFER, 0, nullptr, GL_STATIC_READ);
        }
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_tree_child);
        glBufferData(GL_TEXTURE_BUFFER, 0, nullptr, GL_STATIC_READ);
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
        this->tree = nullptr;
    }

    void resize(const int width, const int height) {
        if (camera.width == width && camera.height == height) return;
        if (width > 0) {
            camera.width = width;
            camera.height = height;
        }
        glViewport(0, 0, width, height);
    }

   private:
    void upload_data() {
        const GLint data_size =
            tree->capacity * tree->N * tree->N * tree->N * tree->data_dim;
        // Clearly would be better to use SSBO,
        // but using TBO for WebGL compatibility
        const half* data_ptr = tree->data_ptr();
        GLint prev_size = 0;
        for (int i = 0; i < tbo_blocks_needed; ++i) {
            GLint blksz = std::min(data_size - prev_size, tbo_size_limit);
            glBindBuffer(GL_TEXTURE_BUFFER, tbo_tree_data[i]);
            glBufferData(GL_TEXTURE_BUFFER, blksz * sizeof(half),
                         data_ptr + prev_size, GL_STATIC_READ);
            prev_size += blksz;
        }
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }

    void upload_child_links() {
        const size_t child_size = size_t(tree->capacity) * tree->N * tree->N *
                                  tree->N * sizeof(int32_t);
        glBindBuffer(GL_TEXTURE_BUFFER, tbo_tree_child);
        glBufferData(GL_TEXTURE_BUFFER, child_size,
                     tree->child_.data<int32_t>(), GL_STATIC_READ);

        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }

    void upload_tree_spec() {
        glUniform1i(glGetUniformLocation(program, "tree.N"), tree->N);
        glUniform1i(glGetUniformLocation(program, "tree.data_dim"),
                    tree->data_dim);
        glUniform1i(glGetUniformLocation(program, "tree.sh_order"),
                    tree->sh_order);
        glUniform1i(glGetUniformLocation(program, "tree.n_coe"),
                    (tree->sh_order + 1) * (tree->sh_order + 1));
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
        glShaderSource(vert_shader, 1, &passthru_vert_shader_src, NULL);
        glCompileShader(vert_shader);
        check_compile_errors(vert_shader, "VERTEX");

        // Fragment shader
        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

        std::ifstream shader_ifs(shader_fname);
        shader_ifs.seekg(0, std::ios::end);
        size_t size = shader_ifs.tellg();
        std::string shader_source(size, ' ');
        shader_ifs.seekg(0);
        shader_ifs.read(&shader_source[0], size);

        const GLchar* frag_shader_src = shader_source.c_str();
        glShaderSource(frag_shader, 1, &frag_shader_src, NULL);
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

        glUniform1i(glGetUniformLocation(program, "tbo_size_limit"),
                    tbo_size_limit);

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

    const N3Tree* tree;

    GLuint program = -1;
    GLuint tbo_tree_data[MAX_BUFFER_TEXTURE_BLOCKS], tbo_tree_child;
    GLuint tbo_tex_tree_data[MAX_BUFFER_TEXTURE_BLOCKS], tbo_tex_tree_child;
    GLuint vao_quad;
    GLint tbo_size_limit;
    GLint tbo_blocks_needed;

    std::string shader_fname = "shaders/rt.frag";

    _RenderUniforms u;
};

VolumeRenderer::VolumeRenderer(int device_id)
    : impl_(std::make_unique<Impl>(camera, options)) {
    const GLubyte* vendor = glGetString(GL_VENDOR);  // Returns the vendor
    const GLubyte* renderer =
        glGetString(GL_RENDERER);  // Returns a hint to the model
    printf("GPU: %s %s\n", vendor, renderer);
}

VolumeRenderer::~VolumeRenderer() {}

void VolumeRenderer::render() { impl_->render(); }

void VolumeRenderer::set(const N3Tree& tree) { impl_->set(tree); }
void VolumeRenderer::clear() { impl_->clear(); }

void VolumeRenderer::resize(int width, int height) {
    impl_->resize(width, height);
}
const char* VolumeRenderer::get_backend() { return "Shader"; }

}  // namespace volrend

#endif
